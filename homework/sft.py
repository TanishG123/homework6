from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        SFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        example = (
            "Example:\n"
            "Question: 2 m in cm\n"
            "Answer (numeric only, inside <answer>...</answer>): <answer>200</answer>\n\n"
            "Now answer a new question.\n"
        )
        return (
            example
            + f"Question: {question}\n"
            "Answer (numeric only, inside <answer>...</answer>): "
        )


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # code help from CHatGPT
    ans_val = float(answer)
    ans_str = f"{ans_val:.3f}".rstrip("0").rstrip(".")

    example = (
        "Example:\n"
        "Question: 2 m in cm\n"
        "Answer (numeric only, inside <answer>...</answer>): <answer>200</answer>\n\n"
        "Now answer a new question.\n"
    )

    question_text = (
        example
        + f"Question: {prompt}\n"
        "Answer (numeric only, inside <answer>...</answer>): "
    )

    return {
        "question": question_text,
        "answer": f"<answer>{ans_str}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "./homework/sft_model",
    **kwargs,
):
    # code help form ChatGPT
    # 1) Initialize base LLM (loads the base SmolLM2 and tokenizer)
    llm = SFTModel()

    # 2) Wrap underlying model with a LoRA adapter
    #    Keep r small enough so adapter stays under the size limit.
    r = 16
    lora_alpha = 64

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(llm.model, lora_config)

    # Avoid bug when using gradient_checkpointing on GPU
    if hasattr(lora_model, "enable_input_require_grads") and torch.cuda.is_available():
        lora_model.enable_input_require_grads()

    # 3) Build datasets
    train_data = Dataset("train")
    valid_data = Dataset("valid")

    train_dataset = TokenizedDataset(llm.tokenizer, train_data, format_example)
    valid_dataset = TokenizedDataset(llm.tokenizer, valid_data, format_example)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to=["tensorboard"],
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=8,   
        learning_rate=3e-5,    
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    # 5) Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # 6) Train and save adapter
    trainer.train()
    trainer.save_model(output_dir)

    # Quick sanity check on valid set
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
