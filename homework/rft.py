from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel

from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset, tokenize
from .data import Dataset


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        # code help from ChatGPT
        return question


def format_example_rft(question: str, correct_answer, cot_text: str) -> dict[str, str]:
    """
    Format one RFT example.

    The CoT text `cot_text` already includes reasoning and the <answer> tag,
    so we just use:
        input:  question
        target: cot_text
    """
    return {
        "question": question.strip(),
        "answer": cot_text.strip(),
    }


def load() -> RFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "./homework/rft_model",
    **kwargs,
):
    # code help from ChatGPT
    # 1) Base RFT LLM
    llm = RFTModel()

    # 2) LoRA config: focus on main projection layers (like your friend)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(llm.model, lora_config)

    # Disable cache during training if possible
    try:
        lora_model.config.use_cache = False
    except Exception:
        pass

    if hasattr(lora_model, "enable_input_require_grads") and torch.cuda.is_available():
        lora_model.enable_input_require_grads()

    # 3) Load RFT dataset if available, otherwise fall back to train data
    try:
        rft_data = Dataset("rft")
        train_source = "rft"
    except Exception:
        rft_data = Dataset("train")
        train_source = "train"

    print(f"Using dataset '{train_source}' with {len(rft_data)} examples")

    train_dataset = TokenizedDataset(llm.tokenizer, rft_data, format_example_rft)

    # Language modeling data collator (causal LM, no masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=llm.tokenizer,
        mlm=False,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 4) TrainingArguments – small batch, more epochs, higher LR, warmup, cosine schedule
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to=["tensorboard"],
        num_train_epochs=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,   # effective batch ~32
        learning_rate=1.5e-4,
        gradient_checkpointing=False,
        save_strategy="epoch",
        logging_steps=5,
        fp16=False,
        warmup_steps=40,
        weight_decay=0.02,
        lr_scheduler_type="cosine",
        save_total_limit=3,
    )

    # 5) Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("===================================")
    print("Launching RFT training...")
    print(f"Training samples  : {len(train_dataset)}")
    print(f"Epochs scheduled  : {training_args.num_train_epochs}")
    print(f"Model output path : {output_dir}")
    print("===================================")

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
