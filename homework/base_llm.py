from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here

        This would be a default implementation applies a basic chat template.
        Override this in subclasses for different behavior (e.g., SFT/RFT models should return raw questions).
        """
        # Code help from ChatGPT
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that continues text in a natural way.",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # Use the model's chat template; do NOT tokenize here
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - apply format_prompt to the input prompt
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        # raise NotImplementedError()
        return self.batched_generate([prompt])[0]   # If you feel confident, just use this line of code and move straight to batched_generate.

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: None = None,
        temperature: float = 0,
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - apply format_prompt to the input prompts
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        # code help from ChatGPT
        from tqdm import tqdm

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM Running on Micro Batches {micro_batch_size}",
                )
                for r in self.batched_generate(
                    prompts[idx : idx + micro_batch_size],
                    num_return_sequences,
                    temperature,
                )
            ]

        # 1) Format prompts
        formatted_prompts = [self.format_prompt(p) for p in prompts]

        # 2) Left padding as required
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3) Number of sequences per prompt
        num_seqs = 1 if num_return_sequences is None else num_return_sequences

        gen_kwargs = {
            "max_new_tokens": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": num_seqs,
        }

        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **gen_kwargs,
            )

        # 4) Keep only generated tokens, masking out the inputs
        #    (exactly as suggested in the assignment)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # 5) Shape the return according to num_return_sequences
        if num_return_sequences is None:
            # One generation per prompt
            # len(decoded) == len(prompts)
            return decoded

        # Multiple generations per prompt: group into list[list[str]]
        grouped: list[list[str]] = []
        for i in range(len(prompts)):
            start = i * num_seqs
            end = (i + 1) * num_seqs
            grouped.append(decoded[start:end])

        return grouped

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        generations = self.batched_generate(questions)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
