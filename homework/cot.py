from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        #code help from ChatGPT

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that performs unit conversions "
                    "(length, mass, time, etc.). "
                    "For each question, briefly reason step by step, then output ONLY "
                    "the final numeric result inside <answer>...</answer>. "
                    "Do NOT include units or any other text inside <answer>. "
                    "Do NOT write anything after </answer>."
                ),
            },
            {
                "role": "user",
                "content": "How many centimeters are in 2.5 meters?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 meter = 100 centimeters. "
                    "So 2.5 meters = 2.5 * 100 = <answer>250</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How many grams are in 3 kilograms?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 1000 grams. "
                    "So 3 kilograms = 3 * 1000 = <answer>3000</answer>"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # start assistant turn
            tokenize=False,              # we want a string, not token IDs
        )
        return prompt_str


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
