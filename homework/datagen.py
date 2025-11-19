import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset, is_answer_valid, DATA_DIR


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # Code help from ChatGPT
    # Fire passes CLI args as strings; make sure types are correct
    oversample = int(oversample)
    temperature = float(temperature)

    base_dataset = Dataset("train")
    model = CoTModel()

    results: List[list] = []
    num_success = 0

    for question, correct_answer in tqdm(base_dataset, desc="Generating RFT data"):
        # Generate multiple diverse CoT samples
        gens = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]  # list[list[str]] -> list[str] for this question

        chosen = None
        for g in gens:
            pred = model.parse_answer(g)
            if pred == pred and is_answer_valid(pred, correct_answer):
                chosen = g
                break

        if chosen is not None:
            num_success += 1
            results.append([question, float(correct_answer), chosen])

    out_path = Path(output_json)
    if not out_path.is_absolute():
        out_path = DATA_DIR / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f)

    total = len(base_dataset)
    print(
        f"Wrote {len(results)} examples to {out_path} "
        f"(success rate {num_success / total:.3f})"
    )


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
