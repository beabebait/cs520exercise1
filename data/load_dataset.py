import json
from pathlib import Path

def load_humaneval(export=False):
    dataset_path = Path("data/humaneval.jsonl")
    problems = []

    # Read and parse JSONL (each line is a separate problem dict)
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                problems.append(json.loads(line))

    if export:
        export_dir = Path("data/exported_prompts")
        export_dir.mkdir(exist_ok=True)
        for i, sample in enumerate(problems):
            prompt_path = export_dir / f"problem_{i}.txt"
            with open(prompt_path, "w") as f_out:
                f_out.write(sample["prompt"])

    return problems

if __name__ == "__main__":
    problems = load_humaneval(export=True)
    print(f"Loaded {len(problems)} problems.")
    print("Example problem:\n", problems[0]["prompt"])