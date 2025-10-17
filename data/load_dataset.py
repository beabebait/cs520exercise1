import json
from pathlib import Path

def load_humaneval(num_problems=10, export=False):
    dataset_path = Path("data/humaneval.jsonl")
    problems = []

    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():  
                problems.append(json.loads(line))
            if len(problems) >= num_problems:  
                break

    if export:
        export_dir = Path("data/exported_prompts")
        export_dir.mkdir(exist_ok=True)
        for i, sample in enumerate(problems):
            prompt_path = export_dir / f"problem_{i+1}.txt"
            with open(prompt_path, "w") as f_out:
                f_out.write(sample["prompt"])

        print(f"Exported {len(problems)} problems to {export_dir}")

    return problems

if __name__ == "__main__":
    problems = load_humaneval(export=True)
    print(f"Loaded {len(problems)} problems.")
    print("Example problem:\n", problems[0]["prompt"])