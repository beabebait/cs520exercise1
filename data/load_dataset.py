from datasets import load_dataset
import os

def load_humaneval(num_problems=10, export=False):
    dataset = load_dataset("openai_humaneval")
    problems = dataset["test"][:num_problems]

    if export:
        os.makedirs("data/problems", exist_ok=True)
        for i, sample in enumerate(problems):
            with open(f"data/problems/problem{i+1}.txt", "w") as f:
                f.write(sample["prompt"])
        print(f" Exported {num_problems} problems to data/problems/")
    return problems

if __name__ == "__main__":
    problems = load_humaneval(export=True)
    print(f"Loaded {len(problems)} HumanEval problems.")
    print("Example problem:", problems[0]["prompt"])
