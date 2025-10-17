from datasets import load_dataset
import json

def download_humaneval():
    dataset = load_dataset("openai_humaneval")

    with open("data/humaneval.jsonl", "w") as f:
        for example in dataset["test"]:
            f.write(json.dumps(example) + "\n")

    print(f"Saved HumanEval with {len(dataset['test'])} problems to data/humaneval.jsonl")

if __name__ == "__main__":
    download_humaneval()