import os
import json
import subprocess
import csv
from tqdm import tqdm
from datasets import load_dataset

# === Configuration ===
OUTPUT_DIR = "data/innovation_outputs" 
RESULTS_FILE = "evaluation/innovation_pass_at_k_results.csv"
K = 10  # pass@k

# Load HumanEval dataset
dataset = load_dataset("openai_humaneval")["test"]

# Helper: run code against test
def run_tests(code, tests):
    with open("temp_eval.py", "w") as f:
        f.write(code + "\n" + tests)
    try:
        subprocess.run(["python", "temp_eval.py"], check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

# Gather all JSON files
json_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("problem_") and f.endswith(".json")])

# Initialize results
summary = {}

for fname in tqdm(json_files, desc="Evaluating problems"):
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "r") as f:
        entry = json.load(f)

    problem_index = int(entry["task_id"].split("_")[1]) - 1
    sample = dataset[problem_index]
    tests = sample["test"]

    for model_key in ["gpt_rcotd", "qwen_rcotd"]:  # Only RCOTD outputs
        code = entry.get(model_key)
        if not code or code.strip() == "" or code == "API error":
            passed = False
        else:
            passed = run_tests(code, tests)

        if model_key not in summary:
            summary[model_key] = {"passed_count": 0, "failures": []}

        if passed:
            summary[model_key]["passed_count"] += 1
        else:
            if len(summary[model_key]["failures"]) < 2:  # track up to 2 failing cases
                summary[model_key]["failures"].append(entry["task_id"])

# Compute pass@k percentages
rows = [("Model_Strategy", "pass@10 (%)", "Example Failures")]
for model_key, data in summary.items():
    percent = (data["passed_count"] / K) * 100
    rows.append((model_key, f"{percent:.1f}", ", ".join(data["failures"])))

# Save CSV
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

# Print summary
print("pass@10 evaluation complete")
for row in rows[1:]:
    print(f"{row[0]}: pass@10 = {row[1]}%, failures: {row[2]}")

