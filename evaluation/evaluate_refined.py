import os
import json
import subprocess
from datasets import load_dataset

# === Paths ===
ORIGINAL_DIR = "data/generated"
REFINED_DIR = "data/generated_refined"
RESULTS_FILE = "evaluation/refined_comparison.csv"

# === Load Humaneval dataset ===
dataset = load_dataset("openai_humaneval")["test"]

# === Problems to focus on ===
FOCUS_PROBLEMS = ["problem_2", "problem_10"]

# === Helper: run code with test cases ===
def run_tests(code, tests):
    """Run the code with provided test cases. Return True if all tests pass."""
    with open("temp_eval.py", "w") as f:
        f.write(code + "\n" + tests)
    try:
        subprocess.run(["python", "temp_eval.py"], check=True, timeout=5)
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False

# === Evaluate a folder of results ===
def evaluate_folder(folder_path):
    results = {}
    for fname in os.listdir(folder_path):
        task_id = fname.replace(".json", "")
        if task_id not in FOCUS_PROBLEMS:
            continue
        with open(os.path.join(folder_path, fname), "r") as f:
            data = json.load(f)
        sample_index = int(task_id.split("_")[1]) - 1
        sample = dataset[sample_index]
        tests = sample["test"]

        # Evaluate each strategy
        gpt_cot_pass = run_tests(data["gpt_cot"], tests)
        gpt_self_pass = run_tests(data["gpt_selfdebug"], tests)
        qwen_cot_pass = run_tests(data["qwen_cot"], tests)
        qwen_self_pass = run_tests(data["qwen_selfdebug"], tests)

        results[task_id] = {
            "gpt_cot": gpt_cot_pass,
            "gpt_selfdebug": gpt_self_pass,
            "qwen_cot": qwen_cot_pass,
            "qwen_selfdebug": qwen_self_pass
        }
    return results

# === Load both original and refined results ===
original_results = evaluate_folder(ORIGINAL_DIR)
refined_results = evaluate_folder(REFINED_DIR)

# === Compute pass@k for 10 problems (k=10 for your case) ===
def pass_at_k(results):
    total = len(results)
    passed = 0
    for res in results.values():
        # If any strategy passed, count as a pass for pass@k
        if any(res.values()):
            passed += 1
    return passed / total * 100

# === Save comparison CSV ===
import csv
os.makedirs("evaluation", exist_ok=True)
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Problem", 
        "Strategy", 
        "Original Pass", 
        "Refined Pass"
    ])
    for task_id in FOCUS_PROBLEMS:
        for strategy in ["gpt_cot", "gpt_selfdebug", "qwen_cot", "qwen_selfdebug"]:
            writer.writerow([
                task_id,
                strategy,
                original_results[task_id][strategy],
                refined_results[task_id][strategy]
            ])

# === Print summary ===
print(f" Comparison saved in {RESULTS_FILE}\n")
print("=== Pass@k Percentages ===")
print(f"Original pass@k (any strategy passed): {pass_at_k(original_results):.1f}%")
print(f"Refined pass@k (any strategy passed): {pass_at_k(refined_results):.1f}%")

# Highlight which strategies improved or failed
print("\n=== Detailed Improvements ===")
for task_id in FOCUS_PROBLEMS:
    print(f"\n{task_id}:")
    for strategy in ["gpt_cot", "gpt_selfdebug", "qwen_cot", "qwen_selfdebug"]:
        orig = original_results[task_id][strategy]
        ref = refined_results[task_id][strategy]
        status = "Improved" if (not orig and ref) else "No change" if orig == ref else "Regression"
        print(f"  {strategy}: {orig} -> {ref} ({status})")
