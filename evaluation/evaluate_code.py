import os
import subprocess
import csv
from datasets import load_dataset

dataset = load_dataset("openai_humaneval")["test"]
output_dir = "generation/generated_code"
results_file = "evaluation/results.csv"

def run_tests(code, tests):
    with open("temp_eval.py", "w") as f:
        f.write(code + "\n" + tests)
    try:
        subprocess.run(["python", "temp_eval.py"], check=True, timeout=5)
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False

rows = [("Problem", "Model", "Strategy", "Pass")]

for fname in os.listdir(output_dir):
    path = os.path.join(output_dir, fname)
    with open(path, "r") as f:
        code = f.read()

    problem_index = int(fname.split("_")[0].replace("problem", "")) - 1
    sample = dataset[problem_index]
    tests = sample["test"]

    passed = run_tests(code, tests)
    model = "GPT" if "GPT" in fname else "Claude"
    strategy = "CoT" if "CoT" in fname else "SelfDebug"
    rows.append((fname, model, strategy, passed))

# Save CSV
os.makedirs("evaluation", exist_ok=True)
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"✅ Evaluation complete — results in {results_file}")
