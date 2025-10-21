import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
import openai

# === Configure API keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")

openai.api_key = OPENAI_API_KEY

# === I/O paths ===
EXPORT_FOLDER = Path("data/exported_prompts")
DATA_DIR = Path("data/generated_refined")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Refined prompts for problems 2 and 10 ===
refined_problems = [
    {
        "task_id": "problem_2",
        "prompt": "Problem: Implement a function sum_product(nums: List[int]) -> Tuple[int, int] that returns the sum and product of all integers in the list."
    },
    {
        "task_id": "problem_10",
        "prompt": "Problem: Implement a function filter_by_substring(strings: List[str], substr: str) -> List[str] that returns all strings containing the given substring."
    }
]

# === GPT helper ===
def call_gpt(prompt, cot=True):
    instruction = (
        "You are an expert Python programmer tasked with solving competitive programming problems.Your task: 1. Carefully analyze the problem requirements. 2. Identify potential logical or edge-case errors in the original code. 3. Re-derive the correct algorithm step by step. 4. Write a *fully functional, correct Python solution* that passes all hidden and public tests. ### Requirements - Define all functions exactly as in the starter code. - Match the input/output format expected by the test harness. - Avoid unnecessary prints, explanations, or comments.- Return *only* the code, without markdown formatting or explanations.- The final output must be directly executable Python code."
        if cot
        else "You are an expert Python debugger. Your task: 1. Diagnose the likely cause of the failure. 2. Correct the issue(s) in the code. 3. Return a new *complete, working solution* that passes all tests. ### Requirements - Define all functions exactly as in the starter code. - Produce code that passes the provided and hidden test cases. - Do not include explanations, markdown, or extraneous prints.- Return only the corrected, runnable Python code."
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Python expert generating function implementations."},
            {"role": "user", "content": f"{prompt}\n\n{instruction}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# === Qwen helper via OpenRouter API ===
def call_qwen(prompt, cot=True):
    instruction = (
        "You are an expert Python programmer tasked with solving competitive programming problems.Your task: 1. Carefully analyze the problem requirements. 2. Identify potential logical or edge-case errors in the original code. 3. Re-derive the correct algorithm step by step. 4. Write a *fully functional, correct Python solution* that passes all hidden and public tests. ### Requirements - Define all functions exactly as in the starter code. - Match the input/output format expected by the test harness. - Avoid unnecessary prints, explanations, or comments.- Return *only* the code, without markdown formatting or explanations.- The final output must be directly executable Python code."
        if cot
        else "You are an expert Python debugger. Your task: 1. Diagnose the likely cause of the failure. 2. Correct the issue(s) in the code. 3. Return a new *complete, working solution* that passes all tests. ### Requirements - Define all functions exactly as in the starter code. - Produce code that passes the provided and hidden test cases. - Do not include explanations, markdown, or extraneous prints.- Return only the corrected, runnable Python code."
    )
    data = {
        "model": "qwen/qwen3-14b:free",  
        "messages": [{"role": "user", "content": f"{prompt}\n\n{instruction}"}],
        "temperature": 0.2
    }
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()  # raise error if HTTP code != 200
        resp_json = response.json()
        return resp_json["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        print(f"Qwen API HTTP Error: {e}")
        print("Response:", response.text)
        return "API error"
    except Exception as e:
        print("Qwen call failed:", e)
        return "API error"


# === Main generation loop ===
def generate_refined():
    results = []
    for problem in tqdm(refined_problems, desc="Generating refined code"):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        # GPT outputs
        gpt_cot = call_gpt(prompt, cot=True)
        gpt_selfdebug = call_gpt(prompt, cot=False)

        # Qwen outputs
        qwen_cot = call_qwen(prompt, cot=True)
        qwen_selfdebug = call_qwen(prompt, cot=False)

        entry = {
            "task_id": task_id,
            "gpt_cot": gpt_cot,
            "gpt_selfdebug": gpt_selfdebug,
            "qwen_cot": qwen_cot,
            "qwen_selfdebug": qwen_selfdebug,
        }

        results.append(entry)

        # Save each problem separately
        with open(DATA_DIR / f"{task_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

    # Save all results together
    with open(DATA_DIR / "all_results_refined.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Refined code generation complete! Files saved in {DATA_DIR}/")

if __name__ == "__main__":
    generate_refined()
