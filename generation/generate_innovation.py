import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
import time

# === API Keys ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini key from Google AI Studio
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For Qwen

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")

# === Paths ===
EXPORT_FOLDER = Path("data/exported_prompts")
DATA_DIR = Path("data/innovation_outputs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Load exported problems ===
def load_prompts_from_txt(folder_path=EXPORT_FOLDER):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"{folder_path} does not exist.")
    
    txt_files = sorted(folder.glob("*.txt"))
    problems = []
    for i, txt_file in enumerate(txt_files[:10]):  # first 10 problems
        with open(txt_file, "r") as f:
            prompt = f.read().strip()
        problems.append({"task_id": f"problem_{i+1}", "prompt": prompt})
    return problems

problems = load_prompts_from_txt()

# === Gemini helper ===
def call_gemini(prompt, temperature=0.2, max_output_tokens=500):
    """Call Gemini 2.5 (Flash) model safely with fallback parsing."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        # Safely extract text
        candidates = result.get("candidates", [])
        if not candidates:
            print(" Gemini returned no candidates.")
            return "API error"

        # Try multiple possible JSON paths
        content = candidates[0].get("content", {})
        parts = content.get("parts")
        if parts and isinstance(parts, list) and "text" in parts[0]:
            return parts[0]["text"].strip()

        # Alternate structure (Gemini sometimes puts output in .get("text") directly)
        text = candidates[0].get("text")
        if text:
            return text.strip()

        print(" Unexpected Gemini response format:", json.dumps(result, indent=2))
        return "API error"

    except Exception as e:
        print("Gemini API Error:", e)
        return "API error"

# === Qwen helper (OpenRouter) ===
def call_qwen(prompt):
    data = {
        "model": "qwen/qwen3-14b:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Qwen API Error:", e)
        return "API error"

# === RCOTD Role Chain ===
def make_rcotd_prompt(problem_text):
    return (
        "## Role: Coder\n"
        "You are a Python expert tasked to solve the problem step by step and write the code.\n"
        "Think step-by-step carefully before coding.\n"
        "Return only valid Python code, no markdown.\n\n"
        f"{problem_text}"
    )

def make_refiner_prompt(draft_code, review_text="No major issues.", debug_text="No debug hints."):
    return (
        "# Refiner role\n"
        "Combine draft code, reviewer feedback, and debugger suggestions to fix any bugs.\n"
        "Follow these requirements:\n"
        "1. Define all functions exactly as in the starter code.\n"
        "2. Match input/output format expected by the test harness.\n"
        "3. Avoid unnecessary prints, explanations, or comments.\n"
        "4. Return only the final code, no markdown or extra formatting.\n\n"
        f"Draft Code:\n{draft_code}\n"
        f"Reviewer Feedback:\n{review_text}\n"
        f"Debugger Suggestions:\n{debug_text}"
    )

# === Generate RCOTD Refined Outputs ===
def generate_rcotd_refined():
    results = []

    for problem in tqdm(problems, desc="Generating RCOTD code"):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        # --- Step 1: Coder draft ---
        coder_prompt = make_rcotd_prompt(prompt)
        gemini_draft = call_gemini(coder_prompt)
        qwen_draft = call_qwen(coder_prompt)

        # --- Step 2: Reviewer + Debugger roles (simple simulation) ---
        review_text = "Ensure correctness, clarity, and compliance with starter code."
        debug_text = "Check boundary conditions and example test cases."

        # --- Step 3: Refiner ---
        gemini_refined = call_gemini(make_refiner_prompt(gemini_draft, review_text, debug_text))
        qwen_refined = call_qwen(make_refiner_prompt(qwen_draft, review_text, debug_text))

        entry = {
            "task_id": task_id,
            "gemini_draft": gemini_draft,
            "gemini_refined": gemini_refined,
            "qwen_draft": qwen_draft,
            "qwen_refined": qwen_refined
        }
        results.append(entry)

        with open(DATA_DIR / f"{task_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

        time.sleep(2)  # short delay to avoid any minor rate limits

    with open(DATA_DIR / "rcotd_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("RCOTD generation complete! Outputs saved in data/innovation_outputs/")

if __name__ == "__main__":
    generate_rcotd_refined()
