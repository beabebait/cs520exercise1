import os
import json
from pathlib import Path
from tqdm import tqdm

# === Import LLM SDKs ===
from openai import OpenAI                # GPT
import google.generativeai as genai      # Gemini

# === Configure API keys ===
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='your_key'")
if not gemini_key:
    raise ValueError("❌ Missing GEMINI_API_KEY. Run: export GEMINI_API_KEY='your_key'")

# OpenAI client
openai_client = OpenAI(api_key=openai_key)

# Gemini configuration
genai.configure(api_key=gemini_key)

# === I/O paths ===
DATA_DIR = Path("data/generated")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS_PATH = Path("data/humaneval_10.jsonl")  # Make sure this exists

# === Load dataset ===
def load_problems(path=PROBLEMS_PATH):
    with open(path) as f:
        return [json.loads(line) for line in f.readlines()]

problems = load_problems()

# === Helper functions ===
def call_gpt(prompt, cot=True):
    """Run GPT model with or without Chain-of-Thought (CoT)."""
    instruction = "Solve step-by-step before writing the final code." if cot else "Generate only the final code."
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Python expert generating function implementations."},
            {"role": "user", "content": f"{prompt}\n\n{instruction}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def call_gemini(prompt, cot=True):
    """Run Gemini model with or without Chain-of-Thought (CoT)."""
    instruction = "Think step-by-step before writing the final code." if cot else "Generate only the final code."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{prompt}\n\n{instruction}")
    return response.text.strip()


# === Main generation loop ===
def generate_all():
    results = []
    for i, problem in enumerate(tqdm(problems, desc="Generating code")):
        prompt = problem["prompt"]
        task_id = problem.get("task_id", f"problem_{i+1}")

        # --- GPT outputs ---
        gpt_cot = call_gpt(prompt, cot=True)
        gpt_selfdebug = call_gpt(prompt, cot=False)

        # --- Gemini outputs ---
        gemini_cot = call_gemini(prompt, cot=True)
        gemini_selfdebug = call_gemini(prompt, cot=False)

        # --- Save results per problem ---
        entry = {
            "task_id": task_id,
            "gpt_cot": gpt_cot,
            "gpt_selfdebug": gpt_selfdebug,
            "gemini_cot": gemini_cot,
            "gemini_selfdebug": gemini_selfdebug,
        }

        results.append(entry)

        # Save incremental file
        with open(DATA_DIR / f"{task_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

    # Save all results together
    with open(DATA_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Code generation complete! All files saved in data/generated/")


if __name__ == "__main__":
    generate_all()
