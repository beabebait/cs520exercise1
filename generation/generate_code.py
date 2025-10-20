import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
import json

# === Configure API keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Qwen API

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")

# === I/O paths ===
EXPORT_FOLDER = Path("data/exported_prompts")
DATA_DIR = Path("data/generated")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Load prompts from .txt files ===
def load_prompts_from_txt(folder_path=EXPORT_FOLDER):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"{folder_path} does not exist.")
    
    txt_files = sorted(folder.glob("*.txt"))
    
    problems = []
    for i, txt_file in enumerate(txt_files[:10]):  # Only first 10 prompts
        with open(txt_file, "r") as f:
            prompt = f.read().strip()
        problems.append({
            "task_id": f"problem_{i+1}",
            "prompt": prompt
        })
    return problems

problems = load_prompts_from_txt()

# === OpenAI GPT helper ===
import openai
openai.api_key = OPENAI_API_KEY

def call_gpt(prompt, cot=True):
    instruction = "Solve step-by-step before writing the final code." if cot else "Generate only the final code."
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
    instruction = "Solve step-by-step before writing the final code." if cot else "Generate only the final code."
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
def generate_all():
    results = []
    for problem in tqdm(problems, desc="Generating code"):
        prompt = problem["prompt"]
        task_id = problem["task_id"]

        # --- GPT outputs ---
        gpt_cot = call_gpt(prompt, cot=True)
        gpt_selfdebug = call_gpt(prompt, cot=False)

        # --- Qwen outputs ---
        qwen_cot = call_qwen(prompt, cot=True)
        qwen_selfdebug = call_qwen(prompt, cot=False)

        # --- Save results per problem ---
        entry = {
            "task_id": task_id,
            "gpt_cot": gpt_cot,
            "gpt_selfdebug": gpt_selfdebug,
            "qwen_cot": qwen_cot,
            "qwen_selfdebug": qwen_selfdebug,
        }

        results.append(entry)

        with open(DATA_DIR / f"{task_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

    with open(DATA_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Code generation complete! All files saved in data/generated/")

if __name__ == "__main__":
    generate_all()
