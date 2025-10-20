import os
import json
from pathlib import Path
from tqdm import tqdm
import openai

# === Configure API keys ===
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='your_key'")
openai.api_key = openai_key

# === Qwen (OpenRouter) API Key ===
qwen_key = os.getenv("OPENROUTER_API_KEY")
if not qwen_key:
    raise ValueError("❌ Missing OPENROUTER_API_KEY. Run: export OPENROUTER_API_KEY='your_qwen_key'")
# Set OpenRouter API base
openai.api_base = "https://openrouter.ai/api/v1"

# === I/O paths ===
EXPORT_FOLDER = Path("data/exported_prompts")
DATA_DIR = Path("data/generated")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Load prompts from .txt files ===
def load_prompts_from_txt(folder_path=EXPORT_FOLDER):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"{folder_path} does not exist.")
    
    txt_files = sorted(folder.glob("*.txt"))[:10]  # only 10 problems
    
    problems = []
    for i, txt_file in enumerate(txt_files):
        with open(txt_file, "r") as f:
            prompt = f.read().strip()
        problems.append({
            "task_id": f"problem_{i+1}",
            "prompt": prompt
        })
    return problems

problems = load_prompts_from_txt()

# === GPT helper ===
def call_gpt(prompt, cot=True):
    instruction = "Solve step-by-step before writing the final code." if cot else "Generate only the final code."
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Python expert generating function implementations."},
            {"role": "user", "content": f"{prompt}\n\n{instruction}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# === Qwen helper ===
def call_qwen(prompt, cot=True):
    instruction = "Solve step-by-step before writing the final code." if cot else "Generate only the final code."
    response = openai.ChatCompletion.create(
        model="qwen-7b",
        messages=[
            {"role": "system", "content": "You are a Python expert generating function implementations."},
            {"role": "user", "content": f"{prompt}\n\n{instruction}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

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
