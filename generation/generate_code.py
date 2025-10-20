import os
import json
from pathlib import Path
from tqdm import tqdm

# === Import LLM SDKs ===
from openai import OpenAI                  # GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# === Configure API keys ===
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='your_key'")

openai_client = OpenAI(api_key=openai_key)

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
    for i, txt_file in enumerate(txt_files):
        with open(txt_file, "r") as f:
            prompt = f.read().strip()
        problems.append({
            "task_id": f"problem_{i+1}",
            "prompt": prompt
        })
    return problems

problems = load_prompts_from_txt()

# === Load Vicuna model ===
vicuna_model_name = "TheBloke/vicuna-7B-1.1-HF"
print(f"Loading Vicuna model: {vicuna_model_name} (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(vicuna_model_name)
model = AutoModelForCausalLM.from_pretrained(vicuna_model_name, device_map="auto", torch_dtype=torch.float16)

def call_vicuna(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=400)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === GPT helper ===
def call_gpt(prompt, cot=True):
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


# === Main generation loop ===
def generate_all():
    results = []
    for i, problem in enumerate(tqdm(problems, desc="Generating code")):
        prompt = problem["prompt"]
        task_id = problem["task_id"]

        # --- GPT outputs ---
        gpt_cot = call_gpt(prompt, cot=True)
        gpt_selfdebug = call_gpt(prompt, cot=False)

        # --- Vicuna outputs ---
        vicuna_cot = call_vicuna(f"Step-by-step solution then final Python code:\n{prompt}")
        vicuna_selfdebug = call_vicuna(f"Final Python code only:\n{prompt}")

        # --- Save results per problem ---
        entry = {
            "task_id": task_id,
            "gpt_cot": gpt_cot,
            "gpt_selfdebug": gpt_selfdebug,
            "vicuna_cot": vicuna_cot,
            "vicuna_selfdebug": vicuna_selfdebug,
        }

        results.append(entry)

        # Save incremental file
        with open(DATA_DIR / f"{task_id}.json", "w") as f:
            json.dump(entry, f, indent=2)

    # Save all results together
    with open(DATA_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Code generation complete! All files saved in data/generated/")


if __name__ == "__main__":
    generate_all()
