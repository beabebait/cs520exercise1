import os
from pathlib import Path
from openai import OpenAI
import google.generativeai as genai

# -----------------------------
# Configuration
# -----------------------------
NUM_PROBLEMS = 10
PROMPTS_DIR = Path("../data/exported_prompts")
OUTPUT_DIR  = Path("../generated_code")

# LLM Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For Gemini: uses Google Cloud credentials / API key via environment
genai_client = genai.Client()

# Models
LLM1_NAME = "gpt-4"
LLM2_NAME = "gemini-2.5-flash"  # You can pick the version you have access to

# Prompt templates
COT_TEMPLATE       = Path("../prompts/cot_prompt.txt").read_text()
SELFDEBUG_TEMPLATE = Path("../prompts/selfdebug_prompt.txt").read_text()

# -----------------------------
# Helper Functions
# -----------------------------
def read_prompt_file(problem_path: Path):
    with open(problem_path, "r") as f:
        return f.read()

def save_generated_code(problem_id, model_name, strategy, code):
    folder = OUTPUT_DIR / f"{model_name}_{strategy}"
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"problem_{problem_id}.py"
    with open(file_path, "w") as f:
        f.write(code)
    print(f"✅ Saved: {file_path}")

def query_openai(prompt):
    response = openai_client.chat.completions.create(
        model=LLM1_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def query_gemini(prompt):
    response = genai_client.models.generate_content(
        model=LLM2_NAME,
        contents=prompt
    )
    return response.text.strip()

# -----------------------------
# Main Generation Loop
# -----------------------------
def main():
    prompt_files = sorted(list(PROMPTS_DIR.glob("problem_*.txt")))[:NUM_PROBLEMS]
    for i, prompt_file in enumerate(prompt_files, start=1):
        problem_text = read_prompt_file(prompt_file)
        print(f"\n=== Problem {i} ===")

        cot_prompt      = COT_TEMPLATE.replace("{{problem_text}}", problem_text)
        sdebug_prompt   = SELFDEBUG_TEMPLATE.replace("{{problem_text}}", problem_text)

        # OpenAI GPT-4
        print("Generating GPT-4 CoT …")
        code = query_openai(cot_prompt)
        save_generated_code(i, "GPT4", "CoT", code)

        print("Generating GPT-4 SelfDebug …")
        code = query_openai(sdebug_prompt)
        save_generated_code(i, "GPT4", "SelfDebug", code)

        # Google Gemini
        print("Generating Gemini CoT …")
        code = query_gemini(cot_prompt)
        save_generated_code(i, "Gemini", "CoT", code)

        print("Generating Gemini SelfDebug …")
        code = query_gemini(sdebug_prompt)
        save_generated_code(i, "Gemini", "SelfDebug", code)

    print("\n Code generation complete!")

if __name__ == "__main__":
    main()