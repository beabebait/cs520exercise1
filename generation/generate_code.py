import os
from openai import OpenAI
import anthropic
from datasets import load_dataset

# Initialize clients
openai_client = OpenAI(api_key="YOUR_OPENAI_KEY")
anthropic_client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_KEY")

# Define models (different families)
LLM1 = "gpt-4-turbo"
LLM2 = "claude-3-opus-20240229"

# Load prompt templates
def read_prompt(path):
    with open(path, 'r') as f:
        return f.read()

cot_template = read_prompt("prompts/cot_prompt.txt")
sdebug_template = read_prompt("prompts/self_debug_prompt.txt")

# Load dataset
dataset = load_dataset("openai_humaneval")["test"]

# Helper functions
def save_output(problem_id, model_name, strategy, code):
    os.makedirs("generation/generated_code", exist_ok=True)
    filename = f"generation/generated_code/{problem_id}_{model_name}_{strategy}.py"
    with open(filename, "w") as f:
        f.write(code)
    print(f"✅ Saved {filename}")

def query_openai(prompt):
    response = openai_client.chat.completions.create(
        model=LLM1,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def query_claude(prompt):
    response = anthropic_client.messages.create(
        model=LLM2,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# Generate for the first 10 problems
for i, sample in enumerate(dataset[:10]):
    problem_id = f"problem{i+1}"
    problem_text = sample["prompt"]

    # Build prompts
    cot_prompt = cot_template.replace("{{problem_text}}", problem_text)
    sdebug_prompt = sdebug_template.replace("{{problem_text}}", problem_text)

    # GPT
    gpt_cot = query_openai(cot_prompt)
    save_output(problem_id, "GPT", "CoT", gpt_cot)

    gpt_sdebug = query_openai(sdebug_prompt)
    save_output(problem_id, "GPT", "SelfDebug", gpt_sdebug)

    # Claude
    claude_cot = query_claude(cot_prompt)
    save_output(problem_id, "Claude", "CoT", claude_cot)

    claude_sdebug = query_claude(sdebug_prompt)
    save_output(problem_id, "Claude", "SelfDebug", claude_sdebug)
