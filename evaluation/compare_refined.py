import pandas as pd
import os

# === Input files ===
orig_passk_file = "evaluation/pass_at_k_results.csv"
refined_detail_file = "evaluation/refined_comparison.csv"
output_file = "evaluation/refined_pass_at_k_comparison.csv"

# === Load data ===
orig_df = pd.read_csv(orig_passk_file)
refined_df = pd.read_csv(refined_detail_file)

# Extract all strategies present in refined results
strategies = refined_df["Strategy"].unique()

# Prepare refined pass@10 info
refined_passk_summary = []

for strategy in strategies:
    strategy_rows = refined_df[refined_df["Strategy"] == strategy]
    total = len(strategy_rows)
    refined_pass = strategy_rows["Refined Pass"].sum()
    refined_passk = (refined_pass / total) * 100 if total > 0 else 0
    failed_problems = strategy_rows.loc[~strategy_rows["Refined Pass"], "Problem"].tolist()
    refined_passk_summary.append({
        "Model_Strategy": strategy,
        "Refined pass@10 (%)": refined_passk,
        "Example Failures (Refined)": ", ".join(failed_problems) if failed_problems else "None"
    })

refined_summary_df = pd.DataFrame(refined_passk_summary)

# Merge original + refined summaries
merged = pd.merge(orig_df, refined_summary_df, on="Model_Strategy", how="outer")

# Determine change column
def compare(row):
    if pd.isna(row["pass@10 (%)"]) or pd.isna(row["Refined pass@10 (%)"]):
        return "N/A"
    diff = row["Refined pass@10 (%)"] - row["pass@10 (%)"]
    if diff > 0:
        return f"↑ +{diff:.1f}"
    elif diff < 0:
        return f"↓ {diff:.1f}"
    else:
        return "No change"

merged["Change"] = merged.apply(compare, axis=1)

# Reorder columns
merged = merged[
    ["Model_Strategy", "pass@10 (%)", "Refined pass@10 (%)", 
     "Example Failures", "Example Failures (Refined)", "Change"]
]

# === Save ===
os.makedirs("evaluation", exist_ok=True)
merged.to_csv(output_file, index=False)

print(f" Combined pass@k comparison saved to {output_file}")
