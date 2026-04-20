import pandas as pd
from difflib import SequenceMatcher

files = {
    "T5 ChatGPT (humarin)": "paraphrase_t5_chatgpt.csv",
    "BART (eugenesiow)": "paraphrase_bart.csv",
    "Back-translation (Marian EN-FR-EN)": "paraphrase_backtranslation.csv",
}

print(f"{'Paraphraser':<40} {'Avg Sim':>8} {'<0.7':>6} {'Avg Overlap':>12} {'>90% ovlap':>11} {'>95% ovlap':>11}")
print("-" * 92)

results = {}
for name, path in files.items():
    df = pd.read_csv(path)
    sims = df["similarity"].tolist()
    overlaps = [SequenceMatcher(None, str(r["original_text"]), str(r["paraphrase"])).ratio()
                for _, r in df.iterrows()]

    avg_sim = sum(sims) / len(sims)
    below_07 = sum(1 for s in sims if s < 0.7)
    avg_ovlap = sum(overlaps) / len(overlaps)
    over_90 = sum(1 for o in overlaps if o > 0.9)
    over_95 = sum(1 for o in overlaps if o > 0.95)

    results[name] = {
        "avg_sim": avg_sim,
        "below_07": below_07,
        "avg_overlap": avg_ovlap,
        "over_90": over_90,
        "over_95": over_95,
    }

    print(f"{name:<40} {avg_sim:>8.4f} {below_07:>6} {avg_ovlap:>12.4f} {over_90:>11} {over_95:>11}")

print("\n--- Sample outputs (first 3 rows) ---")
for i in range(3):
    print(f"\n=== Row {i} ===")
    for name, path in files.items():
        df = pd.read_csv(path)
        print(f"  [{name}]")
        print(f"    {df.iloc[i]['paraphrase'][:160]}")
    print(f"  [ORIGINAL] {pd.read_csv(files['T5 ChatGPT (humarin)']).iloc[i]['original_text'][:160]}")
