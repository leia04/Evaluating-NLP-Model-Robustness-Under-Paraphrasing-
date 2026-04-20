import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer, util

torch.manual_seed(42)

print("Loading sample_1000.csv...", flush=True)
df = pd.read_csv("sample_1000.csv")
originals = df["original_text"].tolist()

print("Loading BART paraphraser...", flush=True)
mname = "eugenesiow/bart-paraphrase"
tok = BartTokenizer.from_pretrained(mname)
bart = BartForConditionalGeneration.from_pretrained(mname)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
bart = bart.to(dev)
bart.eval()
print(f"Using device: {dev}", flush=True)


def rewrite_bart(txt, in_len=256, out_len=256):
    enc = tok(txt, max_length=in_len, truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    with torch.no_grad():
        out = bart.generate(
            ids,
            max_length=out_len,
            num_beams=3,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tok.decode(out[0], skip_special_tokens=True)


print("\nGenerating paraphrases...", flush=True)
paras = []
for i, txt in enumerate(originals):
    try:
        paras.append(rewrite_bart(txt))
    except Exception as e:
        print(f"  idx {i} failed: {e}", flush=True)
        paras.append(txt)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(originals)}", flush=True)

print("\nComputing similarity...", flush=True)
sim_m = SentenceTransformer("all-MiniLM-L6-v2")
orig_e = sim_m.encode(originals, batch_size=32, convert_to_tensor=True)
para_e = sim_m.encode(paras, batch_size=32, convert_to_tensor=True)
sims = util.cos_sim(orig_e, para_e).diagonal().cpu().tolist()

out_df = df.copy()
out_df["paraphrase"] = paras
out_df["similarity"] = [round(s, 4) for s in sims]
out_df.to_csv("paraphrase_bart.csv", index=False)

avg = sum(sims) / len(sims)
below = sum(1 for s in sims if s < 0.7)
print(f"\nAvg similarity: {avg:.4f}", flush=True)
print(f"Below 0.7: {below}/{len(sims)}", flush=True)
print("Saved to paraphrase_bart.csv", flush=True)
