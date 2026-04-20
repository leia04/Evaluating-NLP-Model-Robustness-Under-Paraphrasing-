import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util

torch.manual_seed(42)

print("Loading sample_1000.csv...", flush=True)
df = pd.read_csv("sample_1000.csv")
originals = df["original_text"].tolist()

print("Loading EN->FR and FR->EN MarianMT models...", flush=True)
en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
fr_en_name = "Helsinki-NLP/opus-mt-fr-en"

en_fr_tok = MarianTokenizer.from_pretrained(en_fr_name)
en_fr_mod = MarianMTModel.from_pretrained(en_fr_name)

fr_en_tok = MarianTokenizer.from_pretrained(fr_en_name)
fr_en_mod = MarianMTModel.from_pretrained(fr_en_name)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
en_fr_mod = en_fr_mod.to(dev)
fr_en_mod = fr_en_mod.to(dev)
en_fr_mod.eval()
fr_en_mod.eval()
print(f"Using device: {dev}", flush=True)


def translate(batch_texts, tk, mdl, in_len=512, out_len=512):
    enc = tk(batch_texts, max_length=in_len, truncation=True,
             padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = mdl.generate(**enc, max_length=out_len, num_beams=4,
                           early_stopping=True)
    return [tk.decode(o, skip_special_tokens=True) for o in out]


print("\nGenerating paraphrases via back-translation...", flush=True)
paras = []
BATCH = 8
for i in range(0, len(originals), BATCH):
    chunk = originals[i:i + BATCH]
    try:
        fr = translate(chunk, en_fr_tok, en_fr_mod)
        back = translate(fr, fr_en_tok, fr_en_mod)
        paras.extend(back)
    except Exception as e:
        print(f"  batch starting at {i} failed: {e}", flush=True)
        paras.extend(chunk)  # fallback to original on error
    if (i + BATCH) % 80 == 0 or (i + BATCH) >= len(originals):
        print(f"  {min(i + BATCH, len(originals))}/{len(originals)}", flush=True)

print("\nComputing similarity...", flush=True)
sim_m = SentenceTransformer("all-MiniLM-L6-v2")
orig_e = sim_m.encode(originals, batch_size=32, convert_to_tensor=True)
para_e = sim_m.encode(paras, batch_size=32, convert_to_tensor=True)
sims = util.cos_sim(orig_e, para_e).diagonal().cpu().tolist()

out_df = df.copy()
out_df["paraphrase"] = paras
out_df["similarity"] = [round(s, 4) for s in sims]
out_df.to_csv("paraphrase_backtranslation.csv", index=False)

avg = sum(sims) / len(sims)
below = sum(1 for s in sims if s < 0.7)
print(f"\nAvg similarity: {avg:.4f}", flush=True)
print(f"Below 0.7: {below}/{len(sims)}", flush=True)
print("Saved to paraphrase_backtranslation.csv", flush=True)
