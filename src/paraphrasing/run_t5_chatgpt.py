import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

torch.manual_seed(42)

print("Loading sample_1000.csv...")
df = pd.read_csv("sample_1000.csv")
originals = df["original_text"].tolist()

print("Loading humarin T5 ChatGPT paraphraser...")
model_name = "humarin/chatgpt_paraphraser_on_T5_base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5 = T5ForConditionalGeneration.from_pretrained(model_name)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
t5 = t5.to(dev)
t5.eval()
print(f"Using device: {dev}")


def rewrite(txt, in_len=256, out_len=256):
    prompt = f"paraphrase: {txt} </s>"
    enc = tokenizer(prompt, max_length=in_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev)
    with torch.no_grad():
        out = t5.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=out_len,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


print("\nGenerating paraphrases...")
paras = []
for i, txt in enumerate(originals):
    paras.append(rewrite(txt))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(originals)}")

print("\nComputing similarity...")
sim_m = SentenceTransformer("all-MiniLM-L6-v2")
orig_e = sim_m.encode(originals, batch_size=32, convert_to_tensor=True)
para_e = sim_m.encode(paras, batch_size=32, convert_to_tensor=True)
sims = util.cos_sim(orig_e, para_e).diagonal().cpu().tolist()

out_df = df.copy()
out_df["paraphrase"] = paras
out_df["similarity"] = [round(s, 4) for s in sims]
out_df.to_csv("paraphrase_t5_chatgpt.csv", index=False)

avg = sum(sims) / len(sims)
below = sum(1 for s in sims if s < 0.7)
print(f"\nAvg similarity: {avg:.4f}")
print(f"Below 0.7: {below}/{len(sims)}")
print("Saved to paraphrase_t5_chatgpt.csv")
