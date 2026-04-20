import random
import pandas as pd
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# reproducibility
random.seed(42)
torch.manual_seed(42)

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
SAMPLE_PER_CLASS = 50

# ---- Step 1: Load AG News and sample ----
print("Loading AG News dataset...")
ag = load_dataset("fancyzhx/ag_news", split="test")

# grab 50 per class
sampled_indices = []
for cls_id in range(4):
    cls_indices = [i for i, ex in enumerate(ag) if ex["label"] == cls_id]
    sampled_indices.extend(random.sample(cls_indices, SAMPLE_PER_CLASS))

random.shuffle(sampled_indices)
subset = ag.select(sampled_indices)

print(f"Sampled {len(subset)} examples: {SAMPLE_PER_CLASS} per class")
for cls_id in range(4):
    count = sum(1 for ex in subset if ex["label"] == cls_id)
    print(f"  {LABEL_MAP[cls_id]}: {count}")

# ---- Step 2: Load T5 paraphrase model ----
print("\nLoading T5 paraphrase model...")
model_name = "humarin/chatgpt_paraphraser_on_T5_base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"
t5_model = t5_model.to(device)
t5_model.eval()
print(f"Model loaded on: {device}")


def paraphrase_t5(text, max_input_len=256, max_output_len=256):
    inp = f"paraphrase: {text} </s>"
    encoded = tokenizer(
        inp, max_length=max_input_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        out = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_output_len,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ---- Step 3: Generate paraphrases ----
print("\nGenerating paraphrases...")
originals = []
paraphrases = []
labels = []

for i, example in enumerate(subset):
    txt = example["text"]
    lbl = example["label"]

    para = paraphrase_t5(txt)

    originals.append(txt)
    paraphrases.append(para)
    labels.append(lbl)

    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(subset)} done")

print(f"Generated {len(paraphrases)} paraphrases.")

# ---- Step 4: Semantic similarity ----
print("\nComputing semantic similarity...")
sim_model = SentenceTransformer("all-MiniLM-L6-v2")

orig_embeddings = sim_model.encode(originals, batch_size=32, convert_to_tensor=True)
para_embeddings = sim_model.encode(paraphrases, batch_size=32, convert_to_tensor=True)

similarities = util.cos_sim(orig_embeddings, para_embeddings).diagonal().cpu().tolist()

avg_sim = sum(similarities) / len(similarities)
print(f"Average cosine similarity: {avg_sim:.4f}")

low_sim = [(i, similarities[i]) for i in range(len(similarities)) if similarities[i] < 0.7]
if low_sim:
    print(f"Warning: {len(low_sim)} paraphrases have similarity < 0.7")
    for idx, score in low_sim[:5]:
        print(f"  idx={idx} sim={score:.3f}")
        print(f"    orig: {originals[idx][:80]}...")
        print(f"    para: {paraphrases[idx][:80]}...")
else:
    print("All paraphrases above 0.7 similarity threshold.")

# ---- Step 5: Save to CSV ----
out_df = pd.DataFrame({
    "original_text": originals,
    "label": labels,
    "label_name": [LABEL_MAP[l] for l in labels],
    "paraphrase": paraphrases,
    "similarity": [round(s, 4) for s in similarities],
})

out_path = "paraphrase_data.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved {len(out_df)} rows to {out_path}")

# quick summary
print("\n--- Summary ---")
print(out_df.groupby("label_name")["similarity"].describe().round(3))
