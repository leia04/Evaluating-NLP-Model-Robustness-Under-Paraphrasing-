# Evaluating NLP Model Robustness Under Paraphrasing

---

## Abstract

Text classification models often achieve high accuracy on benchmark datasets, but their robustness to semantically equivalent paraphrases remains underexplored.
This project evaluates how stable model predictions are when input texts are paraphrased using different generation techniques. Using the AG News dataset, we analyze whether models maintain consistent predictions under paraphrased inputs and quantify the degree of performance degradation.

---

## Problem

Modern NLP models are typically evaluated on static test sets, assuming that small variations in phrasing do not affect predictions. However, in real-world scenarios, the same information can be expressed in multiple ways.

This raises a key question: Do text classification models preserve their predictions when the input is paraphrased?

If not, high benchmark accuracy may not reflect true robustness.

---

## Key Findings (Preliminary)

* High semantic similarity between original and paraphrased text does **not always guarantee prediction consistency**
* Different paraphrase methods introduce **varying levels of instability**
* Certain classes (e.g., Business, Science/Tech) show **higher flip rates**
* Traditional models (TF-IDF-based) and transformer models (BERT) exhibit **different robustness behaviors**

> ⚠️ Final results and detailed analysis will be included in the final report (in progress)

---

## Approach

### Dataset

* **AG News** (4 classes: World, Sports, Business, Science/Tech)

### Models

* TF-IDF + Logistic Regression (LR)
* TF-IDF + Linear SVM
* BERT

### Paraphrase Generation

* T5-based paraphrasing
* BART paraphrasing
* Back-translation (EN → FR → EN)

### Evaluation

We evaluate robustness using:

* Accuracy / Macro F1
* Prediction Consistency
* Flip Rate
* Performance Drop (before vs after paraphrasing)

---

## Code

### Repository Structure

```bash
data/
  paraphrase_data.csv
  paraphrase_t5_chatgpt.csv
  paraphrase_bart.csv
  paraphrase_backtranslation.csv

outputs/
  overview/
  lr/
  svm/
  bert/

src/
  models/
  paraphrasing/
  evaluation/
  utils/
```

### How to Run

```bash
pip install -r requirements.txt
```

```bash
# Generate paraphrases
python src/paraphrasing/run_t5_chatgpt.py
python src/paraphrasing/run_bart.py
python src/paraphrasing/run_backtranslation.py
```

```bash
# Compare paraphrase methods
python src/evaluation/compare_paraphrasers.py
```

```bash
# Run robustness evaluation
python src/evaluation/evaluate.py
```

---

## Tools and Libraries

* Python
* PyTorch
* Hugging Face Transformers
* scikit-learn
* pandas / numpy
* matplotlib / seaborn
* sentence-transformers

---

## Contribution

* Designed a pipeline to evaluate **robustness under paraphrasing**
* Implemented multiple **paraphrase generation methods**
* Compared **traditional ML models vs transformer models**
* Proposed evaluation metrics beyond accuracy:

  * prediction consistency
  * flip rate
* Analyzed how semantic-preserving transformations affect model behavior

---

## Outputs

* `overview/`: high-level analysis plots
* `lr/`: confusion matrix for LR
* `svm/`, `bert/`: confusion matrices per paraphrase method

---

## Future Work

* Extend to larger datasets and domains
* Evaluate additional paraphrase models
* Investigate adversarial paraphrasing
* Improve robustness via training strategies

---

## Status

🚧 Final report is currently in progress.
