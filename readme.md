# **Building Practical Intent Classification and NER Pipelines on the SNIPS Dataset**

This project evaluates the **SNIPS 2018 Spoken Language Understanding (SLU) dataset**, a benchmark built for modeling natural user commands in voice-assistant applications. SNIPS contains short, task-oriented utterances covering everyday assistant use cases such as playing music, adding items to playlists, booking restaurants, rating books, and retrieving weather information.

The project focuses on the two core components of SLU systems:

* **Intent Classification** — identifying the user's goal (e.g., *PlayMusic*, *BookRestaurant*, *GetWeather*)
* **Slot Filling (NER)** — extracting key arguments from the utterance (e.g., *artist*, *restaurant name*, *location*)

Beyond baseline modeling, the project emphasizes **dataset cleanup, duplicate removal, leakage-free train/val/test splits, and reproducible few-shot experiments**, reflecting the constraints of building reliable, real-world assistant pipelines.

# ⭐ **Results Summary**

| Task                 | Model              | Metric                     | Score     |
|----------------------|--------------------|---------------------------|-----------|
| Intent Classification| BERT-base          | Accuracy                  | **98.56%**|
| Slot Filling (NER)   | DistilBERT         | Entity-level F1           | **94.54%** |
| Few-shot Intent (≈70 samples/intent) | BERT-base         | Accuracy | **97.84%**|

# **Dataset Citation**

> [SONOS NLU Benchmark - GitHub](https://github.com/sonos/nlu-benchmark)

# **Introductory Paper**

> Coucke A. et al., “Snips Voice Platform: an embedded spoken language understanding system for private-by-design voice interfaces.” 2018.
> [https://arxiv.org/abs/1805.10190](https://arxiv.org/abs/1805.10190)

# **Dataset Overview**

The SNIPS dataset contains user utterances annotated for two SLU tasks:

### **1. Intent Classification**

* 13784 instances for `train` and 700 instances for `test`.
* 7 intent labels
* Highly balanced
* Each utterance has exactly one intent
  (e.g., `BookRestaurant`, `SearchCreativeWork`, `PlayMusic`, etc.)

### **2. Slot Filling (NER)**

Utterances contain entity spans such as:

* `object_type`
* `artist`
* `genre`
* `country`
* `poi_type`
* etc.

Entity frequencies are **very imbalanced** (e.g., `object_type`: 3341 vs `genre`: 147), which heavily influences NER difficulty.


# **PART I — Intent Classification**

## **1. Summary**

This section covers the **intent classification pipeline** using **BERT-base**, evaluated under:

* **Few-shot settings:** 10, 20, 50, 70, 100 samples per intent
* **Full-data setting**

All experiments use the **cleaned, leakage-free** dataset.


## **2. Data Cleaning & Preprocessing**

The intent preprocessing pipeline includes:

* Whitespace normalization
* Removal of internal duplicates within each split (train/val/test)
* Removal of cross-split duplicates to prevent leakage (train ↔ val ↔ test)
* Stratified train–validation split to preserve label distribution
* Consistent text formatting across all few-shot and full-data settings

This ensures clean, balanced, leakage-free evaluation across all experiments.

## **3. Modeling Approach**

* **Model:** BERT-base (fine-tuned end-to-end)
* **Head:** Single linear classification layer on `[CLS]`
* **Training:** Cross-entropy loss, AdamW optimizer, linear LR scheduler with warmup, early stopping (patience = 3)
* **Validation:** Best model selected using validation weighted F1
* **Inputs:** Cleaned, tokenized utterances (max length = 44)
* **Evaluation:** Accuracy, weighted F1, confusion matrix, error analysis


## **4. Results**

| Training Size | Accuracy | Weighted F1 | Errors (out of 697) |
| - | -- | -- | - |
| **10-shot**   | 82.21%   | 81.13%      | 124                 |
| **20-shot**   | 95.69%   | 95.66%      | 30                  |
| **50-shot**   | 96.70%   | 96.72%      | 23                  |
| **70-shot**   | 97.84%   | 97.86%      | 15                  |
| **100-shot**  | 97.84%   | 97.86%      | 15                  |
| **Full**      | 98.56%   | 98.57%      | 10                  |

### **Confusion Matrix of the Full Training Size Model**

![image info](output.png)

### **Observations**

* Strong scaling with labeled data, diminishing returns after ~70 examples per intent.
* Remaining errors mostly come from semantically close intent pairs:

  * *SearchCreativeWork* ↔ *SearchScreeningEvent*
* A small set of utterances is genuinely ambiguous/underspecified.


# **PART II — Slot Filling (NER)**

## **1. Summary**

Slot filling extracts key arguments from user commands (e.g., *artist*, *track*, *restaurant name*, *location*), making it the component that turns an intent into an actionable request. The SNIPS dataset provides entity spans at the character level, requiring careful reconstruction, tagging, and subword alignment.

This project builds a clean, reproducible slot-filling pipeline using DistilBERT with BIO tagging, achieving an **entity-level F1 of 94.54%**.

## **2. Data Cleaning & Preprocessing**

SNIPS provides character-level entity spans inside JSON files, which are not directly usable for NER.
Entity frequencies are also **highly imbalanced** (e.g., `object_type`: 3341 occurrences vs. `genre`: 147), making it important to preserve rare entity spans accurately during reconstruction.

To produce a clean, consistent BIO-tagged dataset, the pipeline performs:

* Rebuilding full utterances from raw JSON
* Converting character-level spans into word-level BIO tags
* Handling whitespace inconsistencies and unnamed tokens
* Performing a **stratified train–validation split by intent** to retain intent diversity
* Ensuring consistent formatting across train/validation/test sets

This produces a **clean, word-aligned BIO dataset** suitable for BERT-style subword tokenization.

## **3. Modeling Approach**

* **Model:** DistilBERT fine-tuned for token classification
* **Labeling scheme:** BIO tags aligned to subword tokens
* **Subword alignment:** labels expanded to all wordpiece tokens
* **Loss masking:** `-100` used for padding and special tokens
* **Optimization:** AdamW, linear LR schedule, warmup, early stopping
* **Evaluation:** Entity-level F1 using `seqeval` - chosen because entity frequencies in SNIPS are highly imbalanced, making accuracy uninformative. Token-level accuracy is reported only as a secondary reference.

The entire system is implemented using HuggingFace’s `Trainer`, with explicit control over alignment and masking logic.

## **4. Results**

* **Entity-level F1:** **94.54%**

Performance varies across entity types due to SNIPS’ natural imbalance:

* **High-frequency entities** (e.g., `object_type`, `playlist`) → strong and stable F1
* **Sparse entities** (e.g., `genre`, `service`) → lower stability and wider variance

A complete per-entity breakdown and failure analysis are provided in the notebook.


# **How to Use This Repository**

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the Intent Classification pipeline**

Execute notebooks **in order**:

```
01_build_snips_intent_dataset.ipynb  
02_preprocessing_intent.ipynb  
03_eda_intent.ipynb  
04_train_and_evaluate_full_intent.ipynb  
05_evaluate_intent.ipynb  
06_prototype_intent.ipynb
```

This builds the intent dataset, preprocesses it, trains the model (few-shot + full), evaluates it, and produces the confusion matrix.

3. **Run the NER pipeline**

Execute notebooks **in order**:

```
07_build_snips_ner_dataset.ipynb  
08_train_and_evaluate_full_ner.ipynb  
09_evaluate_ner.ipynb  
10_prototype_ner.ipynb
```

This constructs the NER dataset (BIO word-level), aligns labels to subwords, trains DistilBERT, and evaluates using entity-level F1.