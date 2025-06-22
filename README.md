# 𓁺 Gebeya Lens 𓁺

## Decoding Ethiopia’s Telegram E-Commerce Chaos with AI/ML

This project extracts structured product data (Product, Price, Location) from Amharic Telegram channels and analyzes vendor performance for financial services.

## 🔧 Tasks Overview
1. Scrape Telegram channels with Amharic vendor posts
2. Preprocess and normalize Amharic text
3. Label subset in CoNLL format
4. Fine-tune NER models (XLM-R, mBERT, etc.)
5. Compare and interpret models (SHAP, LIME)
6. Score vendors using extracted entities + metadata

## 📁 Structure
```
📦 gebeyalens
├── data/
│ ├── raw/
│ ├── processed/
│ └── labeled/
├── notebooks/
├── scripts/
├── models/
├── reports/
├── requirements.txt
├── config.py
└── README.md
```

## 👨‍💻 Tech Stack
- Python, Telethon
- HuggingFace Transformers
- Amharic NLP tools
- SHAP, LIME for interpretability

---

## 🛠 Task 2: Preprocess and Label Subset in CoNLL Format

### 🎯 Objective  
Clean and normalize raw Telegram posts, then annotate a representative sample in CoNLL format for Named Entity Recognition (NER). Target entities include **Product**, **Price**, **Location**, **Contact**, and **Delivery**.

---

### 🚀 Approach  

#### 🔄 Preprocessing

- Loaded `data/raw/scraped.csv` (488 rows from 19 channels) using `pandas`
- Filtered **non-product posts** (e.g., “Eid Mubarak”, “በረፍት ቀንዎ”) via keyword-based exclusion
- Normalized text by:
  - Removing ellipses (`...`)
  - Cleaning up Amharic-English mix
- Tokenized text using **space-based splitting**, effective for both scripts

#### 🎯 Sampling

- Selected **~50 messages** using **stratified sampling** to ensure representation across channels
- Covered high-volume (e.g., `belaclassic`, `AwasMart`) and low-volume channels (e.g., `sinayelj`)

#### 🏷️ Labeling

- Used `scripts/data_labeler.py`, a **semi-automated terminal labeling tool** for NER
- Tagged entities:
  - **Product**: e.g., “PUMA SPIRIT”
  - **Price**: e.g., “4900 ብር”
  - **Location**: e.g., “Mexico”
  - **Contact**: e.g., “0944222069”
  - **Delivery**: e.g., “ከነፃ ዲሊቨሪ”
- Used **pattern-based suggestions** to speed up annotation  
  - Product: Tokens at start or before “ዋጋ፦”/“PRICE”  
  - Price: Number after “ዋጋ፦” or “PRICE”, followed by “ብር”  
  - Contact: Phone patterns (“09”, “+251”), Telegram handles (“@”)  
  - Delivery: Terms like “በሞተረኞች”, “ከነፃ”  
- Allowed **user override** of tags during labeling
- Output saved in `data/labeled/labeled.conll` with metadata (e.g., `message_id`, `channel`, `text`)

---

### 📊 Results  

- **Labeled Dataset**: ~50 messages, covering **all 19 channels**
- **Entity Distribution**:
  - `O`: 50% (non-entity tokens)
  - `Product`: 20%
  - `Price`: 15%
  - `Location`: 10%
  - `Contact/Delivery`: 5%
- **Handled Patterns**:
  - Price: “ዋጋ፦ 2,000 ብር”, “PRICE 4900 ብር”
  - Products: “Dancing Cactus Toy”, “የጨቅላ ህፃናት ማቀፊያ”
  - Sparse terms like “ከነፃ ዲሊቨሪ” successfully captured
- **Quality**: No missing text; sampling ensured diverse and balanced training data

---

### ⚠️ Challenges  

| Challenge            | Resolution                                                   |
|----------------------|--------------------------------------------------------------|
| Non-Product Posts    | Filtered using keywords like “ሱቃችን”, “Eid”                   |
| Language Mix         | Tokenized and normalized Amharic-English blends              |
| Manual Labeling      | Semi-automated process with terminal input + suggestions     |
| Sparse Entities      | Expanded keyword lists for rare Delivery/Location terms      |

---

### 📂 Key Files  

- `scripts/data_labeler.py`: Terminal-based labeling interface  
- `run_labeler.py`: Configurable runner for sampling + labeling  
- `data/labeled/labeled.conll`: Final labeled dataset in CoNLL format  
- `labeler.log`: Logs sampling stats and labeling progress  

---

Next up: **Fine-tuning Amharic NER models** to recognize structured product data in the wild 
