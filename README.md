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

## 🛠 Task 1: Scraping Amharic E-Commerce Telegram Channels

### 🎯 Objective  
Collect e-commerce posts from Amharic Telegram channels to build a raw dataset for entity recognition. Focused on extracting product details, pricing, locations, contacts, and delivery info.

### 🚀 Approach  

- **Tool**: `Telethon` for asynchronous interaction with the Telegram API.  
- **Channels Scraped**: 19 public e-commerce channels (e.g., `@ZemenExpress`, `@AwasMart`, `@belaclassic`, `@Leyueqa`)  
- **Excluded**: `@ethio_brand_collection` (private/inactive)

### 📌 Data Fields  

| Field        | Description |
|--------------|-------------|
| `channel`    | Channel name (e.g., belaclassic) |
| `message_id` | Post ID (e.g., 1645) |
| `text`       | Post content (Amharic/English) |
| `timestamp`  | Date/time of post |
| `views`      | View count |
| `media_path` | Image file path (if available) |

### 📤 Output  

- **File**: `data/raw/scraped.csv` (488 posts)  
- **Images**: Stored in `data/raw/photos/` (~60% posts include images)

### 📊 Results  

- **Total Posts**: 488  
- **Channels Covered**: 19  
- **No missing text**, **No duplicate rows**  
- **Top Channels**:  
  - `belaclassic` (50 posts)  
  - `qnashcom` (40 posts)  
  - `sinayelj` (11 posts)  

### 🔍 Content Patterns  

- **Products**: Mixed-language descriptions (e.g., "PUMA SPIRIT", "የጨቅላ ህፃናት ማቀፊያ")  
- **Prices**:  
  - “ዋጋ፦ 2,000 ብር”  
  - “PRICE 4900 ብር”  
- **Locations**:  
  - Amharic: “መገናኛ”  
  - English: “Mexico”  
- **Contacts**:  
  - Phone numbers: “0944222069”  
  - Telegram handles: “@zemencallcenter”  
- **Delivery**: Terms like “ከነፃ ዲሊቨሪ”, “በሞተረኞች”

---

## ⚠️ Challenges  

| Challenge               | Solution                                               |
|-------------------------|--------------------------------------------------------|
| Missing Channel         | Skipped `@ethio_brand_collection` (private/inactive)  |
| Non-Product Posts       | Detected and filtered greetings/promotions             |
| Language Mix            | Handled mixed Amharic-English formats                  |
| Telegram API Rate Limit | Managed using Telethon's built-in async mechanisms     |

---

## 📂 Key Files  

- `scripts/telegram_scraper.py`: Core scraper logic  
- `run_scraper.py`: CLI runner with configs  
- `data/raw/scraped.csv`: Raw dataset  
- `data/raw/photos/`: Image attachments  

---
