# 🛡️ FoodSafe AI - Food Safety Knowledge Assistant

An AI-powered Retrieval-Augmented Generation (RAG) knowledge assistant for food safety regulations, contaminants, hygiene guidelines, and food standards (FSSAI, Codex Alimentarius, WHO).

---

## 🌟 Key Features

- **Hybrid Search**: Combines semantic embeddings (`all-MiniLM-L6-v2`) and lexical matching (`BM25`) for high precision and recall.
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` cross-encoder to accurately rank top context passages.
- **Table-Aware Chunking**: Preserves complex regulatory tables, contaminant limits, and units intact during extraction.
- **Grounded Answering**: Powered by Google Gemini with strict hallucination-prevention constraints.
- **Real-Time Source Attribution**: Clearly displays all referenced source documents with relevance metrics.
- **Modern Dashboard UI**: Custom-designed dark navy & emerald dashboard built on Streamlit with responsive multi-column layout, quick stats, and knowledge base tracking.

---

## 📁 Project Structure

```
├── app.py                     # Streamlit frontend with modern AI dashboard UI
├── main.py                    # Core RAG pipeline (retrieval, reranking, Gemini)
├── documents/                 # PDF knowledge base (FSSAI regulations, Codex standards)
├── assets/                    # Graphic assets and illustrations
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/FoodSafe-AI.git
cd FoodSafe-AI
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 License
MIT License.
