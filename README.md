# 🛡️ PolicyPal – Insurance Policy RAG Assistant

PolicyPal is an AI-powered Retrieval-Augmented Generation (RAG) system designed to analyze insurance policy documents and answer user questions with source-backed explanations.

It supports:

- 📄 PDF policy ingestion
- 🧩 Token-based chunking
- 🧠 OpenAI embeddings
- 🗂️ Chroma vector database indexing
- 🔍 Source-grounded Q&A
- 📊 Structured policy summary extraction
- 💬 Streamlit web interface

---

## 🚀 Features

### ✅ Step 1 - Upload PDF
### ✅ Step 2 – PDF Parsing & Chunking
- Extracts text using `pdfplumber`
- Removes repeated headers & footers
- Token-based chunking with overlap
- Saves structured chunks to JSON

### ✅ Step 3 – Embeddings & Vector Index
- Uses OpenAI Embeddings API
- Stores vectors in persistent Chroma DB
- Supports similarity-based retrieval

### ✅ Step 4 – RAG Question Answering
- Retrieves top-k relevant chunks
- Injects context into LLM
- Enforces:
  - Structured output format
  - Coverage classification
  - Source citation control
  - Anti-hallucination safeguards

### ✅ Step 5 – Intent Classification
Classifies questions into:
- Informational
- Clarification
- Scenario

Each intent enforces a different answer structure.

### ✅ Step 6 – Policy Summary Extraction
Extracts structured insurance fields:

- Coverage Parts
- Limits
- Deductibles / Cost Sharing
- Premium Terms
- Key Exclusions
- Key Conditions

Outputs:
```
storage/policy_summaries.json
```

### ✅ Step 7 – Streamlit UI
Three-panel layout:

| Left | Middle | Right |
|------|--------|-------|
| Pipeline controls | Policy dashboard | AI chat |

---

## 🏗️ Project Structure
```
PolicyPal/
│
├── core.py # Main RAG engine
├── app.py # Streamlit Q&A app
├── app_dashboard.py # Policy summary dashboard
├── step7_build_summary.py # Structured extraction
│
├── requirements.txt
├── setup.bat # Create venv + install deps
├── run.bat # Launch Streamlit
│
├── data/
│ └── sample_policies/
│
└── storage/
  ├── uploads/
  ├── chroma/
  └── policy_summaries.json
```



## ⚙️ Installation

### Option 1 – One-click setup (Windows)

Double-click:
```
setup.bat
```


This will:
- Create `.venv`
- Activate environment
- Install requirements

---

## ▶️ Run the App

Double-click:
```
run.bat
```
Or manually:
```
python -m streamlit run app.py
```






## 🔐 API Key Configuration

You can provide your OpenAI API key by:
```
Enter it in the Streamlit sidebar
```

If no key is provided, the app will look for `OPENAI_API_KEY`.

---

## 🧠 Anti-Hallucination Design

PolicyPal enforces:

- Context-only answering
- Similarity threshold filtering
- Source index correction
- Structured answer templates
- Declarations Page injection for Scenario cases

The system automatically corrects `Sources used:` indices to match actual retrieved chunks.

---

## 📊 Example Scenario Output
```
SUMMARY: If UM/UIM coverage is rejected on your Declarations Page, your policy will not pay for damage caused by an uninsured driver.

Assumptions:

Vehicle must be listed as covered auto

Accident must fall within policy period

Coverage overview:

Bodily Injury Liability: Active

Property Damage Liability: Active

UM/UIM: Rejected

Collision: Not listed on Declarations (not confirmed)

Comprehensive: Not listed on Declarations (not confirmed)

Sources used: [0, 1, 2]
```

---

## 🛠️ Dependencies

See:
```
requirements.txt
```

Main libraries:
- openai
- chromadb
- tiktoken
- pdfplumber
- streamlit

---

## 📌 Future Improvements

- User authentication
- Multi-policy comparison
- Claims workflow assistant
- Production logging
- Cloud deployment (Streamlit Cloud / AWS)

---

## 📄 License

Educational / Demo use.


