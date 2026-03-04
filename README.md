# 🛡️ PolicyPal – Insurance Policy RAG Assistant

PolicyPal is an AI-powered Retrieval-Augmented Generation (RAG) system designed to analyze insurance policy documents and answer user questions with source-backed explanations.

It supports:

- 📄 PDF policy ingestion
- 🧩 Token-based chunking
- 🧠 OpenAI embeddings
- 🗂️ Database indexing
- 🔍 Source-grounded Q&A
- 📊 Structured policies comparison
- 💬 Streamlit web interface


---

## 🏗️ Project Structure
```
PolicyPal/
│
├── app.py                     # Streamlit web interface for policy Q&A
├── core.py                    # Core RAG reasoning engine
│
├── compare_policies.py        # Compare coverage between two insurance policies
├── prod_compare.py            # Production-level policy comparison logic
│
├── prod_index.py              # Build vector index from policy documents
├── prod_retriever.py          # Retrieve relevant chunks from vector store
│
├── config.py                  # Global configuration (paths, model settings)
├── policy_paths.py            # Policy file path utilities
│
├── requirements.txt           # Python dependencies
├── setup.bat                  # Create virtual environment & install dependencies
├── run.bat                    # Launch Streamlit application
│
├── data/                      # Insurance policy documents
│   ├── policy_a/
│   ├── policy_b/
│   ├── qa_policies/
│   └── sample_policies/
│
└── storage/                   # Generated artifacts and vector store
    ├── compare_prod/
    ├── qa_parsed_chunks.json
    └── qa_vector_store.json
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


## 🛠️ Dependencies

See:
```
requirements.txt
```

Main libraries:
- openai
- tiktoken
- pdfplumber
- streamlit
- numpy
- scipy

---

## 📌 Future Improvements

- User authentication
- Claims workflow assistant
- Production logging
- Cloud deployment (Streamlit Cloud / AWS)

---

## 📄 License

Educational / Demo use.


