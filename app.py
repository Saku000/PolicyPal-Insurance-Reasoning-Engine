# app.py  (Stable State + 2 Full-Page Tabs: Q&A / Compare)
# Update (per request):
# - In Q&A tab: add PDF upload + "Build Q&A Index" (one-click)
# - Q&A uses its own isolated index files:
#     storage/qa_parsed_chunks.json
#     storage/qa_vector_store.json
#   so it won't interfere with Compare (which uses storage/compare_prod/*)
# - Compare behavior remains unchanged.

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine

from policy_paths import PROJECT_ROOT, POLICY_A_DIR, POLICY_B_DIR, COMPARE_DIR
from prod_index import build_policy_index
from prod_compare import compare_policies_prod


# =========================
# Helpers
# =========================
def get_api_key() -> Optional[str]:
    k = (st.session_state.get("api_key", "") or "").strip()
    return k or os.getenv("OPENAI_API_KEY")


def open_folder(path: Path):
    abs_path = str(path.resolve())
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(abs_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
    except Exception as e:
        st.error(f"Cannot open folder: {e}")


def require_modules_ok() -> List[str]:
    missing = []
    try:
        import core  # noqa: F401
    except Exception as e:
        missing.append(f"core.py import failed: {e}")
    try:
        import prod_index  # noqa: F401
    except Exception as e:
        missing.append(f"prod_index.py import failed: {e}")
    try:
        import prod_compare  # noqa: F401
    except Exception as e:
        missing.append(f"prod_compare.py import failed: {e}")
    try:
        import prod_retriever  # noqa: F401
    except Exception as e:
        missing.append(f"prod_retriever.py import failed: {e}")
    return missing


# =========================
# Q&A index (isolated paths)
# =========================
QA_PDF_DIR = PROJECT_ROOT / "data" / "qa_policies"
QA_PDF_DIR.mkdir(parents=True, exist_ok=True)

QA_CHUNKS_PATH = PROJECT_ROOT / "storage" / "qa_parsed_chunks.json"
QA_VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "qa_vector_store.json"
QA_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)


def qa_build_index(api_key: str) -> Dict[str, Any]:
    """
    One-click Q&A indexing:
    - Step3 parse+chunk PDFs from data/qa_policies -> storage/qa_parsed_chunks.json
    - Step4 embed+save manual vector store -> storage/qa_vector_store.json
    """
    import core

    # Step 3 (custom input/output)
    payload = core.step3_ingest_to_json(input_dir=str(QA_PDF_DIR), output_path=str(QA_CHUNKS_PATH))
    chunks = payload.get("chunks", []) or []
    if not chunks:
        raise RuntimeError("No chunks were produced. Are your PDFs text-based (not scanned)?")

    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]

    metadatas = []
    for c in chunks:
        p1, p2 = core.extract_page_range(c.get("text", ""))
        metadatas.append(
            {
                "doc_name": str(c.get("doc_name", "")),
                "page_start": int(p1) if p1 is not None else -1,
                "page_end": int(p2) if p2 is not None else -1,
            }
        )

    # Step 4 (custom store path)
    embeddings = core.embed_texts_openai(docs, api_key=api_key)

    store = {
        "ids": ids,
        "documents": docs,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }

    with open(QA_VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)

    return {
        "num_pdfs": payload.get("num_pdfs", 0),
        "num_chunks": payload.get("num_chunks", len(chunks)),
        "chunks_path": str(QA_CHUNKS_PATH),
        "store_path": str(QA_VECTOR_STORE_PATH),
    }


def qa_query_store(query: str, api_key: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Manual cosine retrieval against Q&A store (isolated).
    Output matches the structure core._build_context_from_retrieval expects.
    """
    import core

    if not query.strip():
        raise ValueError("Empty question.")
    if not QA_VECTOR_STORE_PATH.exists():
        raise FileNotFoundError("Q&A vector store not found. Please build the Q&A index first.")

    with open(QA_VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    client = core._openai_client(api_key)
    q_emb = client.embeddings.create(model=core.EMBEDDING_MODEL, input=query).data[0].embedding

    dists = [float(cosine(q_emb, emb)) for emb in store["embeddings"]]
    idx = np.argsort(dists)[:top_k]

    return {
        "ids": [[store["ids"][i] for i in idx]],
        "documents": [[store["documents"][i] for i in idx]],
        "metadatas": [[store["metadatas"][i] for i in idx]],
        "distances": [[dists[i] for i in idx]],
    }


def qa_rag_answer(question: str, api_key: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Q&A RAG answer using isolated Q&A store.
    - Uses core intent router + context builder when available
    - Does NOT inject special "declarations" logic (keeps Q&A stable for arbitrary uploads)
    """
    import core

    retrieval = qa_query_store(question, api_key=api_key, top_k=top_k)
    if not retrieval.get("ids") or not retrieval["ids"][0]:
        return {"intent": "Informational", "answer": "No relevant text found.", "evidence": [], "sources": []}

    intent = core.classify_intent(question, api_key=api_key) if getattr(core, "ENABLE_INTENT_ROUTER", False) else "Informational"
    context_text, sources, evidence = core._build_context_from_retrieval(retrieval)

    instruction = core.build_answer_instruction(intent)
    client = core._openai_client(api_key)

    user_msg = (
        f"Question: {question}\n\n"
        f"Context: {context_text}\n\n"
        f"Rules: Use ONLY context. Follow structure:\n{instruction}"
    )

    resp = client.chat.completions.create(
        model=core.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are PolicyPal. Use provided context only."},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    answer = core._enforce_sources_used_line((resp.choices[0].message.content or "").strip(), sources[:top_k])
    return {"intent": intent, "answer": answer, "evidence": evidence[:top_k], "sources": sources[:top_k]}


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="PolicyPal", layout="wide")


# =========================
# Session state init
# =========================
def init_state():
    defaults = {
        # global
        "api_key": "",
        # compare
        "a_name": "Policy A",
        "b_name": "Policy B",
        "a_store": "",
        "b_store": "",
        "compare_last_answer": "",
        "compare_last_question": "",
        "compare_question": "",
        "force_refresh_summaries": False,
        "indexes_built": False,
        # Q&A
        "qa_question": "",
        "qa_last_answer": "",
        "qa_last_sources": [],
        "qa_index_ready": False,
        "qa_last_build_info": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# =========================
# Sidebar
# =========================
st.sidebar.title("Settings")
st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
    key="api_key",
    help="If empty, the app will use environment variable OPENAI_API_KEY.",
)

missing = require_modules_ok()
if missing:
    st.sidebar.error("Some modules failed to import")
    for m in missing:
        st.sidebar.code(m)

st.sidebar.divider()
st.sidebar.caption(f"Project root: {PROJECT_ROOT}")
st.sidebar.caption(f"Compare cache dir: {COMPARE_DIR}")

st.sidebar.checkbox(
    "Force refresh summaries (ignore cache)",
    value=st.session_state.force_refresh_summaries,
    key="force_refresh_summaries",
    help="If you changed PDFs but kept the same policy name, turn this on for one run.",
)


# =========================
# TOP NAV "TAGS" (Tabs)
# =========================
tab_qa, tab_compare = st.tabs(["Q&A", "Compare"])


# =========================
# Q&A PAGE
# =========================
with tab_qa:
    st.title("PolicyPal — Production-style Policy Q&A")

    st.caption(
        "Upload PDFs into the Q&A workspace and build an index with one click. "
        "Q&A uses an isolated store (storage/qa_vector_store.json) and does not affect Compare."
    )

    # Full-page layout for Q&A
    left, right = st.columns([1.15, 1.85], gap="large")

    with left:
        st.subheader("1) Upload PDFs")
        uploaded = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="qa_uploader",
        )

        ucol1, ucol2 = st.columns(2)
        with ucol1:
            if st.button("Open Q&A Folder", use_container_width=True, disabled=bool(missing), key="btn_open_qa_folder"):
                open_folder(QA_PDF_DIR)

        with ucol2:
            if st.button("Clear Q&A PDFs", use_container_width=True, disabled=bool(missing), key="btn_clear_qa_pdfs"):
                # delete PDFs in QA folder
                removed = 0
                for f in QA_PDF_DIR.glob("*.pdf"):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass
                st.session_state.qa_index_ready = False
                st.session_state.qa_last_build_info = None
                st.success(f"Removed {removed} PDF(s).")

        if uploaded:
            saved = 0
            for uf in uploaded:
                out = QA_PDF_DIR / uf.name
                # overwrite allowed
                with open(out, "wb") as f:
                    f.write(uf.getbuffer())
                saved += 1
            st.success(f"Saved {saved} file(s) to: {QA_PDF_DIR}")

        st.divider()

        st.subheader("2) Build Q&A index (one click)")
        st.caption("This will parse/chunk PDFs and build a vector store for Q&A.")

        if st.button("Build Q&A Index", use_container_width=True, disabled=bool(missing), key="btn_build_qa"):
            key = get_api_key()
            if not key:
                st.error("No API key found. Enter it in the sidebar or set OPENAI_API_KEY.")
            else:
                pdfs = list(QA_PDF_DIR.glob("*.pdf"))
                if not pdfs:
                    st.error("No PDFs found in Q&A folder. Upload PDFs first.")
                else:
                    try:
                        with st.spinner("Building Q&A index (Step3 + embeddings)..."):
                            info = qa_build_index(api_key=key)
                        st.session_state.qa_index_ready = True
                        st.session_state.qa_last_build_info = info
                        st.success(f"✅ Q&A index built: {info['num_pdfs']} PDF(s), {info['num_chunks']} chunks")
                    except Exception as e:
                        st.session_state.qa_index_ready = False
                        st.error(f"Q&A index build failed: {e}")

        if st.session_state.qa_last_build_info:
            with st.expander("Q&A index details"):
                st.json(st.session_state.qa_last_build_info)

        st.divider()

        st.subheader("3) Ask a question")
        st.text_area(
            "Question",
            key="qa_question",
            height=140,
            placeholder="Example: What is the uninsured motorist coverage limit?",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            ask = st.button("Ask", use_container_width=True, disabled=bool(missing), key="btn_qa_ask")
        with c2:
            clear_qa = st.button("Clear", use_container_width=True, key="btn_qa_clear")

        if clear_qa:
            st.session_state.qa_question = ""
            st.session_state.qa_last_answer = ""
            st.session_state.qa_last_sources = []
            st.rerun()

        if ask:
            key = get_api_key()
            q = (st.session_state.get("qa_question", "") or "").strip()

            if not key:
                st.error("No API key found. Enter it in the sidebar or set OPENAI_API_KEY.")
            elif not q:
                st.error("Please enter a question.")
            elif not QA_VECTOR_STORE_PATH.exists():
                st.error("Q&A index not found. Click **Build Q&A Index** first.")
            else:
                try:
                    with st.spinner("Answering with Q&A RAG..."):
                        out = qa_rag_answer(q, api_key=key, top_k=3)
                    st.session_state.qa_last_answer = out.get("answer", "").strip()
                    st.session_state.qa_last_sources = out.get("sources", []) or []
                except Exception as e:
                    st.error(f"Q&A failed: {e}")

    with right:
        st.subheader("Answer")
        if st.session_state.qa_last_answer:
            st.write(st.session_state.qa_last_answer)
        else:
            st.info("Upload PDFs → Build Q&A Index → Ask a question.")

        if st.session_state.qa_last_sources:
            with st.expander("Sources"):
                for i, s in enumerate(st.session_state.qa_last_sources, start=1):
                    if isinstance(s, dict):
                        doc = s.get("doc_name") or s.get("doc_id") or s.get("doc") or "source"
                        ps = s.get("page_start", -1)
                        pe = s.get("page_end", -1)
                        dist = s.get("distance", None)
                        header = f"[{i}] {doc} pages={ps}-{pe}"
                        if dist is not None:
                            header += f" dist={dist:.3f}"
                        st.markdown(f"**{header}**")
                    else:
                        st.write(f"[{i}] {str(s)}")


# =========================
# COMPARE PAGE (UNCHANGED BEHAVIOR)
# =========================
with tab_compare:
    st.title("PolicyPal — Production-style Policy Comparison")
    st.caption(
        "Folders are fixed: data/policy_a and data/policy_b. "
        "This app builds per-policy indexes, generates cached structured summaries, then compares summaries for stable output."
    )

    left, right = st.columns([1.05, 2.2], gap="large")

    with left:
        st.subheader("1) Put PDFs into folders")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Policy A folder**")
            st.code(str(POLICY_A_DIR))
            if st.button("Open Policy A Folder", use_container_width=True, disabled=bool(missing), key="btn_open_a"):
                open_folder(POLICY_A_DIR)

        with c2:
            st.write("**Policy B folder**")
            st.code(str(POLICY_B_DIR))
            if st.button("Open Policy B Folder", use_container_width=True, disabled=bool(missing), key="btn_open_b"):
                open_folder(POLICY_B_DIR)

        st.divider()

        st.subheader("2) Name the policies (labels + summary cache key)")
        st.text_input("Policy A name", value=st.session_state.a_name, key="a_name")
        st.text_input("Policy B name", value=st.session_state.b_name, key="b_name")

        st.divider()

        st.subheader("3) Build / refresh indexes")
        st.caption("Indexes and summaries are stored under storage/compare_prod.")

        if st.button("Build / Refresh Indexes", use_container_width=True, disabled=bool(missing), key="btn_build_indexes"):
            key = get_api_key()
            if not key:
                st.error("No API key found. Enter it in the sidebar or set OPENAI_API_KEY.")
            else:
                try:
                    with st.spinner("Building index for Policy A..."):
                        a_paths = build_policy_index(
                            policy_folder=str(POLICY_A_DIR),
                            policy_name=st.session_state.a_name,
                            api_key=key,
                            out_dir=str(COMPARE_DIR),
                        )
                    with st.spinner("Building index for Policy B..."):
                        b_paths = build_policy_index(
                            policy_folder=str(POLICY_B_DIR),
                            policy_name=st.session_state.b_name,
                            api_key=key,
                            out_dir=str(COMPARE_DIR),
                        )

                    st.session_state.a_store = a_paths.store_path
                    st.session_state.b_store = b_paths.store_path
                    st.session_state.indexes_built = True

                    st.success("✅ Indexes built")
                    st.write(f"- A store: `{st.session_state.a_store}`")
                    st.write(f"- B store: `{st.session_state.b_store}`")

                except Exception as e:
                    st.session_state.indexes_built = False
                    st.error(f"Index build failed: {e}")

        st.divider()

        with st.expander("Troubleshooting"):
            st.markdown(
                "- If outputs look wrong after changing PDFs, either **rebuild indexes** or enable **Force refresh summaries** for one run.\n"
                "- If a PDF is **scanned** (image-only), text extraction may fail without OCR.\n"
                "- Policy names affect summary cache filenames. Changing names creates a new cache.\n"
            )

    with right:
        st.subheader("4) Ask a comparison question")

        st.text_area(
            "Compare question",
            key="compare_question",
            placeholder="Example: Compare uninsured motorist coverage between the two policies.",
            height=120,
        )

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run = st.button("Compare Now", use_container_width=True, disabled=bool(missing), key="btn_compare_now")
        with col_clear:
            clear = st.button("Clear", use_container_width=True, key="btn_compare_clear")

        if clear:
            st.session_state.compare_question = ""
            st.session_state.compare_last_question = ""
            st.session_state.compare_last_answer = ""
            st.rerun()

        if run:
            key = get_api_key()
            q = (st.session_state.get("compare_question", "") or "").strip()

            if not key:
                st.error("No API key found. Enter it in the sidebar or set OPENAI_API_KEY.")
            elif not st.session_state.a_store or not st.session_state.b_store:
                st.error("Please build indexes first.")
            elif not q:
                st.error("Please enter a question.")
            else:
                try:
                    with st.spinner("Comparing policies (summary-based)..."):
                        ans = compare_policies_prod(
                            policy_a_name=st.session_state.a_name,
                            policy_a_store=st.session_state.a_store,
                            policy_b_name=st.session_state.b_name,
                            policy_b_store=st.session_state.b_store,
                            question=q,
                            api_key=key,
                            force_refresh_summaries=bool(st.session_state.force_refresh_summaries),
                        )

                    st.session_state.compare_last_question = q
                    st.session_state.compare_last_answer = ans

                except Exception as e:
                    st.error(f"Compare failed: {e}")

        if st.session_state.compare_last_answer:
            st.subheader("Comparison Result")
            st.write(st.session_state.compare_last_answer)

        st.divider()
        st.caption("Top-K is fixed internally (not shown in UI).")