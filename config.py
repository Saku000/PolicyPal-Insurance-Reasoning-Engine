# ==============================
# PolicyPal Configuration File
# ==============================

# ========= Step 3: Parsing & Chunking =========

# 输入PDF目录
INPUT_PDF_DIR = "data/sample_policies"

# Step 3 输出 JSON
OUTPUT_CHUNKS_PATH = "storage/parsed_chunks.json"

# Token-based chunking
TOKEN_CHUNK_SIZE = 800          # 每块 800 tokens
TOKEN_CHUNK_OVERLAP = 100       # 重叠 100 tokens
TOKEN_ENCODING_NAME = "cl100k_base"

# 过滤过短 chunk（字符数）
MIN_CHUNK_CHARS = 300




# ===== Step 4: Embeddings + Vector DB (Chroma) =====
CHROMA_PERSIST_DIR = "storage/chroma"
CHROMA_COLLECTION_NAME = "policypal_chunks"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval test
RETRIEVAL_TOP_K = 3

# ===== Step 5: RAG Answer Generation =====
CHAT_MODEL = "gpt-4o-mini"

MAX_CONTEXT_CHARS = 12000   # 控制塞进prompt的上下文长度（简单按字符截断）
RAG_TOP_K = 3
RAG_DISTANCE_THRESHOLD = 1.2  # 距离越小越相似

# Step6 Intent Classification
INTENT_MODEL = "gpt-4o-mini"
ENABLE_INTENT_ROUTER = True