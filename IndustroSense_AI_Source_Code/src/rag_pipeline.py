"""
IndustroSense AI - Knowledge Retrieval Module
Implements: document chunking, TF-IDF+SVD dense embedding, FAISS vector index,
and top-k semantic retrieval.

Note on embedding model: the reference design specifies a hosted embedding API
(e.g., OpenAI text-embedding-3-small, 1536-dim). Because this lab environment has
no internet access to model-hosting endpoints, an offline substitute embedding is
used here: TF-IDF vectorization followed by Truncated SVD (Latent Semantic
Analysis) projected to 128 dimensions. This preserves the same pipeline contract
(text -> fixed-length dense vector) so the RAG, agent, and evaluation logic are
unchanged if a hosted embedding API is swapped in later.
"""
import os
import glob
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

CHUNK_SIZE = 60      # words per chunk
CHUNK_OVERLAP = 15   # words of overlap between consecutive chunks
EMBED_DIM = 128


def load_corpus(corpus_dir):
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, "r") as f:
            text = f.read()
        doc_id = text.split("DOC_ID:")[1].split("\n")[0].strip() if "DOC_ID:" in text else os.path.basename(path)
        title = text.split("TITLE:")[1].split("\n")[0].strip() if "TITLE:" in text else os.path.basename(path)
        docs.append({"path": path, "doc_id": doc_id, "title": title, "text": text})
    return docs


def chunk_text(text, doc_id, title, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"{doc_id}::chunk{idx}",
            "doc_id": doc_id,
            "title": title,
            "text": chunk_str,
        })
        idx += 1
        if end == len(words):
            break
        start = end - overlap
    return chunks


class VectorStore:
    """FAISS-backed flat (exact, brute-force L2-on-normalized-vectors ~ cosine) index."""

    def __init__(self, dim=EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vecs = cosine similarity
        self.chunks = []
        self.vectorizer = None
        self.svd = None

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(EMBED_DIM, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 2)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = self.svd.fit_transform(tfidf).astype("float32")
        # pad/truncate to fixed EMBED_DIM for a stable index dimensionality
        if dense.shape[1] < EMBED_DIM:
            pad = np.zeros((dense.shape[0], EMBED_DIM - dense.shape[1]), dtype="float32")
            dense = np.hstack([dense, pad])
        faiss.normalize_L2(dense)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(dense)
        self._chunk_vecs = dense
        return dense

    def embed_query(self, query):
        tfidf_q = self.vectorizer.transform([query])
        dense_q = self.svd.transform(tfidf_q).astype("float32")
        if dense_q.shape[1] < EMBED_DIM:
            pad = np.zeros((dense_q.shape[0], EMBED_DIM - dense_q.shape[1]), dtype="float32")
            dense_q = np.hstack([dense_q, pad])
        faiss.normalize_L2(dense_q)
        return dense_q

    def search(self, query, k=3):
        qvec = self.embed_query(query)
        scores, ids = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append({**c, "score": float(score)})
        return results


def build_index(corpus_dir):
    docs = load_corpus(corpus_dir)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d["text"], d["doc_id"], d["title"]))
    store = VectorStore()
    store.build(all_chunks)
    return store, docs, all_chunks


def augmented_prompt(query, retrieved_chunks):
    """Construct the RAG-augmented LLM prompt with citations."""
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[{i}] (Source: {c['doc_id']} | score={c['score']:.3f})\n{c['text']}")
    context = "\n\n".join(context_blocks)
    prompt = (
        "SYSTEM: You are IndustroSense AI, an industrial diagnostic assistant. "
        "Answer ONLY using the numbered sources below. Cite sources as [n]. "
        "If the sources are insufficient, say so explicitly.\n\n"
        f"SOURCES:\n{context}\n\nUSER QUERY: {query}\n\nANSWER:"
    )
    return prompt


if __name__ == "__main__":
    store, docs, chunks = build_index(os.path.join(os.path.dirname(__file__), "..", "corpus"))
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks. Embedding dim = {EMBED_DIM}")
