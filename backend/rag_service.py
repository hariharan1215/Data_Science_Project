"""
CareerMate - RAG (Retrieval-Augmented Generation) Service
Uses ChromaDB for vector storage and similarity search with Gemini embeddings.
Includes intelligent fallback for seamless offline testing.
"""
import logging
import math
import re
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.config import Config

logger = logging.getLogger(__name__)

class SimpleHashEmbeddingFunction:
    """
    Lightweight, self-contained embedding function for ChromaDB.
    Generates deterministic 64-dimensional normalized vectors without downloading
    external models to ~/.cache, ensuring offline stability.
    """
    def name(self) -> str:
        return "simple_hash"

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        dim = 64
        for text in input:
            vec = [0.0] * dim
            tokens = re.findall(r"\w+", text.lower())
            for token in tokens:
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

    def embed_query(self, input: Any) -> Any:
        if isinstance(input, list):
            return self.embed_documents(input)
        return self.embed_documents([input])[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

class RAGService:
    """
    Manages indexing and vector similarity retrieval of career documents
    stored in ChromaDB.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIRECTORY
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._fallback_store: Dict[str, List[Dict[str, Any]]] = {}
        self._init_chromadb()

    def _init_chromadb(self):
        """Initializes ChromaDB persistent client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            # Collection for all session documents with lightweight embedding function
            self._collection = self._client.get_or_create_collection(
                name="careermate_rag_docs",
                embedding_function=SimpleHashEmbeddingFunction(),
                metadata={"description": "CareerMate Career Documents Vector Store"}
            )
            logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. Falling back to internal memory index.")
            self._client = None
            self._collection = None

    def index_document(
        self,
        session_id: str,
        filename: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Indexes a list of document chunks into ChromaDB.
        Uses the collection's local embedding function to avoid a remote API call.
        Returns the number of indexed chunks.
        """
        if not chunks:
            return 0

        # Always save in fallback memory store for reliable retrieval
        if session_id not in self._fallback_store:
            self._fallback_store[session_id] = []
        # Filter out old chunks of same filename
        self._fallback_store[session_id] = [
            c for c in self._fallback_store[session_id] if c.get("source") != filename
        ]
        self._fallback_store[session_id].extend(chunks)

        if not self._collection:
            logger.info(f"Using fallback memory index for {filename} ({len(chunks)} chunks).")
            return len(chunks)

        # Prepare records for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{session_id}_{filename}_p{chunk.get('page', 1)}_c{i}"
            ids.append(chunk_id)
            documents.append(chunk.get("text", ""))
            metadatas.append({
                "session_id": session_id,
                "source": filename,
                "page": int(chunk.get("page", 1)),
                "chunk_id": chunk_id
            })

        try:
            # Delete any existing chunks for this session and filename
            try:
                self._collection.delete(
                    where={"$and": [{"session_id": session_id}, {"source": filename}]}
                )
            except Exception:
                pass

            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks in ChromaDB for {filename}.")
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}", exc_info=True)

        return len(chunks)

    def retrieve_relevant_chunks(
        self,
        session_id: str,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant session chunks locally to avoid a remote embedding call.
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL
        session_chunks = self._fallback_store.get(session_id, [])
        if not session_chunks and self._collection:
            try:
                stored = self._collection.get(
                    where={"session_id": session_id},
                    include=["documents", "metadatas"]
                )
                documents = stored.get("documents") or []
                metadatas = stored.get("metadatas") or [{}] * len(documents)
                session_chunks = [
                    {
                        "text": text,
                        "source": metadata.get("source", "Uploaded Document"),
                        "page": metadata.get("page", 1)
                    }
                    for text, metadata in zip(documents, metadatas)
                ]
            except Exception as e:
                logger.warning("Could not load persisted RAG chunks for session %s: %s", session_id, e)

        if not session_chunks:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in session_chunks:
            text = chunk.get("text", "")
            tokens = re.findall(r"\w+", text.lower())
            total = len(tokens) or 1
            overlap = sum(1 for t in tokens if t in query_words)
            score = overlap / math.sqrt(total)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in scored[:top_k] if item[0] > 0]

        if not top_results and session_chunks:
            top_results = session_chunks[:2]

        return top_results

    def get_indexed_documents(self, session_id: str) -> List[str]:
        """Returns the list of uploaded filenames for a given session."""
        docs = set()
        for chunk in self._fallback_store.get(session_id, []):
            if chunk.get("source"):
                docs.add(chunk["source"])

        if self._collection:
            try:
                res = self._collection.get(where={"session_id": session_id})
                if res and res.get("metadatas"):
                    for m in res["metadatas"]:
                        if m.get("source"):
                            docs.add(m["source"])
            except Exception:
                pass

        return list(docs)

    def delete_session_documents(self, session_id: str):
        """Clears all indexed document vectors and cached chunks for a session."""
        if session_id in self._fallback_store:
            del self._fallback_store[session_id]

        if self._collection:
            try:
                self._collection.delete(where={"session_id": session_id})
                logger.info(f"Deleted ChromaDB records for session: {session_id}")
            except Exception as e:
                logger.warning(f"Error clearing ChromaDB for session {session_id}: {e}")
