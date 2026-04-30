from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_PATH = Path("data/pubmed_index/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


class PubMedRAGPipeline:
    """
    LangChain + FAISS RAG over PubMed abstracts.
    Load once at startup via get_rag_pipeline().
    """

    def __init__(
        self,
        index_path: str | Path = INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K,
    ):
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self._vectorstore: Optional[FAISS] = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def build_index(self, abstracts: list[dict]) -> None:
        """
        Build and persist a FAISS index from a list of PubMed abstract dicts.

        Each dict: {"pmid": str, "title": str, "abstract": str}
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = []
        for item in abstracts:
            text = f"{item['title']}\n\n{item['abstract']}"
            chunks = splitter.create_documents(
                [text],
                metadatas=[{"pmid": item.get("pmid", ""), "title": item["title"]}],
            )
            docs.extend(chunks)

        embeddings = self._get_embeddings()
        self._vectorstore = FAISS.from_documents(docs, embeddings)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_path))

    def load_index(self) -> None:
        """Load a pre-built FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run build_index() or fetch_and_build() first."
            )
        embeddings = self._get_embeddings()
        self._vectorstore = FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k relevant PubMed abstracts for a query.

        Returns:
            list of {"title": str, "pmid": str, "text": str, "score": float}
        """
        if self._vectorstore is None:
            self.load_index()

        k = k or self.top_k
        results = self._vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "title": doc.metadata.get("title", ""),
                "pmid": doc.metadata.get("pmid", ""),
                "text": doc.page_content,
                "score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def fetch_and_build(self, query_terms: list[str], max_results: int = 100) -> None:
        """
        Fetch PubMed abstracts via Entrez API and build the FAISS index.
        Requires: pip install biopython
        """
        from Bio import Entrez

        Entrez.email = os.getenv("ENTREZ_EMAIL", "researcher@example.com")
        abstracts = []

        for term in query_terms:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results // len(query_terms))
            record = Entrez.read(handle)
            handle.close()
            ids = record["IdList"]

            if not ids:
                continue

            handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                try:
                    med = article["MedlineCitation"]
                    pmid = str(med["PMID"])
                    title = str(med["Article"]["ArticleTitle"])
                    abs_texts = med["Article"].get("Abstract", {}).get("AbstractText", [""])
                    abstract = " ".join(str(a) for a in abs_texts)
                    abstracts.append({"pmid": pmid, "title": title, "abstract": abstract})
                except (KeyError, TypeError):
                    continue

        self.build_index(abstracts)

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved docs into a prompt-ready string."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r['title']}\nPMID: {r['pmid']}\n{r['text']}"
            )
        return "\n\n---\n\n".join(lines)


_rag: Optional[PubMedRAGPipeline] = None


def get_rag_pipeline(index_path: str | Path = INDEX_PATH) -> PubMedRAGPipeline:
    """Return the singleton RAG pipeline, loading the index on first call."""
    global _rag
    if _rag is None:
        _rag = PubMedRAGPipeline(index_path=index_path)
        try:
            _rag.load_index()
        except FileNotFoundError:
            pass  # Index will be built lazily or via CLI
    return _rag
