"""RAG Pipeline: Context Retrieval, Prompt Construction, and LLM Generation."""

import time
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS,
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


SYSTEM_PROMPT = """You are a precise Industrial Systems Technical Assistant specializing in engineering product datasheets and specifications.

Your instructions:
1. Answer the user's question using ONLY the provided technical context.
2. Be technically precise: preserve exact numbers, electrical ratings, tolerances, and engineering units (e.g., V DC, A, °C, IP rating).
3. If the provided context does NOT contain the answer, state clearly: "The provided technical documentation does not specify this parameter." DO NOT guess or hallucinate.
4. Always include concise citations indicating which document/section your facts were extracted from.
"""


class TechnicalRAG:
    def __init__(self):
        # Local embedding model
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="technical_datasheets",
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant technical chunks with cosine distance scores."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Cosine distance to similarity score: similarity = 1 - distance
                similarity = round(1.0 - float(dist), 4)
                retrieved_items.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 1),
                    "chunk_id": meta.get("chunk_id", ""),
                    "similarity": similarity
                })

        return retrieved_items

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Construct the prompt and call the configured LLM provider."""
        if not retrieved_chunks:
            return "No relevant technical documentation found in the vector database. Please ingest documents first."

        context_block = "\n\n---\n\n".join(
            f"[Source: {c['source']} | Page: {c['page']} | Relevance: {c['similarity']:.2f}]\n{c['content']}"
            for c in retrieved_chunks
        )

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"### TECHNICAL CONTEXT:\n{context_block}\n\n"
            f"### USER QUESTION:\n{query}\n\n"
            f"### ANSWER:"
        )

        # 1. Google Gemini API
        if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                return response.text
            except Exception as e:
                return f"[Gemini API Error]: {e}"

        # 2. Groq (Ultra fast & Free tier)
        elif LLM_PROVIDER == "groq" and GROQ_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=GROQ_API_KEY
                )
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                return f"[Groq API Error]: {e}"

        # 3. OpenAI / Ollama / OpenAI-compatible
        elif OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                return f"[OpenAI API Error]: {e}"

        # 4. Fallback if no API key is set yet (demonstrates retrieval & prompt construction)
        else:
            return (
                "[Notice: No LLM API key detected in .env file]\n\n"
                "RAG Retrieval succeeded! Below is the prompt prepared for the LLM:\n\n"
                f"{full_prompt}\n\n"
                "(To get live LLM answers, add GEMINI_API_KEY or GROQ_API_KEY to your .env file)"
            )

    def query(self, user_question: str) -> Dict[str, Any]:
        """Execute end-to-end RAG query: timing, retrieval, and generation."""
        start_time = time.time()
        retrieved_chunks = self.retrieve(user_question)
        retrieval_latency = time.time() - start_time

        gen_start = time.time()
        answer = self.generate_response(user_question, retrieved_chunks)
        gen_latency = time.time() - gen_start

        return {
            "query": user_question,
            "answer": answer,
            "sources": retrieved_chunks,
            "latency": {
                "retrieval_seconds": round(retrieval_latency, 3),
                "generation_seconds": round(gen_latency, 3),
                "total_seconds": round(retrieval_latency + gen_latency, 3)
            }
        }
