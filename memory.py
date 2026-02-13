"""
Dual-Layer Adaptive Memory System (DLAMS) for the AAF.

Implements two-tier memory architecture:
1. Working Memory - Recent reasoning steps with priority-based eviction
2. Persistent Knowledge Base (PKB) - FAISS vector store for long-term retrieval
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv

try:
    from openai import OpenAI
    import openai
except ImportError:
    raise ImportError("Please install openai: pip install openai")

from models import PKBEntry, ReasoningStep, TaskSpec, WorkingMemoryRecord

load_dotenv()


class DualLayerMemorySystem:
    """
    Implements the DLAMS module combining working memory with persistent knowledge.
    
    Working memory maintains recent steps with sliding window eviction.
    PKB uses FAISS for semantic retrieval of past solutions and patterns.
    """
    
    def __init__(
        self,
        faiss_index_path: str,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the dual-layer memory system.
        
        Args:
            faiss_index_path: Path to FAISS index file
            embedding_model: OpenAI embedding model (default: text-embedding-3-large)
        """
        self.faiss_index_path = Path(faiss_index_path)
        self.embedding_model = embedding_model
        self.embedding_dim = 3072  # Dimension for text-embedding-3-large
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            print("WARNING: OPENAI_API_KEY not set. Memory system will work in limited mode.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Initialize working memory
        max_size = int(os.getenv("WORKING_MEMORY_SIZE", "12"))
        self.working_memory = WorkingMemoryRecord(max_size=max_size)
        
        # Initialize or load FAISS index
        self.index = self._load_or_create_index()
        
        # Load PKB entries from JSON sidecar
        self._entries: list[PKBEntry] = []
        self._load_entries()
        
        print(f"[DLAMS] Initialized with {len(self._entries)} PKB entries")
        print(f"[DLAMS] Working memory size: {max_size}")
        print(f"[DLAMS] Embedding model: {embedding_model}")
    
    def _load_or_create_index(self) -> faiss.Index:
        """
        Load existing FAISS index or create a new one.
        
        Returns:
            FAISS index (IndexFlatIP for cosine similarity)
        """
        index_file = self.faiss_index_path.with_suffix(".index")
        
        if index_file.exists():
            try:
                index = faiss.read_index(str(index_file))
                print(f"[DLAMS] Loaded existing FAISS index from {index_file}")
                return index
            except Exception as e:
                print(f"[DLAMS] Error loading index: {e}. Creating new index.")
        
        # Create new IndexFlatIP (inner product, works for normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Ensure parent directory exists
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[DLAMS] Created new FAISS index (dim={self.embedding_dim})")
        return index
    
    def _load_entries(self) -> None:
        """Load PKB entries from JSON sidecar file."""
        json_file = self.faiss_index_path.with_suffix(".json")
        
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    self._entries = [PKBEntry(**entry) for entry in data]
                print(f"[DLAMS] Loaded {len(self._entries)} entries from {json_file}")
            except Exception as e:
                print(f"[DLAMS] Error loading entries: {e}")
                self._entries = []
        else:
            self._entries = []
    
    def _save_entries(self) -> None:
        """Save PKB entries to JSON sidecar file."""
        json_file = self.faiss_index_path.with_suffix(".json")
        json_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(json_file, "w") as f:
                data = [entry.model_dump() for entry in self._entries]
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[DLAMS] Error saving entries: {e}")
    
    def add_to_working_memory(self, step: ReasoningStep) -> None:
        """
        Add a reasoning step to working memory with priority assignment.
        
        Args:
            step: ReasoningStep to add to working memory
        """
        # Set priority based on step characteristics
        if step.is_error or "result" in step.action.lower():
            step.priority = 1.0  # High priority for errors and results
        else:
            step.priority = 0.5  # Normal priority for routine steps
        
        self.working_memory.add(step)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized numpy array (L2 normalization for cosine similarity)
        """
        if not self.client:
            # Return zero vector if no API key
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text[:8000]  # Truncate to avoid token limits
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                # L2 normalization for cosine similarity with IndexFlatIP
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
                
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"[DLAMS] Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[DLAMS] Error generating embedding: {e}")
                    return np.zeros(self.embedding_dim, dtype=np.float32)
        
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def add_to_pkb(self, entry: PKBEntry) -> None:
        """
        Add an entry to the Persistent Knowledge Base.
        
        Args:
            entry: PKBEntry to add to PKB
        """
        # Generate embedding if not already present
        if not entry.embedding or len(entry.embedding) == 0:
            embedding = self.embed_text(entry.content)
            entry.embedding = embedding.tolist()
        else:
            embedding = np.array(entry.embedding, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(np.array([embedding]))
        
        # Add to entries list
        self._entries.append(entry)
        
        # Save to disk
        self._save_entries()
        self.save_index()
        
        print(f"[DLAMS] Added entry to PKB: {entry.entry_type} ({len(entry.content)} chars)")
    
    def retrieve_from_pkb(self, query: str, top_k: int = 3) -> list[PKBEntry]:
        """
        Retrieve most relevant entries from PKB using semantic search.
        
        Args:
            query: Query text for retrieval
            top_k: Number of entries to retrieve
            
        Returns:
            List of top-k most relevant PKBEntry objects
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Search FAISS index
        try:
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(
                np.array([query_embedding]), k
            )
            
            # Return corresponding entries
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._entries):
                    results.append(self._entries[idx])
            
            return results
            
        except Exception as e:
            print(f"[DLAMS] Error retrieving from PKB: {e}")
            return []
    
    def build_memory_context(self, current_task: str) -> str:
        """
        Build a comprehensive memory context for LLM prompt injection.
        
        Combines working memory (recent steps) with PKB retrieval (relevant past knowledge).
        
        Args:
            current_task: Current task description for PKB retrieval
            
        Returns:
            Formatted string with both memory layers
        """
        # Get working memory
        wm_string = self.working_memory.to_prompt_string()
        
        # Retrieve from PKB
        pkb_entries = self.retrieve_from_pkb(current_task, top_k=3)
        
        # Format context
        lines = [
            "=" * 60,
            "=== WORKING MEMORY (Recent Steps) ===",
            "=" * 60,
            wm_string,
            "",
            "=" * 60,
            "=== RETRIEVED KNOWLEDGE (Top-3 Relevant) ===",
            "=" * 60
        ]
        
        if pkb_entries:
            for i, entry in enumerate(pkb_entries, 1):
                lines.append(f"\nEntry {i} [{entry.entry_type}]:")
                lines.append(f"Task: {entry.source_task}")
                content_preview = entry.content[:500]
                if len(entry.content) > 500:
                    content_preview += "..."
                lines.append(f"Content: {content_preview}")
                lines.append("")
        else:
            lines.append("\nNo relevant entries found in knowledge base.")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_successful_solution(self, task: TaskSpec, solution_code: str) -> None:
        """
        Save a successful solution to the PKB for future reference.
        
        Args:
            task: TaskSpec that was solved
            solution_code: The solution code
        """
        # Create task fingerprint (hash of normalized description)
        normalized_desc = task.raw_description.lower().strip()[:64]
        fingerprint = hashlib.sha256(normalized_desc.encode()).hexdigest()[:16]
        
        # Create PKB entry
        entry = PKBEntry(
            task_fingerprint=fingerprint,
            content=solution_code,
            entry_type="code_solution",
            source_task=task.raw_description
        )
        
        # Add to PKB
        self.add_to_pkb(entry)
        
        print(f"[DLAMS] Saved successful solution to PKB (fingerprint: {fingerprint})")
    
    def clear_working_memory(self) -> None:
        """Clear all records from working memory."""
        self.working_memory.clear()
        print("[DLAMS] Working memory cleared")
    
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        index_file = self.faiss_index_path.with_suffix(".index")
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            faiss.write_index(self.index, str(index_file))
        except Exception as e:
            print(f"[DLAMS] Error saving index: {e}")


# ============================================================================
# Test Function
# ============================================================================

def test_memory_system():
    """Standalone test for the DualLayerMemorySystem."""
    print("\n" + "=" * 80)
    print("Testing DualLayerMemorySystem")
    print("=" * 80)
    
    # Create memory system
    memory = DualLayerMemorySystem(faiss_index_path="./test_pkb")
    
    # Test 1: Add to working memory
    print("\n1. Testing Working Memory:")
    step1 = ReasoningStep(
        step_id=1,
        thought="Need to understand the problem",
        action="analyze",
        confidence=0.8
    )
    step2 = ReasoningStep(
        step_id=2,
        thought="Writing initial code",
        action="write_code",
        confidence=0.9,
        is_error=False
    )
    
    memory.add_to_working_memory(step1)
    memory.add_to_working_memory(step2)
    
    print(f"   Working memory size: {len(memory.working_memory.records)}")
    print(f"   Step 1 priority: {step1.priority}")
    print(f"   Step 2 priority: {step2.priority}")
    
    # Test 2: Add to PKB
    print("\n2. Testing PKB:")
    entry = PKBEntry(
        task_fingerprint="test_hash_123",
        content="def hello(): return 'Hello, World!'",
        entry_type="code_solution",
        source_task="Write a hello world function"
    )
    
    memory.add_to_pkb(entry)
    print(f"   PKB size: {memory.index.ntotal}")
    
    # Test 3: Build memory context
    print("\n3. Testing Memory Context:")
    context = memory.build_memory_context("test query about hello world")
    print("   Context length:", len(context))
    print("\n   Context preview:")
    print("   " + "\n   ".join(context[:500].split("\n")))
    
    # Test 4: Retrieve from PKB
    print("\n4. Testing PKB Retrieval:")
    results = memory.retrieve_from_pkb("hello world function", top_k=1)
    print(f"   Retrieved {len(results)} entries")
    if results:
        print(f"   Entry type: {results[0].entry_type}")
        print(f"   Source task: {results[0].source_task}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_memory_system()
