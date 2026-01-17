"""Semantic memory with vector storage for IntelliShell."""

import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Represents a memory record to be indexed."""
    id: str
    user_input: str
    intent_name: str
    provider_name: str
    output_summary: str
    timestamp: str
    confidence: float
    success: bool
    entities: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """
    Vector storage backend for semantic memory.
    
    Uses ChromaDB for local vector storage with background indexing.
    """
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        if persist_directory is None:
            persist_directory = Path.home() / ".intellishell" / "vector_store"
        
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._client = None
        self._collection = None
        self._initialized = False
        self._indexing_queue = []
        self._indexing_lock = threading.Lock()
        self._background_thread: Optional[threading.Thread] = None
        self._should_stop = False  # Flag to signal thread to stop
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize ChromaDB client.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name="intent_history",
                metadata={"description": "IntelliShell command history"}
            )
            
            self._initialized = True
            logger.info(f"Vector store initialized at {self.persist_directory}")
            return True
            
        except ImportError:
            logger.warning(
                "chromadb not installed. Semantic memory disabled. "
                "Install with: pip install chromadb"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if vector store is available."""
        return self._initialized
    
    def add_async(self, record: MemoryRecord) -> None:
        """
        Add record to indexing queue (non-blocking).
        
        Args:
            record: Memory record to index
        """
        if not self._initialized:
            return
        
        with self._indexing_lock:
            self._indexing_queue.append(record)
        
        # Start background thread if not running
        if self._background_thread is None or not self._background_thread.is_alive():
            self._should_stop = False  # Reset stop flag when starting new thread
            self._background_thread = threading.Thread(
                target=self._process_queue,
                daemon=True
            )
            self._background_thread.start()
    
    def _process_queue(self) -> None:
        """Process indexing queue in background thread."""
        while not self._should_stop:
            with self._indexing_lock:
                if not self._indexing_queue:
                    break
                record = self._indexing_queue.pop(0)
            
            try:
                self._add_to_collection(record)
            except Exception as e:
                logger.error(f"Failed to index record: {e}")
    
    def _add_to_collection(self, record: MemoryRecord) -> None:
        """
        Add record to ChromaDB collection.
        
        Args:
            record: Memory record to add
        """
        if not self._initialized or self._collection is None:
            return
        
        # Create document text for embedding
        document = f"{record.user_input} {record.output_summary}"
        
        # Create metadata
        metadata = {
            "intent_name": record.intent_name,
            "provider_name": record.provider_name,
            "timestamp": record.timestamp,
            "confidence": record.confidence,
            "success": str(record.success),
        }
        
        # Add entities to metadata (flatten)
        if record.entities:
            for i, entity in enumerate(record.entities[:3]):  # Limit to 3
                metadata[f"entity_{i}_type"] = entity.get("type", "")
                metadata[f"entity_{i}_value"] = entity.get("value", "")
        
        try:
            self._collection.add(
                ids=[record.id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.debug(f"Indexed record: {record.id}")
        except Exception as e:
            logger.error(f"Failed to add to collection: {e}")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_success: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over memory.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_success: Only return successful commands
            
        Returns:
            List of matching records
        """
        if not self._initialized or self._collection is None:
            logger.warning("Vector store not available for search")
            return []
        
        try:
            # Build filter
            where_filter = None
            if filter_success:
                where_filter = {"success": "True"}
            
            # Query collection
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None
            )
            
            # Parse results
            matches = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    match = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    }
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent memories.
        
        Args:
            limit: Number of recent records to return
            
        Returns:
            List of recent records
        """
        if not self._initialized or self._collection is None:
            return []
        
        try:
            results = self._collection.get(limit=limit)
            
            memories = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    memory = {
                        "id": results["ids"][i],
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    }
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get recent: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all memories (use with caution)."""
        if not self._initialized or self._client is None:
            return
        
        try:
            self._client.delete_collection("intent_history")
            self._collection = self._client.create_collection("intent_history")
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")


class SemanticMemory:
    """
    High-level semantic memory interface.
    
    Manages vector storage and provides semantic search capabilities.
    """
    
    def __init__(self):
        """Initialize semantic memory."""
        self.vector_store = VectorStore()
    
    def is_available(self) -> bool:
        """Check if semantic memory is available."""
        return self.vector_store.is_available()
    
    def remember(
        self,
        user_input: str,
        intent_name: str,
        provider_name: str,
        output_summary: str,
        confidence: float,
        success: bool,
        entities: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store a memory (non-blocking).
        
        Args:
            user_input: User's original input
            intent_name: Matched intent
            provider_name: Provider that executed
            output_summary: Summary of output
            confidence: Confidence score
            success: Whether execution succeeded
            entities: Extracted entities
            metadata: Additional metadata
        """
        record = MemoryRecord(
            id=f"{datetime.now().timestamp()}_{hash(user_input)}",
            user_input=user_input,
            intent_name=intent_name,
            provider_name=provider_name,
            output_summary=output_summary,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            success=success,
            entities=entities or [],
            metadata=metadata or {}
        )
        
        # Add asynchronously
        self.vector_store.add_async(record)
    
    def recall(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recall memories matching query.
        
        Args:
            query: Natural language query
            n_results: Number of results
            
        Returns:
            List of matching memories
        """
        return self.vector_store.search(query, n_results=n_results)
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        return self.vector_store.get_recent(limit=limit)
