"""
RAG (Retrieval Augmented Generation) system for roop-unleashed.
Provides knowledge-based assistance using vector embeddings and retrieval.
"""

import os
import json
import logging
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. RAG functionality will be limited.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available. Using simple vector search.")

from .error_handling import RoopException, retry_on_error
from .llm_integration import LLMManager

logger = logging.getLogger(__name__)


class RAGError(RoopException):
    """Raised when RAG operations fail."""
    pass


class Document:
    """Represents a document in the knowledge base."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                 doc_id: Optional[str] = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
        self.embedding = None
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the document."""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'doc_id': self.doc_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            content=data['content'],
            metadata=data.get('metadata', {}),
            doc_id=data.get('doc_id')
        )


class VectorStore:
    """Vector storage and retrieval system."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vector_path: str = "./rag_vectors"):
        self.embedding_model_name = embedding_model
        self.vector_path = Path(vector_path)
        self.vector_path.mkdir(parents=True, exist_ok=True)
        
        self.documents: List[Document] = []
        self.embeddings = None
        self.embedding_model = None
        self.faiss_index = None
        
        self._initialize_embedding_model()
        self._load_if_exists()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, RAG will be limited")
            return
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
    
    def _load_if_exists(self):
        """Load existing vector store if it exists."""
        docs_file = self.vector_path / "documents.json"
        embeddings_file = self.vector_path / "embeddings.pkl"
        
        if docs_file.exists() and embeddings_file.exists():
            try:
                # Load documents
                with open(docs_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    self.documents = [Document.from_dict(d) for d in doc_data]
                
                # Load embeddings
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                # Initialize FAISS index if available
                if FAISS_AVAILABLE and self.embeddings is not None:
                    self._build_faiss_index()
                
                logger.info(f"Loaded {len(self.documents)} documents from vector store")
                
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.documents = []
                self.embeddings = None
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None or not FAISS_AVAILABLE:
            return
        
        try:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings.astype('float32'))
            logger.debug(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self.faiss_index = None
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the vector store."""
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return False
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([document.content])
            document.embedding = embedding[0]
            
            # Add to documents list
            self.documents.append(document)
            
            # Update embeddings array
            if self.embeddings is None:
                self.embeddings = embedding
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            
            # Rebuild FAISS index
            if FAISS_AVAILABLE:
                self._build_faiss_index()
            
            logger.debug(f"Added document: {document.doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add multiple documents to the vector store."""
        added_count = 0
        for doc in documents:
            if self.add_document(doc):
                added_count += 1
        return added_count
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not self.embedding_model or not self.documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            if self.faiss_index is not None:
                # Use FAISS for fast search
                query_embedding_norm = query_embedding.copy().astype('float32')
                faiss.normalize_L2(query_embedding_norm)
                
                scores, indices = self.faiss_index.search(query_embedding_norm, top_k)
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.documents):
                        results.append((self.documents[idx], float(score)))
                
                return results
            else:
                # Fallback to numpy-based search
                similarities = np.dot(self.embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    results.append((self.documents[idx], float(similarities[idx])))
                
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save(self):
        """Save the vector store to disk."""
        try:
            # Save documents
            docs_data = [doc.to_dict() for doc in self.documents]
            with open(self.vector_path / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self.embeddings is not None:
                with open(self.vector_path / "embeddings.pkl", 'wb') as f:
                    pickle.dump(self.embeddings, f)
            
            logger.info(f"Saved vector store with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def clear(self):
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        logger.info("Cleared vector store")


class RAGSystem:
    """Complete RAG system combining retrieval and generation."""
    
    def __init__(self, settings, llm_manager: LLMManager):
        self.settings = settings
        self.llm_manager = llm_manager
        
        # Initialize vector store
        vector_path = settings.get_ai_setting('rag.vector_store_path', './rag_vectors')
        embedding_model = settings.get_ai_setting('rag.embedding_model', 'all-MiniLM-L6-v2')
        
        self.vector_store = VectorStore(embedding_model, vector_path)
        self.chunk_size = settings.get_ai_setting('rag.chunk_size', 512)
        self.chunk_overlap = settings.get_ai_setting('rag.chunk_overlap', 50)
        
        # Load knowledge base if enabled
        if settings.get_ai_setting('rag.enabled', False):
            self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load documents from knowledge base directory."""
        knowledge_path = Path(self.settings.get_agent_setting('rag_agent', 'knowledge_base_path', './knowledge'))
        
        if not knowledge_path.exists():
            knowledge_path.mkdir(parents=True, exist_ok=True)
            self._create_default_knowledge_base(knowledge_path)
        
        # Load text files from knowledge base
        loaded_count = 0
        for file_path in knowledge_path.glob("*.txt"):
            if self._load_document_file(file_path):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} knowledge base files")
    
    def _create_default_knowledge_base(self, knowledge_path: Path):
        """Create default knowledge base files."""
        default_docs = {
            'face_swapping_guide.txt': """
Face Swapping Best Practices:

1. Source Image Quality:
   - High resolution (1024x1024 or higher recommended)
   - Clear lighting without harsh shadows
   - Front-facing or slight angle (avoid extreme profiles)
   - Single face clearly visible
   - Minimal occlusion (no sunglasses, masks, or hair covering face)

2. Target Image/Video Quality:
   - Similar lighting conditions to source
   - Compatible face angles
   - Good resolution (minimum 512x512 for images)
   - Stable face positioning in videos

3. Processing Parameters:
   - Use GPU acceleration when available (CUDA/ROCm)
   - Adjust face enhancement settings based on target quality
   - Consider using face restoration for low-quality targets
   - Use appropriate video codecs (H.264 for compatibility, H.265 for size)

4. Common Issues and Solutions:
   - Blurry results: Increase source image quality or use face enhancement
   - Color mismatch: Ensure similar lighting conditions
   - Flickering in videos: Use video stabilization or lower frame rate
   - GPU memory errors: Reduce batch size or use CPU fallback
            """,
            
            'gpu_optimization.txt': """
GPU Optimization for Roop Unleashed:

1. NVIDIA CUDA Optimization:
   - Use latest CUDA drivers (11.8+ or 12.1+)
   - Enable TensorRT when available for inference acceleration
   - Use mixed precision (FP16) to reduce memory usage
   - Optimize CUDA streams for parallel processing
   - Monitor GPU memory usage and adjust batch sizes

2. AMD ROCm Support:
   - Ensure ROCm 5.4+ is installed
   - Use ROCm-optimized PyTorch builds
   - Monitor HIP memory allocation

3. Memory Management:
   - Set appropriate memory limits in configuration
   - Use gradient checkpointing for large models
   - Clear GPU cache between operations
   - Monitor for memory leaks

4. Performance Tuning:
   - Use optimal thread counts (typically 4-8)
   - Enable frame buffer optimization
   - Use appropriate video codecs and quality settings
   - Profile operations to identify bottlenecks
            """,
            
            'troubleshooting.txt': """
Common Issues and Troubleshooting:

1. Installation Issues:
   - Verify Python version (3.9-3.12 supported)
   - Install Visual C++ redistributables on Windows
   - Check CUDA/ROCm installation for GPU support
   - Verify FFmpeg installation for video processing

2. Runtime Errors:
   - GPU out of memory: Reduce batch size or use CPU
   - Model loading failures: Check model cache directory
   - Video processing errors: Verify FFmpeg codecs
   - Face detection issues: Check input image quality

3. Performance Issues:
   - Slow processing: Enable GPU acceleration
   - High memory usage: Adjust memory limits
   - Video artifacts: Use appropriate quality settings
   - Inconsistent results: Check source/target compatibility

4. Configuration Problems:
   - Settings not saving: Check file permissions
   - AI features not working: Verify LLM service configuration
   - Agent errors: Check agent dependencies and configuration
            """
        }
        
        for filename, content in default_docs.items():
            with open(knowledge_path / filename, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        logger.info("Created default knowledge base")
    
    def _load_document_file(self, file_path: Path) -> bool:
        """Load a single document file into the vector store."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = self._split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    metadata={
                        'source_file': str(file_path),
                        'chunk_index': i,
                        'filename': file_path.name
                    }
                )
                documents.append(doc)
            
            added_count = self.vector_store.add_documents(documents)
            logger.debug(f"Added {added_count} chunks from {file_path.name}")
            return added_count > 0
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return False
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk]
    
    @retry_on_error(max_retries=2, exceptions=(RAGError,))
    def query(self, question: str, top_k: int = 3) -> Optional[str]:
        """Query the RAG system with a question."""
        if not self.settings.get_ai_setting('rag.enabled', False):
            logger.debug("RAG is disabled")
            return None
        
        if not self.llm_manager.is_available():
            logger.warning("No LLM available for RAG")
            return None
        
        try:
            # Retrieve relevant documents
            results = self.vector_store.search(question, top_k)
            
            if not results:
                logger.debug("No relevant documents found")
                return None
            
            # Build context from retrieved documents
            context_parts = []
            for doc, score in results:
                if score > 0.3:  # Similarity threshold
                    context_parts.append(f"Source: {doc.metadata.get('filename', 'Unknown')}")
                    context_parts.append(doc.content)
                    context_parts.append("---")
            
            if not context_parts:
                logger.debug("No documents meet similarity threshold")
                return None
            
            context = "\n".join(context_parts)
            
            # Generate response using LLM
            prompt = f"""
Based on the following knowledge base information, please answer the user's question. 
Use only the information provided in the context. If the context doesn't contain 
relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            
            response = self.llm_manager.generate(prompt, max_tokens=500)
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise RAGError(f"Failed to process query: {e}")
    
    def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add new knowledge to the system."""
        try:
            chunks = self._split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy() if metadata else {}
                doc_metadata['chunk_index'] = i
                doc_metadata['source'] = 'user_added'
                
                doc = Document(content=chunk, metadata=doc_metadata)
                documents.append(doc)
            
            added_count = self.vector_store.add_documents(documents)
            self.vector_store.save()
            
            logger.info(f"Added {added_count} knowledge chunks")
            return added_count > 0
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            'enabled': self.settings.get_ai_setting('rag.enabled', False),
            'document_count': len(self.vector_store.documents),
            'embedding_model': self.vector_store.embedding_model_name,
            'vector_store_path': str(self.vector_store.vector_path),
            'chunk_size': self.chunk_size,
            'faiss_available': FAISS_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }