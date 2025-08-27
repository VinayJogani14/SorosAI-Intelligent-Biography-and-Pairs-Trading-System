"""
RAG Pipeline: Retrieval-Augmented Generation for code Q&A and search
"""

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path
import json
from config import VECTOR_DB_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for code understanding and Q&A
    """
    
    def __init__(self, 
                 embedding_model_name: Optional[str] = None,
                 vector_db_path: Optional[str] = None):
        """
        Initialize the RAG pipeline
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            vector_db_path: Path to pre-built vector database
        """
        self.embedding_model_name = embedding_model_name or MODEL_CONFIG["embedding_model"]
        self.vector_db_path = vector_db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing RAG Pipeline with model: {self.embedding_model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self._load_models()
            self._initialize_vector_db()
            logger.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline: {str(e)}")
            raise
    
    def _load_models(self):
        """Load the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _initialize_vector_db(self):
        """Initialize or load the vector database"""
        try:
            if self.vector_db_path and Path(self.vector_db_path).exists():
                self._load_vector_db()
            else:
                self._create_empty_vector_db()
            logger.info("Vector database initialized")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
    
    def _create_empty_vector_db(self):
        """Create an empty vector database"""
        dimension = VECTOR_DB_CONFIG["dimension"]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.code_snippets = []
        self.explanations = []
        self.metadata = []
    
    def _load_vector_db(self):
        """Load pre-built vector database"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{self.vector_db_path}.index")
            
            # Load metadata
            with open(f"{self.vector_db_path}.meta", 'rb') as f:
                data = pickle.load(f)
                self.code_snippets = data['code_snippets']
                self.explanations = data['explanations']
                self.metadata = data['metadata']
                
            logger.info(f"Loaded vector database with {len(self.code_snippets)} code snippets")
            
        except Exception as e:
            logger.warning(f"Could not load vector database: {str(e)}")
            self._create_empty_vector_db()
    
    def add_code_snippets(self, code_snippets: List[str], 
                          explanations: List[str], 
                          metadata: Optional[List[Dict]] = None):
        """
        Add code snippets to the vector database
        
        Args:
            code_snippets: List of code strings
            explanations: List of corresponding explanations
            metadata: Optional list of metadata dictionaries
        """
        if len(code_snippets) != len(explanations):
            raise ValueError("Number of code snippets must match number of explanations")
        
        try:
            # Generate embeddings for code snippets
            embeddings = self.embedding_model.encode(code_snippets, show_progress_bar=True)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.code_snippets.extend(code_snippets)
            self.explanations.extend(explanations)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in code_snippets])
            
            logger.info(f"Added {len(code_snippets)} code snippets to vector database")
            
        except Exception as e:
            logger.error(f"Error adding code snippets: {str(e)}")
            raise
    
    def search_similar_code(self, code: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar code snippets
        
        Args:
            code: Code to search for
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar code snippets
        """
        try:
            # Generate embedding for input code
            query_embedding = self.embedding_model.encode([code])
            
            # Search in vector database
            similarities, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.code_snippets))
            )
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.code_snippets):
                    results.append({
                        'code': self.code_snippets[idx],
                        'explanation': self.explanations[idx],
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar code: {str(e)}")
            return []
    
    def search_by_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code snippets by natural language query
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing relevant code snippets
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in vector database
            similarities, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.code_snippets))
            )
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.code_snippets):
                    results.append({
                        'code': self.code_snippets[idx],
                        'explanation': self.explanations[idx],
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by query: {str(e)}")
            return []
    
    def get_answer(self, code: str, question: str) -> str:
        """
        Get answer to a question about specific code using RAG
        
        Args:
            code: Code context
            question: Question about the code
            
        Returns:
            Generated answer
        """
        try:
            # Search for similar code snippets
            similar_snippets = self.search_similar_code(code, top_k=3)
            
            if not similar_snippets:
                return "I couldn't find similar code examples to help answer your question."
            
            # Build context from similar snippets
            context = self._build_context(similar_snippets)
            
            # Generate answer using context
            answer = self._generate_answer(code, question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _build_context(self, similar_snippets: List[Dict[str, Any]]) -> str:
        """Build context string from similar code snippets"""
        context_parts = []
        
        for snippet in similar_snippets:
            context_parts.append(f"Code:\n{snippet['code']}\nExplanation:\n{snippet['explanation']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, code: str, question: str, context: str) -> str:
        """Generate answer using context and question"""
        # For now, return a simple template-based answer
        # In a full implementation, this would use a language model
        
        answer_template = f"""
Based on the code you provided and similar examples, here's what I can tell you:

**Your Code:**
```python
{code}
```

**Question:** {question}

**Answer:** The code appears to be a Python implementation. Based on similar examples in our database, this type of code typically {self._get_general_description(code)}.

**Context from Similar Examples:**
{context[:500]}...
"""
        
        return answer_template.strip()
    
    def _get_general_description(self, code: str) -> str:
        """Get a general description of the code type"""
        code_lower = code.lower()
        
        if 'def ' in code_lower:
            return "defines a function that performs a specific task"
        elif 'class ' in code_lower:
            return "defines a class with methods and attributes"
        elif 'import ' in code_lower:
            return "imports external libraries or modules"
        elif 'for ' in code_lower or 'while ' in code_lower:
            return "implements a loop or iteration"
        elif 'if ' in code_lower:
            return "implements conditional logic"
        else:
            return "performs some computational task"
    
    def save_vector_db(self, path: str):
        """Save the vector database to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save metadata
            metadata = {
                'code_snippets': self.code_snippets,
                'explanations': self.explanations,
                'metadata': self.metadata
            }
            
            with open(f"{path}.meta", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Vector database saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector database: {str(e)}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        return {
            'total_snippets': len(self.code_snippets),
            'embedding_dimension': VECTOR_DB_CONFIG["dimension"],
            'embedding_model': self.embedding_model_name,
            'index_type': VECTOR_DB_CONFIG["index_type"]
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            if hasattr(self, 'index'):
                del self.index
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
