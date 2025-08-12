"""
Filepath: pcs/src/pcs/repositories/chroma_repo.py
Purpose: ChromaDB repository implementation for vector database operations and semantic search
Related Components: ChromaDB client, embedding operations, similarity search
Tags: chromadb, vector-database, embeddings, similarity-search, semantic
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import chromadb

from .base import RepositoryError


class ChromaRepository:
    """
    ChromaDB repository for vector database operations.
    
    Provides operations for:
    - Document storage with embeddings
    - Similarity search
    - Metadata filtering
    - Collection management
    """

    def __init__(self, client: chromadb.Client):
        """
        Initialize ChromaDB repository with client connection.
        
        Args:
            client: ChromaDB client instance
        """
        self.client = client

    async def create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata
            
        Returns:
            Created collection
            
        Raises:
            RepositoryError: If collection creation fails
        """
        try:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            return collection
        except Exception as e:
            raise RepositoryError(f"Failed to create collection {name}: {str(e)}") from e

    async def get_collection(self, name: str) -> Optional[Any]:
        """
        Get an existing collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection if found, None otherwise
        """
        try:
            collection = self.client.get_collection(name=name)
            return collection
        except Exception:
            return None

    async def get_or_create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get existing collection or create if it doesn't exist.
        
        Args:
            name: Collection name
            metadata: Collection metadata for creation
            
        Returns:
            Collection instance
        """
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
            return collection
        except Exception as e:
            raise RepositoryError(f"Failed to get or create collection {name}: {str(e)}") from e

    async def add_documents(
        self,
        collection: Any,
        documents: Any,
        ids: List[str],
        metadatas: Optional[Any] = None,
        embeddings: Optional[Any] = None
    ) -> bool:
        """
        Add documents to a collection.
        
        Args:
            collection: ChromaDB collection
            documents: List of document texts
            ids: List of document IDs
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional pre-computed embeddings
            
        Returns:
            True if successful
        """
        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to add documents to collection: {str(e)}") from e

    async def query_documents(
        self,
        collection: Any,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[Any] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query documents from a collection.
        
        Args:
            collection: ChromaDB collection
            query_texts: Text queries
            query_embeddings: Embedding queries
            n_results: Number of results to return
            where: Metadata filters
            where_document: Document content filters
            
        Returns:
            Query results dictionary
        """
        try:
            results = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            raise RepositoryError(f"Failed to query collection: {str(e)}") from e

    async def get_documents(
        self,
        collection: Any,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get documents from a collection.
        
        Args:
            collection: ChromaDB collection
            ids: Specific document IDs to retrieve
            where: Metadata filters
            where_document: Document content filters
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Documents dictionary
        """
        try:
            results = collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                limit=limit,
                offset=offset
            )
            return results
        except Exception as e:
            raise RepositoryError(f"Failed to get documents from collection: {str(e)}") from e

    async def update_documents(
        self,
        collection: Any,
        ids: List[str],
        documents: Optional[Any] = None,
        metadatas: Optional[Any] = None,
        embeddings: Optional[Any] = None
    ) -> bool:
        """
        Update existing documents in a collection.
        
        Args:
            collection: ChromaDB collection
            ids: Document IDs to update
            documents: New document texts
            metadatas: New metadata
            embeddings: New embeddings
            
        Returns:
            True if successful
        """
        try:
            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to update documents in collection: {str(e)}") from e

    async def delete_documents(
        self,
        collection: Any,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from a collection.
        
        Args:
            collection: ChromaDB collection
            ids: Specific document IDs to delete
            where: Metadata filters
            where_document: Document content filters
            
        Returns:
            True if successful
        """
        try:
            collection.delete(
                ids=ids,
                where=where,
                where_document=where_document
            )
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to delete documents from collection: {str(e)}") from e

    async def count_documents(self, collection: Collection) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection: ChromaDB collection
            
        Returns:
            Number of documents
        """
        try:
            return collection.count()
        except Exception as e:
            raise RepositoryError(f"Failed to count documents in collection: {str(e)}") from e

    async def similarity_search(
        self,
        collection: Any,
        query_text: str,
        n_results: int = 5,
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search with optional filtering.
        
        Args:
            collection: ChromaDB collection
            query_text: Text to search for
            n_results: Number of results to return
            threshold: Similarity threshold (0-1)
            metadata_filter: Metadata-based filtering
            
        Returns:
            List of similar documents with scores
        """
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=metadata_filter
            )
            
            # Format results with scores
            formatted_results = []
            if results and results.get('documents') and results.get('distances'):
                for i, (doc, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (1 - distance for cosine)
                    similarity = 1 - distance
                    
                    if threshold is None or similarity >= threshold:
                        result = {
                            'document': doc,
                            'similarity': similarity,
                            'distance': distance,
                            'id': results['ids'][0][i] if results.get('ids') else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else None
                        }
                        formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            raise RepositoryError(f"Failed to perform similarity search: {str(e)}") from e

    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=name)
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to delete collection {name}: {str(e)}") from e
