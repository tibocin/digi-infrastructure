"""
Filepath: pcs/src/pcs/repositories/neo4j_repo.py
Purpose: Neo4j repository implementation for graph database operations and relationship modeling
Related Components: Neo4j driver, context relationships, graph queries
Tags: neo4j, graph-database, relationships, cypher, async
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

from .base import RepositoryError


class Neo4jRepository:
    """
    Neo4j repository for graph database operations.
    
    Provides async operations for:
    - Node creation and querying
    - Relationship management
    - Cypher query execution
    - Graph traversal operations
    """

    def __init__(self, driver: AsyncDriver):
        """
        Initialize Neo4j repository with driver connection.
        
        Args:
            driver: Async Neo4j driver instance
        """
        self.driver = driver

    async def execute_cypher(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            RepositoryError: If query execution fails
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except Neo4jError as e:
            raise RepositoryError(f"Failed to execute Cypher query: {str(e)}") from e

    async def create_node(
        self, 
        label: str, 
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a node with label and properties.
        
        Args:
            label: Node label
            properties: Node properties
            
        Returns:
            Created node data
        """
        try:
            query = f"CREATE (n:{label} $props) RETURN n"
            result = await self.execute_cypher(query, {"props": properties})
            return result[0]["n"] if result else {}
        except Exception as e:
            raise RepositoryError(f"Failed to create node with label {label}: {str(e)}") from e

    async def get_node_by_id(self, node_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data if found, None otherwise
        """
        try:
            query = "MATCH (n) WHERE n.id = $id RETURN n"
            result = await self.execute_cypher(query, {"id": str(node_id)})
            return result[0]["n"] if result else None
        except Exception as e:
            raise RepositoryError(f"Failed to get node by ID {node_id}: {str(e)}") from e

    async def create_relationship(
        self, 
        from_node_id: Union[str, UUID],
        to_node_id: Union[str, UUID],
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Created relationship data
        """
        try:
            if properties:
                query = """
                MATCH (from), (to)
                WHERE from.id = $from_id AND to.id = $to_id
                CREATE (from)-[r:%s $props]->(to)
                RETURN r
                """ % relationship_type
                params = {
                    "from_id": str(from_node_id),
                    "to_id": str(to_node_id),
                    "props": properties
                }
            else:
                query = """
                MATCH (from), (to)
                WHERE from.id = $from_id AND to.id = $to_id
                CREATE (from)-[r:%s]->(to)
                RETURN r
                """ % relationship_type
                params = {
                    "from_id": str(from_node_id),
                    "to_id": str(to_node_id)
                }
            
            result = await self.execute_cypher(query, params)
            return result[0]["r"] if result else {}
        except Exception as e:
            raise RepositoryError(f"Failed to create relationship: {str(e)}") from e

    async def find_relationships(
        self, 
        node_id: Union[str, UUID],
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Find relationships for a node.
        
        Args:
            node_id: Node identifier
            relationship_type: Type of relationship (optional)
            direction: Relationship direction ("incoming", "outgoing", "both")
            
        Returns:
            List of relationship data
        """
        try:
            if direction == "incoming":
                pattern = "<-[r%s]-"
            elif direction == "outgoing":
                pattern = "-[r%s]->"
            else:  # both
                pattern = "-[r%s]-"
            
            rel_filter = f":{relationship_type}" if relationship_type else ""
            query = f"""
            MATCH (n){pattern % rel_filter}(other)
            WHERE n.id = $id
            RETURN r, other
            """
            
            result = await self.execute_cypher(query, {"id": str(node_id)})
            return result
        except Exception as e:
            raise RepositoryError(f"Failed to find relationships for node {node_id}: {str(e)}") from e

    async def delete_node(self, node_id: Union[str, UUID]) -> bool:
        """
        Delete a node and all its relationships.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node was deleted
        """
        try:
            query = """
            MATCH (n)
            WHERE n.id = $id
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            result = await self.execute_cypher(query, {"id": str(node_id)})
            return result[0]["deleted_count"] > 0 if result else False
        except Exception as e:
            raise RepositoryError(f"Failed to delete node {node_id}: {str(e)}") from e

    async def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
