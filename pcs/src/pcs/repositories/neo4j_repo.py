"""
Filepath: pcs/src/pcs/repositories/neo4j_repo.py
Purpose: Neo4j repository implementation for graph database operations and relationship modeling
Related Components: Neo4j driver, context relationships, graph queries
Tags: neo4j, graph-database, relationships, cypher, async
"""

import time
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
from enum import Enum

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

from .base import RepositoryError
from ..utils.metrics import PerformanceMonitor


class RelationshipType(Enum):
    """Enumeration of supported relationship types."""
    DEPENDS_ON = "DEPENDS_ON"
    CONTAINS = "CONTAINS"
    RELATED_TO = "RELATED_TO"
    DERIVES_FROM = "DERIVES_FROM"
    USES = "USES"
    TRIGGERS = "TRIGGERS"
    FOLLOWS = "FOLLOWS"
    REFERENCES = "REFERENCES"
    CHILD_OF = "CHILD_OF"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    PRECEDES = "PRECEDES"


@dataclass
class GraphNode:
    """Represents a graph node with properties."""
    id: UUID
    label: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "label": self.label,
            "properties": self.properties
        }


@dataclass
class Relationship:
    """Represents a graph relationship."""
    id: Optional[str]
    from_node_id: UUID
    to_node_id: UUID
    type: str
    properties: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "from_node_id": str(self.from_node_id),
            "to_node_id": str(self.to_node_id),
            "type": self.type,
            "properties": self.properties,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ConversationPattern:
    """Represents conversation pattern analysis results."""
    user_id: UUID
    pattern_type: str
    frequency: int
    avg_response_time: float
    common_topics: List[str]
    interaction_flow: List[Dict[str, Any]]
    temporal_patterns: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": str(self.user_id),
            "pattern_type": self.pattern_type,
            "frequency": self.frequency,
            "avg_response_time": self.avg_response_time,
            "common_topics": self.common_topics,
            "interaction_flow": self.interaction_flow,
            "temporal_patterns": self.temporal_patterns
        }


@dataclass
class GraphTraversalResult:
    """Results from graph traversal operations."""
    nodes: List[GraphNode]
    relationships: List[Relationship]
    paths: List[List[Dict[str, Any]]]
    metadata: Dict[str, Any]


class Neo4jRepository:
    """
    Enhanced Neo4j repository for graph database operations and relationship modeling.
    
    Features:
    - Context relationships and hierarchies
    - Prompt template dependencies
    - User interaction patterns
    - Conversation flow modeling
    - Advanced graph traversal and analysis
    - Performance monitoring and optimization
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
        parameters: Optional[Dict[str, Any]] = None,
        monitor_performance: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results with optional performance monitoring.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            monitor_performance: Whether to track query performance
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            RepositoryError: If query execution fails
        """
        try:
            if monitor_performance:
                async with PerformanceMonitor("cypher_query", "neo4j") as monitor:
                    async with self.driver.session() as session:
                        result = await session.run(query, parameters or {})
                        records = await result.data()
                        monitor.set_rows_affected(len(records))
                        return records
            else:
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
    ) -> GraphNode:
        """
        Create a node with label and properties.
        
        Args:
            label: Node label
            properties: Node properties
            
        Returns:
            Created GraphNode instance
        """
        try:
            # Ensure we have an ID and created_at timestamp
            if "id" not in properties:
                from uuid import uuid4
                properties["id"] = str(uuid4())
            if "created_at" not in properties:
                properties["created_at"] = datetime.now(UTC).isoformat()
            
            query = f"CREATE (n:{label} $props) RETURN n"
            result = await self.execute_cypher(query, {"props": properties})
            
            if result:
                node_data = result[0]["n"]
                return GraphNode(
                    id=UUID(node_data["id"]),
                    label=label,
                    properties=node_data
                )
            else:
                raise RepositoryError("Failed to create node - no result returned")
        except Exception as e:
            raise RepositoryError(f"Failed to create node with label {label}: {str(e)}") from e

    async def get_node_by_id(self, node_id: Union[str, UUID]) -> Optional[GraphNode]:
        """
        Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            GraphNode if found, None otherwise
        """
        try:
            query = "MATCH (n) WHERE n.id = $id RETURN n, labels(n) as labels"
            result = await self.execute_cypher(query, {"id": str(node_id)})
            
            if result:
                node_data = result[0]["n"]
                labels = result[0]["labels"]
                return GraphNode(
                    id=UUID(node_data["id"]),
                    label=labels[0] if labels else "Unknown",
                    properties=node_data
                )
            return None
        except Exception as e:
            raise RepositoryError(f"Failed to get node by ID {node_id}: {str(e)}") from e

    async def create_relationship(
        self,
        from_node_id: UUID,
        to_node_id: UUID,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Relationship:
        """
        Create a directed relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Created Relationship instance
            
        Raises:
            RepositoryError: If relationship creation fails
        """
        try:
            rel_props = properties or {}
            rel_props["created_at"] = datetime.now(UTC).isoformat()
            
            query = f"""
            MATCH (from), (to)
            WHERE from.id = $from_id AND to.id = $to_id
            CREATE (from)-[r:{relationship_type} $props]->(to)
            RETURN r, id(r) as rel_id
            """
            params = {
                "from_id": str(from_node_id),
                "to_id": str(to_node_id),
                "props": rel_props
            }
            
            result = await self.execute_cypher(query, params)
            if result:
                rel_data = result[0]["r"]
                rel_id = result[0]["rel_id"]
                
                return Relationship(
                    id=str(rel_id),
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    type=relationship_type,
                    properties=rel_data,
                    created_at=datetime.fromisoformat(rel_data["created_at"])
                )
            else:
                raise RepositoryError("Failed to create relationship - no result returned")
        except Exception as e:
            raise RepositoryError(f"Failed to create relationship: {str(e)}") from e

    async def find_related_nodes(
        self,
        node_id: UUID,
        relationship_type: str,
        max_depth: int = 3,
        direction: str = "outgoing"
    ) -> List[GraphNode]:
        """
        Find all nodes related through specified relationship type.
        
        Args:
            node_id: Starting node ID
            relationship_type: Type of relationship to follow
            max_depth: Maximum traversal depth
            direction: Relationship direction ("incoming", "outgoing", "both")
            
        Returns:
            List of related GraphNode instances
        """
        try:
            # Build direction pattern
            if direction == "incoming":
                pattern = f"<-[r:{relationship_type}*1..{max_depth}]-"
            elif direction == "outgoing":
                pattern = f"-[r:{relationship_type}*1..{max_depth}]->"
            else:  # both
                pattern = f"-[r:{relationship_type}*1..{max_depth}]-"
            
            query = f"""
            MATCH (start){pattern}(related)
            WHERE start.id = $start_id
            RETURN DISTINCT related, labels(related) as labels
            """
            
            result = await self.execute_cypher(query, {"start_id": str(node_id)})
            
            nodes = []
            for record in result:
                node_data = record["related"]
                labels = record["labels"]
                nodes.append(GraphNode(
                    id=UUID(node_data["id"]),
                    label=labels[0] if labels else "Unknown",
                    properties=node_data
                ))
            
            return nodes
        except Exception as e:
            raise RepositoryError(f"Failed to find related nodes: {str(e)}") from e

    async def analyze_conversation_patterns(
        self,
        user_id: UUID,
        time_range: Tuple[datetime, datetime]
    ) -> ConversationPattern:
        """
        Analyze user conversation patterns using graph traversal.
        
        Args:
            user_id: User identifier
            time_range: Time range for analysis (start, end)
            
        Returns:
            ConversationPattern analysis results
        """
        try:
            start_time, end_time = time_range
            
            # Query for conversation interactions
            query = """
            MATCH (user:User {id: $user_id})-[r:PARTICIPATES_IN]->(conv:Conversation)
            WHERE conv.created_at >= $start_time AND conv.created_at <= $end_time
            MATCH (conv)-[:CONTAINS]->(msg:Message)
            RETURN conv, collect(msg) as messages, count(msg) as msg_count
            ORDER BY conv.created_at
            """
            
            params = {
                "user_id": str(user_id),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            result = await self.execute_cypher(query, params)
            
            # Analyze patterns
            conversation_count = len(result)
            total_messages = sum(record["msg_count"] for record in result)
            
            # Calculate response times and extract topics
            response_times = []
            topics = []
            interaction_flow = []
            
            for record in result:
                conv = record["conv"]
                messages = record["messages"]
                
                # Extract topics from conversation
                if "topic" in conv:
                    topics.append(conv["topic"])
                
                # Build interaction flow
                interaction_flow.append({
                    "conversation_id": conv["id"],
                    "message_count": len(messages),
                    "duration": conv.get("duration", 0),
                    "topics": [msg.get("topic", "") for msg in messages if msg.get("topic")]
                })
                
                # Calculate response times
                if len(messages) > 1:
                    for i in range(1, len(messages)):
                        prev_time = datetime.fromisoformat(messages[i-1]["created_at"])
                        curr_time = datetime.fromisoformat(messages[i]["created_at"])
                        response_times.append((curr_time - prev_time).total_seconds())
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            common_topics = list(set(topics))
            
            # Temporal patterns
            temporal_patterns = {
                "peak_hours": self._analyze_peak_hours(result),
                "conversation_frequency": conversation_count / 7 if conversation_count > 0 else 0,  # per week
                "avg_messages_per_conversation": total_messages / conversation_count if conversation_count > 0 else 0
            }
            
            return ConversationPattern(
                user_id=user_id,
                pattern_type="general",
                frequency=conversation_count,
                avg_response_time=avg_response_time,
                common_topics=common_topics,
                interaction_flow=interaction_flow,
                temporal_patterns=temporal_patterns
            )
            
        except Exception as e:
            raise RepositoryError(f"Failed to analyze conversation patterns: {str(e)}") from e

    async def create_context_hierarchy(
        self,
        parent_context_id: UUID,
        child_context_id: UUID,
        hierarchy_type: str = "CHILD_OF"
    ) -> Relationship:
        """
        Create a hierarchical relationship between contexts.
        
        Args:
            parent_context_id: Parent context node ID
            child_context_id: Child context node ID
            hierarchy_type: Type of hierarchical relationship
            
        Returns:
            Created relationship
        """
        return await self.create_relationship(
            from_node_id=child_context_id,
            to_node_id=parent_context_id,
            relationship_type=hierarchy_type,
            properties={"hierarchy_level": 1}
        )

    async def find_context_dependencies(
        self,
        context_id: UUID,
        max_depth: int = 5
    ) -> GraphTraversalResult:
        """
        Find all dependencies for a given context.
        
        Args:
            context_id: Context node ID
            max_depth: Maximum dependency depth to traverse
            
        Returns:
            Graph traversal results with dependencies
        """
        try:
            query = f"""
            MATCH path = (context:Context {{id: $context_id}})-[r:DEPENDS_ON*1..{max_depth}]->(dep:Context)
            RETURN path, nodes(path) as path_nodes, relationships(path) as path_rels
            """
            
            result = await self.execute_cypher(query, {"context_id": str(context_id)})
            
            nodes = []
            relationships = []
            paths = []
            
            for record in result:
                path_nodes = record["path_nodes"]
                path_rels = record["path_rels"]
                
                # Convert nodes
                for node_data in path_nodes:
                    if node_data["id"] not in [n.id for n in nodes]:
                        nodes.append(GraphNode(
                            id=UUID(node_data["id"]),
                            label="Context",
                            properties=node_data
                        ))
                
                # Convert relationships
                for rel_data in path_rels:
                    relationships.append(Relationship(
                        id=str(rel_data.id),
                        from_node_id=UUID(rel_data.start_node["id"]),
                        to_node_id=UUID(rel_data.end_node["id"]),
                        type=rel_data.type,
                        properties=dict(rel_data),
                        created_at=datetime.fromisoformat(rel_data.get("created_at", datetime.now(UTC).isoformat()))
                    ))
                
                # Add path
                paths.append([dict(node) for node in path_nodes])
            
            return GraphTraversalResult(
                nodes=nodes,
                relationships=relationships,
                paths=paths,
                metadata={"max_depth": max_depth, "total_paths": len(paths)}
            )
            
        except Exception as e:
            raise RepositoryError(f"Failed to find context dependencies: {str(e)}") from e

    async def create_prompt_template_dependency(
        self,
        template_id: UUID,
        dependent_template_id: UUID,
        dependency_type: str = "DEPENDS_ON"
    ) -> Relationship:
        """
        Create a dependency relationship between prompt templates.
        
        Args:
            template_id: Source template ID
            dependent_template_id: Dependent template ID
            dependency_type: Type of dependency
            
        Returns:
            Created relationship
        """
        return await self.create_relationship(
            from_node_id=dependent_template_id,
            to_node_id=template_id,
            relationship_type=dependency_type,
            properties={"dependency_strength": 1.0}
        )

    async def find_similar_conversation_patterns(
        self,
        pattern: ConversationPattern,
        similarity_threshold: float = 0.8
    ) -> List[ConversationPattern]:
        """
        Find conversation patterns similar to the given pattern.
        
        Args:
            pattern: Reference conversation pattern
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar conversation patterns
        """
        try:
            # Query for users with similar conversation characteristics
            query = """
            MATCH (user:User)-[r:PARTICIPATES_IN]->(conv:Conversation)
            WITH user, count(conv) as conv_count, avg(conv.duration) as avg_duration
            WHERE conv_count >= $min_frequency 
            AND abs(avg_duration - $target_duration) <= $duration_tolerance
            RETURN user, conv_count, avg_duration
            """
            
            params = {
                "min_frequency": int(pattern.frequency * similarity_threshold),
                "target_duration": pattern.avg_response_time,
                "duration_tolerance": pattern.avg_response_time * (1 - similarity_threshold)
            }
            
            result = await self.execute_cypher(query, params)
            
            similar_patterns = []
            for record in result:
                user_data = record["user"]
                user_id = UUID(user_data["id"])
                
                # Skip the same user
                if user_id == pattern.user_id:
                    continue
                
                # Analyze this user's pattern
                similar_pattern = await self.analyze_conversation_patterns(
                    user_id=user_id,
                    time_range=(
                        datetime.now(UTC) - timedelta(days=30),
                        datetime.now(UTC)
                    )
                )
                similar_patterns.append(similar_pattern)
            
            return similar_patterns
            
        except Exception as e:
            raise RepositoryError(f"Failed to find similar conversation patterns: {str(e)}") from e

    async def get_relationship_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about relationships in the graph.
        
        Returns:
            Dictionary with relationship statistics
        """
        try:
            query = """
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY count DESC
            """
            
            result = await self.execute_cypher(query)
            
            # Node count query
            node_query = "MATCH (n) RETURN labels(n) as label, count(n) as count"
            node_result = await self.execute_cypher(node_query)
            
            return {
                "relationship_counts": {record["relationship_type"]: record["count"] for record in result},
                "node_counts": {record["label"][0] if record["label"] else "Unknown": record["count"] for record in node_result},
                "total_relationships": sum(record["count"] for record in result),
                "total_nodes": sum(record["count"] for record in node_result)
            }
            
        except Exception as e:
            raise RepositoryError(f"Failed to get relationship statistics: {str(e)}") from e

    async def optimize_graph_performance(self) -> Dict[str, Any]:
        """
        Optimize graph performance by creating indexes and constraints.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            optimization_results = {}
            
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX node_id_index IF NOT EXISTS FOR (n) ON (n.id)",
                "CREATE INDEX context_type_index IF NOT EXISTS FOR (c:Context) ON (c.type)",
                "CREATE INDEX conversation_created_index IF NOT EXISTS FOR (c:Conversation) ON (c.created_at)",
                "CREATE INDEX message_timestamp_index IF NOT EXISTS FOR (m:Message) ON (m.created_at)",
                "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.id)"
            ]
            
            for index_query in indexes:
                try:
                    await self.execute_cypher(index_query, monitor_performance=False)
                    optimization_results[index_query] = "created"
                except Exception as e:
                    optimization_results[index_query] = f"failed: {str(e)}"
            
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT unique_node_id IF NOT EXISTS FOR (n) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT unique_user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT unique_conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE"
            ]
            
            for constraint_query in constraints:
                try:
                    await self.execute_cypher(constraint_query, monitor_performance=False)
                    optimization_results[constraint_query] = "created"
                except Exception as e:
                    optimization_results[constraint_query] = f"failed: {str(e)}"
            
            return optimization_results
            
        except Exception as e:
            raise RepositoryError(f"Failed to optimize graph performance: {str(e)}") from e

    def _analyze_peak_hours(self, conversation_data: List[Dict[str, Any]]) -> List[int]:
        """
        Analyze peak conversation hours from conversation data.
        
        Args:
            conversation_data: List of conversation records
            
        Returns:
            List of peak hours (0-23)
        """
        hour_counts = {}
        
        for record in conversation_data:
            conv = record["conv"]
            created_at = datetime.fromisoformat(conv["created_at"])
            hour = created_at.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if not hour_counts:
            return []
        
        # Find hours with above-average activity
        avg_count = sum(hour_counts.values()) / len(hour_counts)
        peak_hours = [hour for hour, count in hour_counts.items() if count > avg_count]
        
        return sorted(peak_hours)

    # Legacy methods for backward compatibility
    async def find_relationships(
        self, 
        node_id: Union[str, UUID],
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Find relationships for a node (legacy method for backward compatibility).
        
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
