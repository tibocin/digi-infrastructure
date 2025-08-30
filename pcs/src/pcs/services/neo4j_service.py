"""
Filepath: pcs/src/pcs/services/neo4j_service.py
Purpose: Neo4j service for PCS reasoning chains and cross-application intelligence
Related Components: Neo4j driver, reasoning engine, cross-app intelligence, prompt optimization
Tags: neo4j, reasoning, cross-app, intelligence, graph-database, prompt-optimization
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import asyncio
import logging

from ..core.exceptions import PCSError


class Neo4jServiceError(PCSError):
    """Custom exception for Neo4j service errors."""
    pass


class ReasoningType(Enum):
    """Types of reasoning that can be applied."""
    QUERY_ANALYSIS = "query_analysis"
    CONTEXT_OPTIMIZATION = "context_optimization"
    PROMPT_SELECTION = "prompt_selection"
    CROSS_APP_INTELLIGENCE = "cross_app_intelligence"
    SUCCESS_PATTERN_ANALYSIS = "success_pattern_analysis"


@dataclass
class ReasoningChain:
    """A reasoning chain from query to prompt."""
    query_id: str
    reasoning_id: str
    prompt_id: str
    confidence: float
    reasoning_type: ReasoningType
    metadata: Dict[str, Any]
    created_at: datetime
    success_rate: Optional[float] = None
    usage_count: int = 0


@dataclass
class ContextOptimization:
    """Context optimization result."""
    original_context_id: str
    optimized_context_id: str
    optimization_type: str
    compression_ratio: float
    performance_improvement: float
    trigger_condition: str
    metadata: Dict[str, Any]


class PCSNeo4jService:
    """
    Neo4j service for PCS reasoning and cross-application intelligence.
    
    This service provides:
    1. Reasoning chain tracking (Query → Reasoning → Prompt)
    2. Context optimization analysis
    3. Cross-application pattern recognition
    4. Success rate tracking and optimization
    """
    
    def __init__(self, neo4j_uri: str, username: str, password: str, database: str = "pcs"):
        """
        Initialize Neo4j service.
        
        Args:
            neo4j_uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: pcs)
        """
        self.neo4j_uri = neo4j_uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_operations = 0
        self.successful_operations = 0
    
    async def initialize(self) -> bool:
        """Initialize Neo4j connection and create schema."""
        try:
            # Import neo4j driver (will be installed via requirements)
            from neo4j import GraphDatabase
            
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            await self._test_connection()
            
            # Create schema
            await self._create_schema()
            
            self.logger.info("Neo4j service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j service: {str(e)}")
            raise Neo4jServiceError(f"Initialization failed: {str(e)}") from e
    
    async def _test_connection(self) -> bool:
        """Test Neo4j connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                return record["test"] == 1
        except Exception as e:
            raise Neo4jServiceError(f"Connection test failed: {str(e)}") from e
    
    async def _create_schema(self) -> bool:
        """Create Neo4j schema for PCS."""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints and indexes
                schema_queries = [
                    # Unique constraints
                    "CREATE CONSTRAINT query_id_unique IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE",
                    "CREATE CONSTRAINT reasoning_id_unique IF NOT EXISTS FOR (r:Reasoning) REQUIRE r.id IS UNIQUE",
                    "CREATE CONSTRAINT prompt_id_unique IF NOT EXISTS FOR (p:Prompt) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (c:Context) REQUIRE c.id IS UNIQUE",
                    
                    # Indexes for performance
                    "CREATE INDEX query_timestamp_index IF NOT EXISTS FOR (q:Query) ON (q.timestamp)",
                    "CREATE INDEX reasoning_type_index IF NOT EXISTS FOR (r:Reasoning) ON (r.type)",
                    "CREATE INDEX prompt_category_index IF NOT EXISTS FOR (p:Prompt) ON (p.category)",
                    "CREATE INDEX context_scope_index IF NOT EXISTS FOR (c:Context) ON (c.scope)"
                ]
                
                for query in schema_queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Some constraints might already exist, that's okay
                        self.logger.debug(f"Schema creation query failed (likely already exists): {str(e)}")
                
                return True
                
        except Exception as e:
            raise Neo4jServiceError(f"Schema creation failed: {str(e)}") from e
    
    async def track_reasoning_chain(
        self,
        query_text: str,
        reasoning_approach: str,
        prompt_id: str,
        app_id: str,
        user_context: Dict[str, Any],
        confidence: float = 0.8
    ) -> str:
        """
        Track a reasoning chain from query to prompt.
        
        Args:
            query_text: The user query
            reasoning_approach: The reasoning approach used
            prompt_id: The selected prompt ID
            app_id: Application identifier
            user_context: User context data
            confidence: Confidence in the reasoning
            
        Returns:
            The reasoning chain ID
        """
        self.total_operations += 1
        
        try:
            reasoning_id = str(uuid4())
            timestamp = datetime.now(timezone.utc)
            
            with self.driver.session(database=self.database) as session:
                # Create query node
                query_query = """
                MERGE (q:Query {id: $query_id})
                SET q.text = $query_text,
                    q.timestamp = $timestamp,
                    q.app_id = $app_id,
                    q.user_context = $user_context
                """
                
                session.run(query_query, {
                    "query_id": f"query_{hash(query_text) % 1000000}",
                    "query_text": query_text,
                    "timestamp": timestamp.isoformat(),
                    "app_id": app_id,
                    "user_context": user_context
                })
                
                # Create reasoning node
                reasoning_query = """
                CREATE (r:Reasoning {
                    id: $reasoning_id,
                    approach: $approach,
                    confidence: $confidence,
                    timestamp: $timestamp,
                    app_id: $app_id
                })
                """
                
                session.run(reasoning_query, {
                    "reasoning_id": reasoning_id,
                    "approach": reasoning_approach,
                    "confidence": confidence,
                    "timestamp": timestamp.isoformat(),
                    "app_id": app_id
                })
                
                # Create prompt node
                prompt_query = """
                MERGE (p:Prompt {id: $prompt_id})
                SET p.app_id = $app_id,
                    p.last_used = $timestamp
                """
                
                session.run(prompt_query, {
                    "prompt_id": prompt_id,
                    "app_id": app_id,
                    "timestamp": timestamp.isoformat()
                })
                
                # Create relationships
                relationships_query = """
                MATCH (q:Query {id: $query_id})
                MATCH (r:Reasoning {id: $reasoning_id})
                MATCH (p:Prompt {id: $prompt_id})
                MERGE (q)-[:REQUIRES_REASONING {confidence: $confidence}]->(r)
                MERGE (r)-[:GENERATES_PROMPT {timestamp: $timestamp}]->(p)
                """
                
                session.run(relationships_query, {
                    "query_id": f"query_{hash(query_text) % 1000000}",
                    "reasoning_id": reasoning_id,
                    "prompt_id": prompt_id,
                    "confidence": confidence,
                    "timestamp": timestamp.isoformat()
                })
                
                self.successful_operations += 1
                return reasoning_id
                
        except Exception as e:
            self.logger.error(f"Failed to track reasoning chain: {str(e)}")
            raise Neo4jServiceError(f"Reasoning chain tracking failed: {str(e)}") from e
    
    async def analyze_success_patterns(
        self,
        app_id: str,
        time_window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Analyze success patterns for prompts and reasoning approaches.
        
        Args:
            app_id: Application identifier
            time_window_days: Number of days to analyze
            
        Returns:
            List of success pattern analysis results
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Query for success patterns
                query = """
                MATCH (q:Query)-[:REQUIRES_REASONING]->(r:Reasoning)-[:GENERATES_PROMPT]->(p:Prompt)
                WHERE q.app_id = $app_id
                AND q.timestamp > datetime() - duration({days: $days})
                WITH r.approach, p.id as prompt_id, count(*) as usage_count
                RETURN r.approach as reasoning_approach,
                       prompt_id,
                       usage_count,
                       usage_count as success_indicator
                ORDER BY usage_count DESC
                LIMIT 10
                """
                
                result = session.run(query, {
                    "app_id": app_id,
                    "days": time_window_days
                })
                
                patterns = []
                for record in result:
                    patterns.append({
                        "reasoning_approach": record["reasoning_approach"],
                        "prompt_id": record["prompt_id"],
                        "usage_count": record["usage_count"],
                        "success_indicator": record["success_indicator"]
                    })
                
                return patterns
                
        except Exception as e:
            self.logger.error(f"Failed to analyze success patterns: {str(e)}")
            raise Neo4jServiceError(f"Success pattern analysis failed: {str(e)}") from e
    
    async def find_cross_app_insights(
        self,
        query_intent: str,
        current_app_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find insights from other applications that could help with the current query.
        
        Args:
            query_intent: The intent of the current query
            current_app_id: Current application ID
            limit: Maximum number of insights to return
            
        Returns:
            List of cross-application insights
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Query for cross-app insights
                query = """
                MATCH (q:Query)-[:REQUIRES_REASONING]->(r:Reasoning)-[:GENERATES_PROMPT]->(p:Prompt)
                WHERE q.app_id <> $current_app_id
                AND q.text CONTAINS $query_intent
                WITH q.app_id as source_app,
                     r.approach as reasoning_approach,
                     p.id as prompt_id,
                     count(*) as success_count
                RETURN source_app,
                       reasoning_approach,
                       prompt_id,
                       success_count
                ORDER BY success_count DESC
                LIMIT $limit
                """
                
                result = session.run(query, {
                    "current_app_id": current_app_id,
                    "query_intent": query_intent,
                    "limit": limit
                })
                
                insights = []
                for record in result:
                    insights.append({
                        "source_app": record["source_app"],
                        "reasoning_approach": record["reasoning_approach"],
                        "prompt_id": record["prompt_id"],
                        "success_count": record["success_count"]
                    })
                
                return insights
                
        except Exception as e:
            self.logger.error(f"Failed to find cross-app insights: {str(e)}")
            raise Neo4jServiceError(f"Cross-app insight discovery failed: {str(e)}") from e
    
    async def optimize_context(
        self,
        context_id: str,
        trigger_condition: str,
        optimization_type: str
    ) -> ContextOptimization:
        """
        Track context optimization for analysis.
        
        Args:
            context_id: Original context ID
            trigger_condition: What triggered the optimization
            optimization_type: Type of optimization applied
            
        Returns:
            Context optimization result
        """
        try:
            optimized_context_id = str(uuid4())
            timestamp = datetime.now(timezone.utc)
            
            with self.driver.session(database=self.database) as session:
                # Create context optimization relationship
                query = """
                MATCH (c:Context {id: $context_id})
                CREATE (oc:Context {id: $optimized_id})
                CREATE (c)-[:OPTIMIZES_TO {
                    trigger_condition: $trigger,
                    optimization_type: $type,
                    timestamp: $timestamp
                }]->(oc)
                """
                
                session.run(query, {
                    "context_id": context_id,
                    "optimized_id": optimized_context_id,
                    "trigger": trigger_condition,
                    "type": optimization_type,
                    "timestamp": timestamp.isoformat()
                })
                
                # For now, return placeholder metrics
                # In production, these would be calculated based on actual performance
                optimization = ContextOptimization(
                    original_context_id=context_id,
                    optimized_context_id=optimized_context_id,
                    optimization_type=optimization_type,
                    compression_ratio=0.6,  # Placeholder
                    performance_improvement=0.25,  # Placeholder
                    trigger_condition=trigger_condition,
                    metadata={
                        "timestamp": timestamp.isoformat(),
                        "app_id": "pcs"
                    }
                )
                
                return optimization
                
        except Exception as e:
            self.logger.error(f"Failed to track context optimization: {str(e)}")
            raise Neo4jServiceError(f"Context optimization tracking failed: {str(e)}") from e
    
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get reasoning and optimization statistics."""
        try:
            with self.driver.session(database=self.database) as session:
                # Get basic counts
                counts_query = """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                """
                
                result = session.run(counts_query)
                counts = {}
                for record in result:
                    counts[record["label"]] = record["count"]
                
                # Get relationship counts
                rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                """
                
                result = session.run(rel_query)
                relationships = {}
                for record in result:
                    relationships[record["relationship_type"]] = record["count"]
                
                return {
                    "node_counts": counts,
                    "relationship_counts": relationships,
                    "total_operations": self.total_operations,
                    "successful_operations": self.successful_operations,
                    "success_rate": (self.successful_operations / self.total_operations) if self.total_operations > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {
                "error": str(e),
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations
            }
    
    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
