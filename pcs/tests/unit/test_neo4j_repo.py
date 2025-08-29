"""
Filepath: tests/unit/test_neo4j_repo.py
Purpose: Unit tests for enhanced Neo4j repository implementation
Related Components: Neo4jRepository, GraphNode, Relationship, ConversationPattern
Tags: testing, neo4j, graph-database, relationships, conversation-analysis
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from pcs.repositories.neo4j_repo import (
    Neo4jRepository,
    GraphNode,
    Relationship,
    ConversationPattern,
    GraphTraversalResult,
    RelationshipType
)
from pcs.repositories.base import RepositoryError


class MockAsyncContextManager:
    """Mock async context manager for testing."""
    
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class MockRelationship:
    """Mock Neo4j relationship that behaves like both dict and object."""
    
    def __init__(self, **kwargs):
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key):
        return key in self._data


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver for testing."""
    driver = Mock()
    session = AsyncMock()
    result = AsyncMock()
    
    # Setup session methods
    session.run.return_value = result
    result.data.return_value = []
    
    # Setup session as proper async context manager
    driver.session.return_value = MockAsyncContextManager(session)
    
    return driver


@pytest.fixture
def repository(mock_driver):
    """Create a Neo4j repository for testing."""
    return Neo4jRepository(mock_driver)


def setup_mock_session(mock_driver, mock_data):
    """Helper function to setup mock session with proper async context manager."""
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data.return_value = mock_data
    mock_session.run.return_value = mock_result
    
    # Setup async context manager using our custom mock
    mock_driver.session.return_value = MockAsyncContextManager(mock_session)
    
    return mock_session, mock_result


class TestNeo4jRepositoryEnhanced:
    """Test suite for enhanced Neo4j repository functionality."""

    def test_initialization(self, mock_driver):
        """Test repository initialization."""
        repo = Neo4jRepository(mock_driver)
        assert repo.driver == mock_driver

    @pytest.mark.asyncio
    async def test_execute_cypher_with_monitoring(self, repository, mock_driver):
        """Test Cypher query execution with performance monitoring."""
        # Setup
        mock_data = [{"n": {"id": "test", "name": "Test Node"}}]
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        query = "MATCH (n) WHERE n.id = $id RETURN n"
        params = {"id": "test"}
        
        with patch('pcs.repositories.neo4j_repo.PerformanceMonitor'):
            result = await repository.execute_cypher(query, params)
        
        # Verify
        assert len(result) == 1
        assert result[0]["n"]["id"] == "test"
        mock_session.run.assert_called_once_with(query, params)

    @pytest.mark.asyncio
    async def test_execute_cypher_without_monitoring(self, repository, mock_driver):
        """Test Cypher query execution without performance monitoring."""
        # Setup
        mock_session, mock_result = setup_mock_session(mock_driver, [{"count": 5}])
        
        # Execute
        result = await repository.execute_cypher("MATCH (n) RETURN count(n)", monitor_performance=False)
        
        # Verify
        assert len(result) == 1
        assert result[0]["count"] == 5

    @pytest.mark.asyncio
    async def test_create_node_enhanced(self, repository, mock_driver):
        """Test enhanced node creation with auto-generated ID and timestamp."""
        # Setup
        node_id = str(uuid4())
        created_at = datetime.utcnow().isoformat()
        
        mock_data = [{
            "n": {
                "id": node_id,
                "name": "Test Node",
                "created_at": created_at
            }
        }]
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        properties = {"name": "Test Node"}
        result = await repository.create_node("TestLabel", properties)
        
        # Verify
        assert isinstance(result, GraphNode)
        assert result.label == "TestLabel"
        assert result.properties["name"] == "Test Node"
        assert "id" in result.properties
        assert "created_at" in result.properties

    @pytest.mark.asyncio
    async def test_get_node_by_id_enhanced(self, repository, mock_driver):
        """Test enhanced node retrieval with label information."""
        # Setup
        node_id = uuid4()
        mock_data = [{
            "n": {
                "id": str(node_id),
                "name": "Test Node"
            },
            "labels": ["TestLabel"]
        }]
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        result = await repository.get_node_by_id(node_id)
        
        # Verify
        assert isinstance(result, GraphNode)
        assert result.id == node_id
        assert result.label == "TestLabel"
        assert result.properties["name"] == "Test Node"

    @pytest.mark.asyncio
    async def test_get_node_by_id_not_found(self, repository, mock_driver):
        """Test node retrieval when node doesn't exist."""
        # Setup
        mock_data = []
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        result = await repository.get_node_by_id(uuid4())
        
        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_create_relationship_enhanced(self, repository, mock_driver):
        """Test enhanced relationship creation with timestamp."""
        # Setup
        from_id = uuid4()
        to_id = uuid4()
        rel_id = 12345
        created_at = datetime.utcnow().isoformat()
        
        mock_data = [{
            "r": {
                "type": "DEPENDS_ON",
                "created_at": created_at,
                "strength": 0.8
            },
            "rel_id": rel_id
        }]
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        result = await repository.create_relationship(
            from_node_id=from_id,
            to_node_id=to_id,
            relationship_type="DEPENDS_ON",
            properties={"strength": 0.8}
        )
        
        # Verify
        assert isinstance(result, Relationship)
        assert result.from_node_id == from_id
        assert result.to_node_id == to_id
        assert result.type == "DEPENDS_ON"
        assert result.properties["strength"] == 0.8
        assert "created_at" in result.properties

    @pytest.mark.asyncio
    async def test_find_related_nodes(self, repository, mock_driver):
        """Test finding related nodes through relationships."""
        # Setup
        start_id = uuid4()
        related_id1 = uuid4()
        related_id2 = uuid4()
        
        mock_data = [
            {
                "related": {"id": str(related_id1), "name": "Related 1"},
                "labels": ["Context"]
            },
            {
                "related": {"id": str(related_id2), "name": "Related 2"},
                "labels": ["Context"]
            }
        ]
        mock_session, mock_result = setup_mock_session(mock_driver, mock_data)
        
        # Execute
        result = await repository.find_related_nodes(
            node_id=start_id,
            relationship_type="DEPENDS_ON",
            max_depth=2,
            direction="outgoing"
        )
        
        # Verify
        assert len(result) == 2
        assert all(isinstance(node, GraphNode) for node in result)
        assert result[0].id == related_id1
        assert result[1].id == related_id2

    @pytest.mark.asyncio
    async def test_analyze_conversation_patterns(self, repository, mock_driver):
        """Test conversation pattern analysis."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        user_id = uuid4()
        conv_id1 = uuid4()
        conv_id2 = uuid4()
        
        # Mock conversation data
        mock_result.data.return_value = [
            {
                "conv": {
                    "id": str(conv_id1),
                    "created_at": "2023-01-01T10:00:00",
                    "topic": "programming",
                    "duration": 300
                },
                "messages": [
                    {"id": "msg1", "created_at": "2023-01-01T10:00:00", "topic": "python"},
                    {"id": "msg2", "created_at": "2023-01-01T10:01:00", "topic": "python"}
                ],
                "msg_count": 2
            },
            {
                "conv": {
                    "id": str(conv_id2),
                    "created_at": "2023-01-01T15:00:00",
                    "topic": "ai",
                    "duration": 450
                },
                "messages": [
                    {"id": "msg3", "created_at": "2023-01-01T15:00:00", "topic": "machine-learning"},
                    {"id": "msg4", "created_at": "2023-01-01T15:02:00", "topic": "neural-networks"}
                ],
                "msg_count": 2
            }
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        time_range = (
            datetime(2023, 1, 1),
            datetime(2023, 1, 31)
        )
        result = await repository.analyze_conversation_patterns(user_id, time_range)
        
        # Verify
        assert isinstance(result, ConversationPattern)
        assert result.user_id == user_id
        assert result.frequency == 2
        assert result.avg_response_time == 90.0  # Average of 60 and 120 seconds
        assert "programming" in result.common_topics
        assert "ai" in result.common_topics
        assert len(result.interaction_flow) == 2

    @pytest.mark.asyncio
    async def test_create_context_hierarchy(self, repository, mock_driver):
        """Test context hierarchy creation."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        parent_id = uuid4()
        child_id = uuid4()
        
        mock_result.data.return_value = [{
            "r": {
                "type": "CHILD_OF",
                "created_at": datetime.utcnow().isoformat(),
                "hierarchy_level": 1
            },
            "rel_id": 123
        }]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.create_context_hierarchy(parent_id, child_id)
        
        # Verify
        assert isinstance(result, Relationship)
        assert result.from_node_id == child_id
        assert result.to_node_id == parent_id
        assert result.type == "CHILD_OF"
        assert result.properties["hierarchy_level"] == 1

    @pytest.mark.asyncio
    async def test_find_context_dependencies(self, repository, mock_driver):
        """Test finding context dependencies."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        context_id = uuid4()
        dep_id1 = uuid4()
        dep_id2 = uuid4()
        
        # Mock path data
        mock_result.data.return_value = [{
            "path": "mock_path",
            "path_nodes": [
                {"id": str(context_id), "name": "Context 1"},
                {"id": str(dep_id1), "name": "Dependency 1"},
                {"id": str(dep_id2), "name": "Dependency 2"}
            ],
            "path_rels": [
                MockRelationship(
                    id=1,
                    type="DEPENDS_ON",
                    start_node={"id": str(context_id)},
                    end_node={"id": str(dep_id1)},
                    created_at=datetime.now(UTC).isoformat()
                ),
                MockRelationship(
                    id=2,
                    type="DEPENDS_ON",
                    start_node={"id": str(dep_id1)},
                    end_node={"id": str(dep_id2)},
                    created_at=datetime.now(UTC).isoformat()
                )
            ]
        }]
        
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.find_context_dependencies(context_id, max_depth=3)
        
        # Verify
        assert isinstance(result, GraphTraversalResult)
        assert len(result.nodes) == 3
        assert len(result.relationships) == 2
        assert len(result.paths) == 1
        assert result.metadata["max_depth"] == 3

    @pytest.mark.asyncio
    async def test_create_prompt_template_dependency(self, repository, mock_driver):
        """Test prompt template dependency creation."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        template_id = uuid4()
        dependent_id = uuid4()
        
        mock_result.data.return_value = [{
            "r": {
                "type": "DEPENDS_ON",
                "created_at": datetime.utcnow().isoformat(),
                "dependency_strength": 1.0
            },
            "rel_id": 456
        }]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.create_prompt_template_dependency(template_id, dependent_id)
        
        # Verify
        assert isinstance(result, Relationship)
        assert result.from_node_id == dependent_id
        assert result.to_node_id == template_id
        assert result.type == "DEPENDS_ON"
        assert result.properties["dependency_strength"] == 1.0

    @pytest.mark.asyncio
    async def test_get_relationship_statistics(self, repository, mock_driver):
        """Test relationship statistics retrieval."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        # Mock relationship counts
        mock_result.data.side_effect = [
            [
                {"relationship_type": "DEPENDS_ON", "count": 15},
                {"relationship_type": "CONTAINS", "count": 8},
                {"relationship_type": "RELATES_TO", "count": 3}
            ],
            [
                {"label": ["Context"], "count": 25},
                {"label": ["User"], "count": 10},
                {"label": ["Conversation"], "count": 30}
            ]
        ]
        
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.get_relationship_statistics()
        
        # Verify
        assert "relationship_counts" in result
        assert "node_counts" in result
        assert "total_relationships" in result
        assert "total_nodes" in result
        assert result["relationship_counts"]["DEPENDS_ON"] == 15
        assert result["node_counts"]["Context"] == 25
        assert result["total_relationships"] == 26  # 15 + 8 + 3
        assert result["total_nodes"] == 65  # 25 + 10 + 30

    @pytest.mark.asyncio
    async def test_optimize_graph_performance(self, repository, mock_driver):
        """Test graph performance optimization."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.optimize_graph_performance()
        
        # Verify
        assert isinstance(result, dict)
        # Check that index creation queries were attempted
        index_keys = [key for key in result.keys() if "INDEX" in key]
        constraint_keys = [key for key in result.keys() if "CONSTRAINT" in key]
        assert len(index_keys) > 0
        assert len(constraint_keys) > 0

    def test_analyze_peak_hours(self, repository):
        """Test peak hour analysis helper method."""
        # Setup
        conversation_data = [
            {"conv": {"created_at": "2023-01-01T09:00:00"}},  # 9 AM
            {"conv": {"created_at": "2023-01-01T09:30:00"}},  # 9 AM
            {"conv": {"created_at": "2023-01-01T14:00:00"}},  # 2 PM
            {"conv": {"created_at": "2023-01-01T14:15:00"}},  # 2 PM
            {"conv": {"created_at": "2023-01-01T14:45:00"}},  # 2 PM
            {"conv": {"created_at": "2023-01-01T20:00:00"}},  # 8 PM
        ]
        
        # Execute
        peak_hours = repository._analyze_peak_hours(conversation_data)
        
        # Verify - only hours above average (2.0) are considered peaks
        # Average = (2 + 3 + 1) / 3 = 2.0, so only hour 14 (3 conversations) is above average
        assert 14 in peak_hours  # 2 PM should be peak (3 conversations > 2.0 avg)
        assert 9 not in peak_hours   # 9 AM not peak (2 conversations = 2.0 avg)
        assert 20 not in peak_hours  # 8 PM should not be peak (1 conversation < 2.0 avg)

    def test_analyze_peak_hours_empty_data(self, repository):
        """Test peak hour analysis with empty data."""
        result = repository._analyze_peak_hours([])
        assert result == []


class TestGraphNode:
    """Test GraphNode data class."""

    def test_graph_node_creation(self):
        """Test GraphNode creation and to_dict method."""
        node_id = uuid4()
        properties = {"name": "Test Node", "type": "context"}
        
        node = GraphNode(
            id=node_id,
            label="Context",
            properties=properties
        )
        
        assert node.id == node_id
        assert node.label == "Context"
        assert node.properties == properties
        
        # Test to_dict
        node_dict = node.to_dict()
        assert node_dict["id"] == str(node_id)
        assert node_dict["label"] == "Context"
        assert node_dict["properties"] == properties


class TestRelationship:
    """Test Relationship data class."""

    def test_relationship_creation(self):
        """Test Relationship creation and to_dict method."""
        from_id = uuid4()
        to_id = uuid4()
        created_at = datetime.utcnow()
        properties = {"strength": 0.8}
        
        relationship = Relationship(
            id="123",
            from_node_id=from_id,
            to_node_id=to_id,
            type="DEPENDS_ON",
            properties=properties,
            created_at=created_at
        )
        
        assert relationship.id == "123"
        assert relationship.from_node_id == from_id
        assert relationship.to_node_id == to_id
        assert relationship.type == "DEPENDS_ON"
        assert relationship.properties == properties
        assert relationship.created_at == created_at
        
        # Test to_dict
        rel_dict = relationship.to_dict()
        assert rel_dict["id"] == "123"
        assert rel_dict["from_node_id"] == str(from_id)
        assert rel_dict["to_node_id"] == str(to_id)
        assert rel_dict["type"] == "DEPENDS_ON"
        assert rel_dict["created_at"] == created_at.isoformat()


class TestConversationPattern:
    """Test ConversationPattern data class."""

    def test_conversation_pattern_creation(self):
        """Test ConversationPattern creation and to_dict method."""
        user_id = uuid4()
        interaction_flow = [
            {"conversation_id": str(uuid4()), "message_count": 5},
            {"conversation_id": str(uuid4()), "message_count": 3}
        ]
        temporal_patterns = {"peak_hours": [9, 14], "frequency": 2.5}
        
        pattern = ConversationPattern(
            user_id=user_id,
            pattern_type="general",
            frequency=10,
            avg_response_time=45.0,
            common_topics=["programming", "ai"],
            interaction_flow=interaction_flow,
            temporal_patterns=temporal_patterns
        )
        
        assert pattern.user_id == user_id
        assert pattern.pattern_type == "general"
        assert pattern.frequency == 10
        assert pattern.avg_response_time == 45.0
        assert pattern.common_topics == ["programming", "ai"]
        assert pattern.interaction_flow == interaction_flow
        assert pattern.temporal_patterns == temporal_patterns
        
        # Test to_dict
        pattern_dict = pattern.to_dict()
        assert pattern_dict["user_id"] == str(user_id)
        assert pattern_dict["pattern_type"] == "general"
        assert pattern_dict["frequency"] == 10


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_relationship_types(self):
        """Test that all expected relationship types are defined."""
        assert RelationshipType.DEPENDS_ON.value == "DEPENDS_ON"
        assert RelationshipType.CONTAINS.value == "CONTAINS"
        assert RelationshipType.RELATED_TO.value == "RELATED_TO"
        assert RelationshipType.CHILD_OF.value == "CHILD_OF"
        assert RelationshipType.SIMILAR_TO.value == "SIMILAR_TO"
        
        # Test enum usage
        assert len(RelationshipType) >= 10  # Should have at least 10 relationship types


class TestBackwardCompatibility:
    """Test backward compatibility with legacy methods."""

    @pytest.mark.asyncio
    async def test_legacy_find_relationships(self, repository, mock_driver):
        """Test legacy find_relationships method still works."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"r": {"type": "DEPENDS_ON"}, "other": {"id": "other1"}}
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.find_relationships(
            node_id=uuid4(),
            relationship_type="DEPENDS_ON",
            direction="outgoing"
        )
        
        # Verify
        assert len(result) == 1
        assert result[0]["r"]["type"] == "DEPENDS_ON"

    @pytest.mark.asyncio
    async def test_legacy_delete_node(self, repository, mock_driver):
        """Test legacy delete_node method still works."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [{"deleted_count": 1}]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        result = await repository.delete_node(uuid4())
        
        # Verify
        assert result is True


class TestErrorHandling:
    """Test error handling in Neo4j repository."""

    @pytest.mark.asyncio
    async def test_cypher_execution_error(self, repository, mock_driver):
        """Test error handling in Cypher execution."""
        # Setup
        from neo4j.exceptions import Neo4jError
        mock_session = AsyncMock()
        mock_session.run.side_effect = Neo4jError("Connection failed")
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute and verify
        with pytest.raises(RepositoryError, match="Failed to execute Cypher query"):
            await repository.execute_cypher("INVALID QUERY")

    @pytest.mark.asyncio
    async def test_node_creation_error(self, repository, mock_driver):
        """Test error handling in node creation."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = []  # Empty result should cause error
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute and verify
        with pytest.raises(RepositoryError, match="Failed to create node - no result returned"):
            await repository.create_node("TestLabel", {"name": "test"})

    @pytest.mark.asyncio
    async def test_relationship_creation_error(self, repository, mock_driver):
        """Test error handling in relationship creation."""
        # Setup
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = []  # Empty result should cause error
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute and verify
        with pytest.raises(RepositoryError, match="Failed to create relationship - no result returned"):
            await repository.create_relationship(uuid4(), uuid4(), "TEST_REL")


class TestPerformanceFeatures:
    """Test performance-related features."""

    @pytest.mark.asyncio
    async def test_conversation_pattern_performance(self, repository, mock_driver):
        """Test that conversation pattern analysis handles large datasets efficiently."""
        # Setup large dataset
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        # Simulate 100 conversations
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                "conv": {
                    "id": str(uuid4()),
                    "created_at": f"2023-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00",
                    "topic": f"topic_{i % 10}",
                    "duration": 300 + (i % 200)
                },
                "messages": [
                    {"id": f"msg_{i}_1", "created_at": f"2023-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00"},
                    {"id": f"msg_{i}_2", "created_at": f"2023-01-{(i % 30) + 1:02d}T{(i % 24):02d}:01:00"}
                ],
                "msg_count": 2
            })
        
        mock_result.data.return_value = large_dataset
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = MockAsyncContextManager(mock_session)
        
        # Execute
        user_id = uuid4()
        time_range = (datetime(2023, 1, 1), datetime(2023, 1, 31))
        
        result = await repository.analyze_conversation_patterns(user_id, time_range)
        
        # Verify
        assert isinstance(result, ConversationPattern)
        assert result.frequency == 100
        assert len(result.common_topics) == 10  # topic_0 through topic_9
        assert result.avg_response_time == 60.0  # 1 minute between messages
