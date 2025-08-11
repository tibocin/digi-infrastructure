"""
Unit tests for Rule Engine components.

Tests the RuleEngine, ConditionEvaluator, ActionExecutor, and RuleCompiler
to ensure proper rule evaluation, condition matching, and action execution.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path

# Add src to path for direct imports without triggering main app initialization
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pcs.services.rule_engine import (
    RuleEngine,
    ConditionEvaluator,
    ActionExecutor,
    RuleCompiler,
    Rule,
    Condition,
    LogicalCondition,
    Action,
    RuleEvaluationResult,
    ComparisonOperator,
    LogicalOperator,
    ActionType,
    RuleError
)


class TestConditionEvaluator:
    """Test ConditionEvaluator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ConditionEvaluator()
    
    def test_evaluate_condition_equals(self):
        """Test equals operator."""
        condition = Condition("name", ComparisonOperator.EQUALS, "Alice")
        context = {"name": "Alice", "score": 95}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
        
        # Test case insensitive
        condition_ci = Condition("name", ComparisonOperator.EQUALS, "alice", case_sensitive=False)
        result_ci = self.evaluator.evaluate_condition(condition_ci, context)
        assert result_ci is True
    
    def test_evaluate_condition_not_equals(self):
        """Test not equals operator."""
        condition = Condition("name", ComparisonOperator.NOT_EQUALS, "Bob")
        context = {"name": "Alice"}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
    
    def test_evaluate_condition_greater_than(self):
        """Test greater than operator."""
        condition = Condition("score", ComparisonOperator.GREATER_THAN, 90)
        context = {"score": 95}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
        
        # Test false case
        context_low = {"score": 85}
        result_low = self.evaluator.evaluate_condition(condition, context_low)
        assert result_low is False
    
    def test_evaluate_condition_contains(self):
        """Test contains operator."""
        condition = Condition("message", ComparisonOperator.CONTAINS, "hello")
        context = {"message": "Hello World!"}
        
        # Test case insensitive
        condition_ci = Condition("message", ComparisonOperator.CONTAINS, "hello", case_sensitive=False)
        result = self.evaluator.evaluate_condition(condition_ci, context)
        assert result is True
        
        # Test case sensitive
        result_cs = self.evaluator.evaluate_condition(condition, context)
        assert result_cs is False
    
    def test_evaluate_condition_starts_with(self):
        """Test starts with operator."""
        condition = Condition("email", ComparisonOperator.STARTS_WITH, "admin")
        context = {"email": "admin@example.com"}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
    
    def test_evaluate_condition_regex_match(self):
        """Test regex match operator."""
        condition = Condition("email", ComparisonOperator.REGEX_MATCH, r".*@example\.com$")
        context = {"email": "user@example.com"}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
        
        # Test non-matching
        context_invalid = {"email": "user@other.com"}
        result_invalid = self.evaluator.evaluate_condition(condition, context_invalid)
        assert result_invalid is False
    
    def test_evaluate_condition_in_operator(self):
        """Test in operator."""
        condition = Condition("role", ComparisonOperator.IN, ["admin", "moderator"])
        context = {"role": "admin"}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
        
        # Test case insensitive
        condition_ci = Condition("role", ComparisonOperator.IN, ["ADMIN", "MODERATOR"], case_sensitive=False)
        result_ci = self.evaluator.evaluate_condition(condition_ci, context)
        assert result_ci is True
    
    def test_evaluate_condition_nested_field(self):
        """Test condition evaluation with nested field access."""
        condition = Condition("user.profile.name", ComparisonOperator.EQUALS, "Alice")
        context = {
            "user": {
                "profile": {
                    "name": "Alice",
                    "age": 30
                }
            }
        }
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is True
    
    def test_evaluate_condition_missing_field(self):
        """Test condition evaluation with missing field."""
        condition = Condition("missing_field", ComparisonOperator.EQUALS, "value")
        context = {"name": "Alice"}
        
        result = self.evaluator.evaluate_condition(condition, context)
        assert result is False  # Missing field should evaluate to False
    
    def test_evaluate_logical_condition_and(self):
        """Test AND logical condition."""
        conditions = [
            Condition("name", ComparisonOperator.EQUALS, "Alice"),
            Condition("score", ComparisonOperator.GREATER_THAN, 90)
        ]
        logical_condition = LogicalCondition(LogicalOperator.AND, conditions)
        context = {"name": "Alice", "score": 95}
        
        result = self.evaluator.evaluate_logical_condition(logical_condition, context)
        assert result is True
        
        # Test with one false condition
        context_partial = {"name": "Alice", "score": 85}
        result_partial = self.evaluator.evaluate_logical_condition(logical_condition, context_partial)
        assert result_partial is False
    
    def test_evaluate_logical_condition_or(self):
        """Test OR logical condition."""
        conditions = [
            Condition("name", ComparisonOperator.EQUALS, "Bob"),
            Condition("score", ComparisonOperator.GREATER_THAN, 90)
        ]
        logical_condition = LogicalCondition(LogicalOperator.OR, conditions)
        context = {"name": "Alice", "score": 95}
        
        result = self.evaluator.evaluate_logical_condition(logical_condition, context)
        assert result is True  # One condition is true (score > 90)
    
    def test_evaluate_logical_condition_not(self):
        """Test NOT logical condition."""
        condition = Condition("active", ComparisonOperator.EQUALS, True)
        logical_condition = LogicalCondition(LogicalOperator.NOT, [condition])
        context = {"active": False}
        
        result = self.evaluator.evaluate_logical_condition(logical_condition, context)
        assert result is True  # NOT True = False, NOT False = True
    
    def test_evaluate_logical_condition_nested(self):
        """Test nested logical conditions."""
        # ((name == "Alice" AND score > 90) OR (role == "admin"))
        inner_and = LogicalCondition(LogicalOperator.AND, [
            Condition("name", ComparisonOperator.EQUALS, "Alice"),
            Condition("score", ComparisonOperator.GREATER_THAN, 90)
        ])
        
        outer_or = LogicalCondition(LogicalOperator.OR, [
            inner_and,
            Condition("role", ComparisonOperator.EQUALS, "admin")
        ])
        
        # Test with admin role (should be true even with low score)
        context_admin = {"name": "Bob", "score": 50, "role": "admin"}
        result_admin = self.evaluator.evaluate_logical_condition(outer_or, context_admin)
        assert result_admin is True
        
        # Test with Alice and high score
        context_alice = {"name": "Alice", "score": 95, "role": "user"}
        result_alice = self.evaluator.evaluate_logical_condition(outer_or, context_alice)
        assert result_alice is True


class TestActionExecutor:
    """Test ActionExecutor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ActionExecutor()
    
    @pytest.mark.asyncio
    async def test_execute_set_variable_action(self):
        """Test set variable action."""
        action = Action(ActionType.SET_VARIABLE, {"name": "new_var", "value": "test_value"})
        context = {"existing": "data"}
        
        result = await self.executor.execute_action(action, context)
        
        assert result["existing"] == "data"  # Original data preserved
        assert result["new_var"] == "test_value"  # New variable added
    
    @pytest.mark.asyncio
    async def test_execute_modify_template_action(self):
        """Test modify template action."""
        action = Action(ActionType.MODIFY_TEMPLATE, {
            "updates": {"style": "formal", "tone": "professional"}
        })
        context = {"template": {"existing": "value"}}
        
        result = await self.executor.execute_action(action, context)
        
        assert result["template"]["existing"] == "value"  # Existing preserved
        assert result["template"]["style"] == "formal"  # New values added
        assert result["template"]["tone"] == "professional"
    
    @pytest.mark.asyncio
    async def test_execute_add_context_action(self):
        """Test add context action."""
        action = Action(ActionType.ADD_CONTEXT, {
            "data": {"user_id": 123, "session_id": "abc123"}
        })
        context = {"existing": "data"}
        
        result = await self.executor.execute_action(action, context)
        
        assert result["existing"] == "data"
        assert result["user_id"] == 123
        assert result["session_id"] == "abc123"
    
    @pytest.mark.asyncio
    async def test_execute_remove_context_action(self):
        """Test remove context action."""
        action = Action(ActionType.REMOVE_CONTEXT, {
            "keys": ["temp_data", "cache_key"]
        })
        context = {
            "keep_this": "value",
            "temp_data": "remove_me",
            "cache_key": "also_remove",
            "another_keep": "value2"
        }
        
        result = await self.executor.execute_action(action, context)
        
        assert result["keep_this"] == "value"
        assert result["another_keep"] == "value2"
        assert "temp_data" not in result
        assert "cache_key" not in result
    
    @pytest.mark.asyncio
    async def test_execute_log_message_action(self):
        """Test log message action."""
        action = Action(ActionType.LOG_MESSAGE, {
            "message": "Test log message",
            "level": "info"
        })
        context = {"data": "value"}
        
        # Capture print output
        with patch("builtins.print") as mock_print:
            result = await self.executor.execute_action(action, context)
        
        # Check log was called
        mock_print.assert_called_once_with("[RULE LOG - INFO] Test log message")
        
        # Context should be unchanged
        assert result == context
    
    @pytest.mark.asyncio
    async def test_execute_invalid_action_type(self):
        """Test execution with invalid action type."""
        # Create action with invalid type (bypassing enum validation)
        action = Action(ActionType.SET_VARIABLE, {})
        action.action_type = "INVALID_TYPE"
        
        with pytest.raises(RuleError) as exc_info:
            await self.executor.execute_action(action, {})
        
        assert "unsupported action type" in str(exc_info.value).lower()


class TestRuleCompiler:
    """Test RuleCompiler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = RuleCompiler()
    
    def test_compile_simple_rule(self):
        """Test compiling a simple rule."""
        rule_def = {
            "name": "test_rule",
            "description": "A test rule",
            "condition": {
                "field": "score",
                "operator": "gt",
                "value": 90
            },
            "actions": [
                {
                    "type": "set_variable",
                    "parameters": {"name": "grade", "value": "A"}
                }
            ]
        }
        
        rule = self.compiler.compile_rule(rule_def)
        
        assert rule.name == "test_rule"
        assert rule.description == "A test rule"
        assert isinstance(rule.condition, Condition)
        assert rule.condition.field == "score"
        assert rule.condition.operator == ComparisonOperator.GREATER_THAN
        assert rule.condition.value == 90
        assert len(rule.actions) == 1
        assert rule.actions[0].action_type == ActionType.SET_VARIABLE
    
    def test_compile_logical_rule(self):
        """Test compiling a rule with logical conditions."""
        rule_def = {
            "name": "complex_rule",
            "description": "A complex rule with AND logic",
            "condition": {
                "operator": "and",
                "conditions": [
                    {
                        "field": "name",
                        "operator": "eq",
                        "value": "Alice"
                    },
                    {
                        "field": "score",
                        "operator": "gte",
                        "value": 95
                    }
                ]
            },
            "actions": [
                {
                    "type": "add_context",
                    "parameters": {"data": {"status": "excellent"}}
                }
            ]
        }
        
        rule = self.compiler.compile_rule(rule_def)
        
        assert isinstance(rule.condition, LogicalCondition)
        assert rule.condition.operator == LogicalOperator.AND
        assert len(rule.condition.conditions) == 2
    
    def test_compile_nested_logical_rule(self):
        """Test compiling a rule with nested logical conditions."""
        rule_def = {
            "name": "nested_rule",
            "description": "A rule with nested logic",
            "condition": {
                "operator": "or",
                "conditions": [
                    {
                        "operator": "and",
                        "conditions": [
                            {"field": "role", "operator": "eq", "value": "admin"},
                            {"field": "active", "operator": "eq", "value": True}
                        ]
                    },
                    {
                        "field": "override", "operator": "eq", "value": True
                    }
                ]
            },
            "actions": []
        }
        
        rule = self.compiler.compile_rule(rule_def)
        
        assert isinstance(rule.condition, LogicalCondition)
        assert rule.condition.operator == LogicalOperator.OR
        assert len(rule.condition.conditions) == 2
        assert isinstance(rule.condition.conditions[0], LogicalCondition)
    
    def test_compile_rule_missing_fields(self):
        """Test compiling rule with missing required fields."""
        incomplete_rules = [
            {},  # No name
            {"name": "test"},  # No condition
            {"name": "test", "condition": {"field": "score"}},  # No operator
        ]
        
        for rule_def in incomplete_rules:
            with pytest.raises(RuleError):
                self.compiler.compile_rule(rule_def)


class TestRuleEngine:
    """Test RuleEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RuleEngine()
    
    def test_add_rule(self):
        """Test adding a rule to the engine."""
        rule_def = {
            "name": "test_rule",
            "description": "A test rule",
            "condition": {
                "field": "score",
                "operator": "gt",
                "value": 90
            },
            "actions": []
        }
        
        rule_name = self.engine.add_rule(rule_def)
        
        assert rule_name == "test_rule"
        assert "test_rule" in self.engine.rules
        assert self.engine.get_rule("test_rule") is not None
    
    def test_remove_rule(self):
        """Test removing a rule from the engine."""
        rule_def = {
            "name": "temp_rule",
            "description": "A temporary rule",
            "condition": {"field": "test", "operator": "eq", "value": "value"},
            "actions": []
        }
        
        self.engine.add_rule(rule_def)
        assert self.engine.get_rule("temp_rule") is not None
        
        removed = self.engine.remove_rule("temp_rule")
        assert removed is True
        assert self.engine.get_rule("temp_rule") is None
        
        # Try removing non-existent rule
        removed_again = self.engine.remove_rule("temp_rule")
        assert removed_again is False
    
    def test_list_rules(self):
        """Test listing all rules."""
        initial_count = len(self.engine.list_rules())
        
        rule_defs = [
            {
                "name": "rule1",
                "condition": {"field": "test", "operator": "eq", "value": "1"},
                "actions": []
            },
            {
                "name": "rule2",
                "condition": {"field": "test", "operator": "eq", "value": "2"},
                "actions": []
            }
        ]
        
        for rule_def in rule_defs:
            self.engine.add_rule(rule_def)
        
        rules = self.engine.list_rules()
        assert len(rules) == initial_count + 2
        assert "rule1" in rules
        assert "rule2" in rules
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_matching(self):
        """Test evaluating rules with matching conditions."""
        rule_def = {
            "name": "score_rule",
            "description": "High score rule",
            "condition": {
                "field": "score",
                "operator": "gt",
                "value": 90
            },
            "actions": [
                {
                    "type": "set_variable",
                    "parameters": {"name": "grade", "value": "A"}
                }
            ]
        }
        
        self.engine.add_rule(rule_def)
        context = {"score": 95}
        
        results = await self.engine.evaluate_rules(context)
        
        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "score_rule"
        assert result.matched is True
        assert "set_variable" in result.executed_actions
        assert result.context_updates["grade"] == "A"
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_non_matching(self):
        """Test evaluating rules with non-matching conditions."""
        rule_def = {
            "name": "score_rule",
            "condition": {
                "field": "score",
                "operator": "gt",
                "value": 90
            },
            "actions": []
        }
        
        self.engine.add_rule(rule_def)
        context = {"score": 85}  # Below threshold
        
        results = await self.engine.evaluate_rules(context)
        
        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "score_rule"
        assert result.matched is False
        assert len(result.executed_actions) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_specific_rules(self):
        """Test evaluating only specific rules."""
        rule_defs = [
            {
                "name": "rule1",
                "condition": {"field": "test", "operator": "eq", "value": "value"},
                "actions": []
            },
            {
                "name": "rule2", 
                "condition": {"field": "test", "operator": "eq", "value": "value"},
                "actions": []
            }
        ]
        
        for rule_def in rule_defs:
            self.engine.add_rule(rule_def)
        
        context = {"test": "value"}
        
        # Evaluate only rule1
        results = await self.engine.evaluate_rules(context, ["rule1"])
        
        assert len(results) == 1
        assert results[0].rule_name == "rule1"
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_with_priority(self):
        """Test evaluating rules respects priority order."""
        rule_defs = [
            {
                "name": "low_priority",
                "priority": 1,
                "condition": {"field": "test", "operator": "eq", "value": "value"},
                "actions": [{"type": "set_variable", "parameters": {"name": "order", "value": "second"}}]
            },
            {
                "name": "high_priority",
                "priority": 10,
                "condition": {"field": "test", "operator": "eq", "value": "value"},
                "actions": [{"type": "set_variable", "parameters": {"name": "order", "value": "first"}}]
            }
        ]
        
        for rule_def in rule_defs:
            self.engine.add_rule(rule_def)
        
        context = {"test": "value"}
        results = await self.engine.evaluate_rules(context)
        
        # High priority rule should be evaluated first
        assert results[0].rule_name == "high_priority"
        assert results[1].rule_name == "low_priority"
    
    @pytest.mark.asyncio
    async def test_evaluate_disabled_rule(self):
        """Test that disabled rules are not evaluated."""
        rule_def = {
            "name": "disabled_rule",
            "enabled": False,
            "condition": {"field": "test", "operator": "eq", "value": "value"},
            "actions": []
        }
        
        self.engine.add_rule(rule_def)
        context = {"test": "value"}
        
        results = await self.engine.evaluate_rules(context)
        
        # Should return no results since rule is disabled
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
