"""
Filepath: pcs/src/pcs/services/rule_engine.py
Purpose: Rule engine for conditional prompt logic with safe expression evaluation and action execution
Related Components: Template engine, context management, conditional logic
Tags: rule-engine, conditional-logic, expression-evaluation, security, actions
"""

import ast
import json
import operator
import re
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.exceptions import PCSError


class RuleError(PCSError):
    """Custom exception for rule-related errors."""
    pass


class ComparisonOperator(Enum):
    """Supported comparison operators."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex"
    IN = "in"
    NOT_IN = "not_in"


class LogicalOperator(Enum):
    """Supported logical operators."""
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """Supported action types."""
    SET_VARIABLE = "set_variable"
    MODIFY_TEMPLATE = "modify_template"
    ADD_CONTEXT = "add_context"
    REMOVE_CONTEXT = "remove_context"
    CALL_FUNCTION = "call_function"
    LOG_MESSAGE = "log_message"


@dataclass
class Condition:
    """
    Represents a single condition in a rule.
    
    A condition evaluates a field against a value using an operator.
    """
    field: str
    operator: ComparisonOperator
    value: Any
    case_sensitive: bool = True
    
    def __post_init__(self):
        if isinstance(self.operator, str):
            self.operator = ComparisonOperator(self.operator)


@dataclass
class LogicalCondition:
    """
    Represents a logical combination of conditions.
    
    Can combine multiple conditions or other logical conditions using AND/OR/NOT.
    """
    operator: LogicalOperator
    conditions: List[Union[Condition, 'LogicalCondition']]
    
    def __post_init__(self):
        if isinstance(self.operator, str):
            self.operator = LogicalOperator(self.operator)


@dataclass
class Action:
    """
    Represents an action to execute when a rule condition is met.
    """
    action_type: ActionType
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.action_type, str):
            self.action_type = ActionType(self.action_type)


@dataclass
class Rule:
    """
    Represents a complete rule with conditions and actions.
    """
    name: str
    description: str
    condition: Union[Condition, LogicalCondition]
    actions: List[Action]
    priority: int = 0
    enabled: bool = True


@dataclass
class RuleEvaluationResult:
    """
    Result of rule evaluation.
    """
    rule_name: str
    matched: bool
    executed_actions: List[str]
    errors: List[str]
    context_updates: Dict[str, Any]


class ConditionEvaluator:
    """
    Evaluates conditions safely against context data.
    
    Provides secure evaluation of conditions without allowing arbitrary code execution.
    """
    
    def __init__(self):
        """Initialize condition evaluator."""
        self.operators = {
            ComparisonOperator.EQUALS: self._equals,
            ComparisonOperator.NOT_EQUALS: self._not_equals,
            ComparisonOperator.GREATER_THAN: self._greater_than,
            ComparisonOperator.GREATER_THAN_OR_EQUAL: self._greater_than_or_equal,
            ComparisonOperator.LESS_THAN: self._less_than,
            ComparisonOperator.LESS_THAN_OR_EQUAL: self._less_than_or_equal,
            ComparisonOperator.CONTAINS: self._contains,
            ComparisonOperator.STARTS_WITH: self._starts_with,
            ComparisonOperator.ENDS_WITH: self._ends_with,
            ComparisonOperator.REGEX_MATCH: self._regex_match,
            ComparisonOperator.IN: self._in,
            ComparisonOperator.NOT_IN: self._not_in,
        }
    
    def evaluate_condition(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """
        Evaluate a single condition against context.
        
        Args:
            condition: Condition to evaluate
            context: Context data to evaluate against
            
        Returns:
            True if condition matches, False otherwise
            
        Raises:
            RuleError: If evaluation fails
        """
        try:
            # Get field value from context
            field_value = self._get_field_value(condition.field, context)
            
            # Get evaluation function
            eval_func = self.operators.get(condition.operator)
            if not eval_func:
                raise RuleError(f"Unsupported operator: {condition.operator}")
            
            # Evaluate condition
            return eval_func(field_value, condition.value, condition.case_sensitive)
            
        except Exception as e:
            raise RuleError(f"Failed to evaluate condition: {str(e)}") from e
    
    def evaluate_logical_condition(
        self, 
        logical_condition: LogicalCondition, 
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a logical condition (AND/OR/NOT).
        
        Args:
            logical_condition: Logical condition to evaluate
            context: Context data to evaluate against
            
        Returns:
            True if logical condition matches, False otherwise
        """
        try:
            results = []
            
            for condition in logical_condition.conditions:
                if isinstance(condition, Condition):
                    result = self.evaluate_condition(condition, context)
                elif isinstance(condition, LogicalCondition):
                    result = self.evaluate_logical_condition(condition, context)
                else:
                    raise RuleError(f"Invalid condition type: {type(condition)}")
                
                results.append(result)
            
            # Apply logical operator
            if logical_condition.operator == LogicalOperator.AND:
                return all(results)
            elif logical_condition.operator == LogicalOperator.OR:
                return any(results)
            elif logical_condition.operator == LogicalOperator.NOT:
                # NOT should have exactly one condition
                if len(results) != 1:
                    raise RuleError("NOT operator requires exactly one condition")
                return not results[0]
            else:
                raise RuleError(f"Unsupported logical operator: {logical_condition.operator}")
                
        except Exception as e:
            raise RuleError(f"Failed to evaluate logical condition: {str(e)}") from e
    
    def _get_field_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """
        Get field value from context using dot notation.
        
        Args:
            field_path: Field path (e.g., "user.name" or "settings.theme")
            context: Context dictionary
            
        Returns:
            Field value or None if not found
        """
        try:
            value = context
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
                
                if value is None:
                    return None
            
            return value
        except Exception:
            return None
    
    # Operator implementations
    def _equals(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        if isinstance(field_value, str) and isinstance(compare_value, str) and not case_sensitive:
            return field_value.lower() == compare_value.lower()
        return field_value == compare_value
    
    def _not_equals(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        return not self._equals(field_value, compare_value, case_sensitive)
    
    def _greater_than(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            return field_value > compare_value
        except TypeError:
            return False
    
    def _greater_than_or_equal(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            return field_value >= compare_value
        except TypeError:
            return False
    
    def _less_than(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            return field_value < compare_value
        except TypeError:
            return False
    
    def _less_than_or_equal(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            return field_value <= compare_value
        except TypeError:
            return False
    
    def _contains(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            if isinstance(field_value, str) and isinstance(compare_value, str):
                if not case_sensitive:
                    return compare_value.lower() in field_value.lower()
                return compare_value in field_value
            elif hasattr(field_value, '__contains__'):
                return compare_value in field_value
            return False
        except Exception:
            return False
    
    def _starts_with(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            if isinstance(field_value, str) and isinstance(compare_value, str):
                if not case_sensitive:
                    return field_value.lower().startswith(compare_value.lower())
                return field_value.startswith(compare_value)
            return False
        except Exception:
            return False
    
    def _ends_with(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            if isinstance(field_value, str) and isinstance(compare_value, str):
                if not case_sensitive:
                    return field_value.lower().endswith(compare_value.lower())
                return field_value.endswith(compare_value)
            return False
        except Exception:
            return False
    
    def _regex_match(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            if not isinstance(field_value, str) or not isinstance(compare_value, str):
                return False
            
            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(compare_value, field_value, flags))
        except Exception:
            return False
    
    def _in(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        try:
            if hasattr(compare_value, '__contains__'):
                if isinstance(field_value, str) and not case_sensitive:
                    # For string comparison in lists, check case-insensitive
                    return any(
                        isinstance(item, str) and item.lower() == field_value.lower()
                        for item in compare_value
                    )
                return field_value in compare_value
            return False
        except Exception:
            return False
    
    def _not_in(self, field_value: Any, compare_value: Any, case_sensitive: bool) -> bool:
        return not self._in(field_value, compare_value, case_sensitive)


class ActionExecutor:
    """
    Executes actions when rule conditions are met.
    
    Provides safe execution of actions with proper error handling.
    """
    
    def __init__(self):
        """Initialize action executor."""
        self.action_handlers = {
            ActionType.SET_VARIABLE: self._set_variable,
            ActionType.MODIFY_TEMPLATE: self._modify_template,
            ActionType.ADD_CONTEXT: self._add_context,
            ActionType.REMOVE_CONTEXT: self._remove_context,
            ActionType.CALL_FUNCTION: self._call_function,
            ActionType.LOG_MESSAGE: self._log_message,
        }
    
    async def execute_action(
        self, 
        action: Action, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
            context: Current context
            
        Returns:
            Updated context after action execution
            
        Raises:
            RuleError: If action execution fails
        """
        try:
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                raise RuleError(f"Unsupported action type: {action.action_type}")
            
            return await handler(action.parameters, context)
            
        except Exception as e:
            raise RuleError(f"Failed to execute action {action.action_type}: {str(e)}") from e
    
    async def _set_variable(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Set a variable in the context."""
        variable_name = parameters.get('name')
        variable_value = parameters.get('value')
        
        if not variable_name:
            raise RuleError("set_variable action requires 'name' parameter")
        
        # Create a copy of context to avoid modifying original
        updated_context = context.copy()
        updated_context[variable_name] = variable_value
        
        return updated_context
    
    async def _modify_template(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify template-related context."""
        template_updates = parameters.get('updates', {})
        
        updated_context = context.copy()
        template_context = updated_context.get('template', {})
        template_context.update(template_updates)
        updated_context['template'] = template_context
        
        return updated_context
    
    async def _add_context(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add context data."""
        context_data = parameters.get('data', {})
        
        updated_context = context.copy()
        updated_context.update(context_data)
        
        return updated_context
    
    async def _remove_context(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove context data."""
        keys_to_remove = parameters.get('keys', [])
        
        updated_context = context.copy()
        for key in keys_to_remove:
            updated_context.pop(key, None)
        
        return updated_context
    
    async def _call_function(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a registered function (placeholder for extensibility)."""
        function_name = parameters.get('function')
        function_args = parameters.get('args', [])
        function_kwargs = parameters.get('kwargs', {})
        
        # This is a placeholder - in a real implementation, you'd have a registry
        # of safe functions that can be called
        # For now, just return the context unchanged
        return context.copy()
    
    async def _log_message(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Log a message (placeholder)."""
        message = parameters.get('message', '')
        level = parameters.get('level', 'info')
        
        # In a real implementation, this would use the logging system
        print(f"[RULE LOG - {level.upper()}] {message}")
        
        return context.copy()


class RuleCompiler:
    """
    Compiles rule definitions into executable rule objects.
    
    Handles parsing and validation of rule definitions.
    """
    
    def __init__(self):
        """Initialize rule compiler."""
        pass
    
    def compile_rule(self, rule_definition: Dict[str, Any]) -> Rule:
        """
        Compile a rule definition into a Rule object.
        
        Args:
            rule_definition: Dictionary containing rule definition
            
        Returns:
            Compiled Rule object
            
        Raises:
            RuleError: If compilation fails
        """
        try:
            name = rule_definition.get('name')
            description = rule_definition.get('description', '')
            condition_def = rule_definition.get('condition')
            actions_def = rule_definition.get('actions', [])
            priority = rule_definition.get('priority', 0)
            enabled = rule_definition.get('enabled', True)
            
            if not name:
                raise RuleError("Rule must have a name")
            
            if not condition_def:
                raise RuleError("Rule must have a condition")
            
            # Compile condition
            condition = self._compile_condition(condition_def)
            
            # Compile actions
            actions = [self._compile_action(action_def) for action_def in actions_def]
            
            return Rule(
                name=name,
                description=description,
                condition=condition,
                actions=actions,
                priority=priority,
                enabled=enabled
            )
            
        except Exception as e:
            raise RuleError(f"Failed to compile rule: {str(e)}") from e
    
    def _compile_condition(self, condition_def: Dict[str, Any]) -> Union[Condition, LogicalCondition]:
        """Compile a condition definition."""
        if 'operator' in condition_def and condition_def['operator'] in ['and', 'or', 'not']:
            # Logical condition
            operator = LogicalOperator(condition_def['operator'])
            conditions = [
                self._compile_condition(cond_def) 
                for cond_def in condition_def.get('conditions', [])
            ]
            return LogicalCondition(operator=operator, conditions=conditions)
        else:
            # Simple condition
            field = condition_def.get('field')
            operator = condition_def.get('operator')
            value = condition_def.get('value')
            case_sensitive = condition_def.get('case_sensitive', True)
            
            if not all([field, operator]):
                raise RuleError("Condition must have field and operator")
            
            return Condition(
                field=field,
                operator=ComparisonOperator(operator),
                value=value,
                case_sensitive=case_sensitive
            )
    
    def _compile_action(self, action_def: Dict[str, Any]) -> Action:
        """Compile an action definition."""
        action_type = action_def.get('type')
        parameters = action_def.get('parameters', {})
        
        if not action_type:
            raise RuleError("Action must have a type")
        
        return Action(
            action_type=ActionType(action_type),
            parameters=parameters
        )


class RuleEngine:
    """
    Main rule engine for evaluating rules and executing actions.
    
    Provides the primary interface for rule-based processing in the PCS system.
    """
    
    def __init__(self):
        """Initialize rule engine."""
        self.condition_evaluator = ConditionEvaluator()
        self.action_executor = ActionExecutor()
        self.rule_compiler = RuleCompiler()
        self.rules: Dict[str, Rule] = {}
    
    def add_rule(self, rule_definition: Dict[str, Any]) -> str:
        """
        Add a rule to the engine.
        
        Args:
            rule_definition: Dictionary containing rule definition
            
        Returns:
            Rule name
            
        Raises:
            RuleError: If rule compilation fails
        """
        rule = self.rule_compiler.compile_rule(rule_definition)
        self.rules[rule.name] = rule
        return rule.name
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule from the engine.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        return self.rules.pop(rule_name, None) is not None
    
    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """
        Get a rule by name.
        
        Args:
            rule_name: Name of rule to get
            
        Returns:
            Rule object or None if not found
        """
        return self.rules.get(rule_name)
    
    def list_rules(self) -> List[str]:
        """
        List all rule names.
        
        Returns:
            List of rule names
        """
        return list(self.rules.keys())
    
    async def evaluate_rules(
        self, 
        context: Dict[str, Any],
        rule_names: Optional[List[str]] = None
    ) -> List[RuleEvaluationResult]:
        """
        Evaluate rules against context and execute matching actions.
        
        Args:
            context: Context data to evaluate rules against
            rule_names: Optional list of specific rule names to evaluate
            
        Returns:
            List of evaluation results for each rule
        """
        results = []
        
        # Get rules to evaluate
        rules_to_evaluate = []
        if rule_names:
            rules_to_evaluate = [
                rule for name, rule in self.rules.items() 
                if name in rule_names and rule.enabled
            ]
        else:
            rules_to_evaluate = [rule for rule in self.rules.values() if rule.enabled]
        
        # Sort by priority (higher priority first)
        rules_to_evaluate.sort(key=lambda r: r.priority, reverse=True)
        
        # Evaluate each rule
        current_context = context.copy()
        
        for rule in rules_to_evaluate:
            result = await self._evaluate_single_rule(rule, current_context)
            results.append(result)
            
            # Update context with any changes from executed actions
            if result.context_updates:
                current_context.update(result.context_updates)
        
        return results
    
    async def _evaluate_single_rule(
        self, 
        rule: Rule, 
        context: Dict[str, Any]
    ) -> RuleEvaluationResult:
        """
        Evaluate a single rule against context.
        
        Args:
            rule: Rule to evaluate
            context: Context data
            
        Returns:
            Evaluation result
        """
        result = RuleEvaluationResult(
            rule_name=rule.name,
            matched=False,
            executed_actions=[],
            errors=[],
            context_updates={}
        )
        
        try:
            # Evaluate condition
            if isinstance(rule.condition, Condition):
                matched = self.condition_evaluator.evaluate_condition(rule.condition, context)
            elif isinstance(rule.condition, LogicalCondition):
                matched = self.condition_evaluator.evaluate_logical_condition(rule.condition, context)
            else:
                raise RuleError(f"Invalid condition type: {type(rule.condition)}")
            
            result.matched = matched
            
            # Execute actions if condition matched
            if matched:
                updated_context = context.copy()
                
                for action in rule.actions:
                    try:
                        updated_context = await self.action_executor.execute_action(action, updated_context)
                        result.executed_actions.append(action.action_type.value)
                    except Exception as e:
                        result.errors.append(f"Action {action.action_type.value} failed: {str(e)}")
                
                # Calculate context updates
                result.context_updates = {
                    key: value for key, value in updated_context.items()
                    if key not in context or context[key] != value
                }
            
        except Exception as e:
            result.errors.append(f"Rule evaluation failed: {str(e)}")
        
        return result
