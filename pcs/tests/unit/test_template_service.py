"""
Unit tests for Template Service components.

Tests the TemplateEngine, TemplateValidator, and VariableInjector
to ensure proper template processing, validation, and security.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add src to path for direct imports without triggering main app initialization
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pcs.services.template_service import (
    TemplateEngine,
    TemplateValidator,
    TemplateValidationResult,
    VariableInjector,
    TemplateError
)


class TestTemplateValidator:
    """Test TemplateValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.template_engine = TemplateEngine()
        self.validator = TemplateValidator(self.template_engine.env)
    
    def test_validate_syntax_valid_template(self):
        """Test syntax validation with valid template."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        result = self.validator.validate_syntax(template)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_syntax_invalid_template(self):
        """Test syntax validation with invalid template."""
        template = "Hello {{ name! Your score is {{ score }}."  # Missing closing brace
        result = self.validator.validate_syntax(template)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "syntax error" in result.errors[0].lower()
    
    def test_validate_variables_all_present(self):
        """Test variable validation when all required variables are present."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        required_vars = {"name", "score"}
        
        result = self.validator.validate_variables(template, required_vars)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_variables_missing_required(self):
        """Test variable validation with missing required variables."""
        template = "Hello {{ name }}!"
        required_vars = {"name", "score", "level"}
        
        result = self.validator.validate_variables(template, required_vars)
        
        assert not result.is_valid
        assert len(result.errors) == 2  # missing score and level
        assert any("score" in error for error in result.errors)
        assert any("level" in error for error in result.errors)
    
    def test_validate_variables_with_available_vars(self):
        """Test variable validation with available variables check."""
        template = "Hello {{ name }}! You have {{ notifications }} and {{ extra }}."
        required_vars = {"name"}
        available_vars = {"name", "notifications"}
        
        result = self.validator.validate_variables(template, required_vars, available_vars)
        
        assert result.is_valid
        assert len(result.warnings) == 1  # extra is not available
        assert "extra" in result.warnings[0]
    
    def test_extract_variables(self):
        """Test variable extraction from template."""
        template = "Hello {{ user.name }}! Your {{ item_type }} count is {{ count }}."
        variables = self.validator.extract_variables(template)
        
        expected_vars = {"user", "item_type", "count"}
        assert variables == expected_vars
    
    def test_validate_security_safe_template(self):
        """Test security validation with safe template."""
        template = "Hello {{ name }}! Your score is {{ score | default(0) }}."
        result = self.validator.validate_security(template)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_security_dangerous_template(self):
        """Test security validation with dangerous patterns."""
        dangerous_templates = [
            "{{ name.__class__ }}",
            "{{ ''.__class__.mro() }}",
            "{% import os %}",
            "{{ exec('print()') }}",
            "{{ eval('1+1') }}"
        ]
        
        for template in dangerous_templates:
            result = self.validator.validate_security(template)
            assert not result.is_valid, f"Template should be rejected: {template}"
            assert len(result.errors) > 0


class TestVariableInjector:
    """Test VariableInjector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.injector = VariableInjector()
    
    def test_prepare_context_basic(self):
        """Test basic context preparation."""
        variables = {
            "name": "John",
            "score": 100,
            "active": True
        }
        
        context = self.injector.prepare_context(variables)
        
        assert context["name"] == "John"
        assert context["score"] == 100
        assert context["active"] is True
        # Check utility functions are added
        assert "now" in context
        assert "json_dumps" in context
    
    def test_prepare_context_with_datetime(self):
        """Test context preparation with datetime objects."""
        now = datetime.utcnow()
        variables = {
            "created_at": now,
            "expires_in": timedelta(hours=1)
        }
        
        context = self.injector.prepare_context(variables)
        
        assert context["created_at"] == now  # Datetime objects are preserved for filters
        assert context["expires_in"] == str(timedelta(hours=1))
    
    def test_prepare_context_with_none_values(self):
        """Test context preparation with None values."""
        variables = {
            "name": "John",
            "optional_field": None,
            "empty_list": []
        }
        
        context = self.injector.prepare_context(variables)
        
        assert context["name"] == "John"
        assert context["optional_field"] == ""  # None converted to empty string
        assert context["empty_list"] == []
    
    def test_prepare_context_with_filters(self):
        """Test context preparation with custom filters."""
        variables = {"score": 95}
        filters = {"score": lambda x: f"{x}%"}
        
        context = self.injector.prepare_context(variables, filters)
        
        assert context["score"] == "95%"
    
    def test_prepare_context_with_complex_objects(self):
        """Test context preparation with complex objects."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
            def __str__(self):
                return f"CustomObject({self.value})"
        
        variables = {
            "obj": CustomObject(42),
            "normal": "string"
        }
        
        context = self.injector.prepare_context(variables)
        
        assert context["obj"] == "CustomObject(42)"
        assert context["normal"] == "string"


class TestTemplateEngine:
    """Test TemplateEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemplateEngine(enable_cache=False)  # Disable cache for testing
    
    @pytest.mark.asyncio
    async def test_render_template_basic(self):
        """Test basic template rendering."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        variables = {"name": "Alice", "score": 95}
        
        result = await self.engine.render_template(template, variables)
        
        assert result == "Hello Alice! Your score is 95."
    
    @pytest.mark.asyncio
    async def test_render_template_with_filters(self):
        """Test template rendering with built-in filters."""
        template = "Created: {{ timestamp | datetime_format('%Y-%m-%d') }}"
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        variables = {"timestamp": timestamp}
        
        result = await self.engine.render_template(template, variables)
        
        assert result == "Created: 2024-01-15"
    
    @pytest.mark.asyncio
    async def test_render_template_with_validation_errors(self):
        """Test template rendering with validation errors."""
        template = "Hello {{ name.__class__ }}!"  # Security violation
        variables = {"name": "Alice"}
        
        with pytest.raises(TemplateError) as exc_info:
            await self.engine.render_template(template, variables, validate=True)
        
        assert "validation failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_render_template_undefined_variable(self):
        """Test template rendering with undefined variables."""
        template = "Hello {{ name }}! Your score is {{ undefined_var }}."
        variables = {"name": "Alice"}
        
        with pytest.raises(TemplateError) as exc_info:
            await self.engine.render_template(template, variables)
        
        assert "undefined variable" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_render_template_without_validation(self):
        """Test template rendering without validation."""
        template = "Hello {{ name }}! Score: {{ score | default('N/A') }}."
        variables = {"name": "Alice"}
        
        result = await self.engine.render_template(template, variables, validate=False)
        
        assert result == "Hello Alice! Score: N/A."
    
    @pytest.mark.asyncio
    async def test_render_template_with_required_vars(self):
        """Test template rendering with required variables check."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        variables = {"name": "Alice", "score": 95}
        required_vars = {"name", "score"}
        
        result = await self.engine.render_template(
            template, variables, required_vars=required_vars
        )
        
        assert result == "Hello Alice! Your score is 95."
    
    @pytest.mark.asyncio
    async def test_render_template_missing_required_vars(self):
        """Test template rendering with missing required variables."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        variables = {"name": "Alice"}
        required_vars = {"name", "score"}
        
        with pytest.raises(TemplateError):
            await self.engine.render_template(
                template, variables, required_vars=required_vars
            )
    
    def test_extract_variables(self):
        """Test variable extraction from template."""
        template = "Hello {{ user.name }}! You have {{ count }} {{ item_type }}(s)."
        variables = self.engine.extract_variables(template)
        
        expected_vars = {"user", "count", "item_type"}
        assert variables == expected_vars
    
    def test_validate_template(self):
        """Test comprehensive template validation."""
        template = "Hello {{ name }}! Your score is {{ score }}."
        required_vars = {"name", "score"}
        available_vars = {"name", "score", "level"}
        
        result = self.engine.validate_template(template, required_vars, available_vars)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_apply_filters(self):
        """Test applying custom filters to template engine."""
        custom_filters = {
            "uppercase": lambda x: x.upper(),
            "add_prefix": lambda x, prefix=">>": f"{prefix} {x}"
        }
        
        # Apply filters
        self.engine.apply_filters("", custom_filters)
        
        # Check filters were added
        assert "uppercase" in self.engine.env.filters
        assert "add_prefix" in self.engine.env.filters
    
    def test_cache_operations(self):
        """Test template cache operations."""
        # Create engine with caching enabled
        engine = TemplateEngine(enable_cache=True)
        
        # Initially empty cache
        assert engine.get_cache_size() == 0
        
        # Create a template (this should cache it)
        template_string = "Hello {{ name }}!"
        template = engine._get_template(template_string)
        assert template is not None
        assert engine.get_cache_size() == 1
        
        # Get same template again (should come from cache)
        template2 = engine._get_template(template_string)
        assert template is template2  # Same object reference
        assert engine.get_cache_size() == 1
        
        # Clear cache
        engine.clear_cache()
        assert engine.get_cache_size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
