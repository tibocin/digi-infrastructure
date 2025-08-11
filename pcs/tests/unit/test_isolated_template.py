"""
Isolated unit tests for Template Engine functionality.

This test module implements and tests template engine components
without importing from the main PCS package to avoid circular dependencies.
"""

import pytest
import re
import json
from datetime import datetime, timedelta
from jinja2 import (
    Environment, DictLoader, Template, TemplateSyntaxError, UndefinedError, 
    select_autoescape, meta, Undefined, StrictUndefined
)
from jinja2.sandbox import SandboxedEnvironment


# Custom exception for testing
class TemplateError(Exception):
    """Custom exception for template-related errors."""
    pass


class TemplateValidationResult:
    """Result of template validation."""
    
    def __init__(self, is_valid: bool, errors: list = None, warnings: list = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)


class TemplateValidator:
    """Template validator for syntax and security validation."""
    
    def __init__(self, environment: Environment):
        """Initialize validator with Jinja2 environment."""
        self.env = environment
        
    def validate_syntax(self, template_string: str) -> TemplateValidationResult:
        """Validate template syntax."""
        result = TemplateValidationResult(True)
        
        try:
            self.env.parse(template_string)
        except TemplateSyntaxError as e:
            result.add_error(f"Template syntax error: {e.message} at line {e.lineno}")
        except Exception as e:
            result.add_error(f"Unexpected template error: {str(e)}")
        
        return result
    
    def validate_variables(self, template_string: str, required_vars: set,
                          available_vars: set = None) -> TemplateValidationResult:
        """Validate template variables."""
        result = TemplateValidationResult(True)
        
        try:
            template_vars = self.extract_variables(template_string)
            
            missing_required = required_vars - template_vars
            if missing_required:
                for var in missing_required:
                    result.add_error(f"Required variable '{var}' not found in template")
            
            if available_vars is not None:
                undefined_vars = template_vars - available_vars
                if undefined_vars:
                    for var in undefined_vars:
                        result.add_warning(f"Variable '{var}' may not be available at runtime")
        
        except Exception as e:
            result.add_error(f"Error validating variables: {str(e)}")
        
        return result
    
    def extract_variables(self, template_string: str) -> set:
        """Extract all variables used in a template."""
        try:
            ast = self.env.parse(template_string)
            return meta.find_undeclared_variables(ast)
        except Exception as e:
            raise TemplateError(f"Failed to extract variables: {str(e)}") from e
    
    def validate_security(self, template_string: str) -> TemplateValidationResult:
        """Validate template for security issues."""
        result = TemplateValidationResult(True)
        
        dangerous_patterns = [
            (r'__\w+__', 'Double underscore attributes are not allowed'),
            (r'\.mro\(\)', 'Method resolution order access is not allowed'),
            (r'\.class\b', 'Class attribute access is not allowed'),
            (r'import\s+\w+', 'Import statements are not allowed'),
            (r'exec\s*\(', 'Exec function calls are not allowed'),
            (r'eval\s*\(', 'Eval function calls are not allowed'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, template_string, re.IGNORECASE):
                result.add_error(f"Security violation: {message}")
        
        return result


class VariableInjector:
    """Variable injector for template context preparation."""
    
    def __init__(self):
        """Initialize variable injector."""
        pass
    
    def prepare_context(self, variables: dict, filters: dict = None) -> dict:
        """Prepare template context with variables and filters."""
        context = {}
        
        for key, value in variables.items():
            if filters and key in filters:
                filter_func = filters[key]
                if callable(filter_func):
                    value = filter_func(value)
            
            context[key] = self._make_template_safe(value)
        
        context.update(self._get_utility_functions())
        return context
    
    def _make_template_safe(self, value):
        """Convert a value to a template-safe format."""
        if value is None:
            return ""
        
        # Keep datetime objects as-is for filters to work
        if isinstance(value, (datetime, timedelta)):
            return value
        
        if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
            return str(value)
        
        return value
    
    def _get_utility_functions(self) -> dict:
        """Get utility functions for template context."""
        return {
            'now': datetime.now,
            'utcnow': datetime.utcnow,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'json_dumps': json.dumps,
            'json_loads': json.loads,
        }


class TemplateEngine:
    """Core template engine using Jinja2 with security and validation."""
    
    def __init__(self, template_dir=None, enable_cache: bool = True):
        """Initialize template engine."""
        self.template_dir = template_dir
        self.enable_cache = enable_cache
        self.env = self._create_environment()
        self.validator = TemplateValidator(self.env)
        self.injector = VariableInjector()
        self._template_cache = {}
    
    def _create_environment(self) -> SandboxedEnvironment:
        """Create a secure Jinja2 environment."""
        loader = DictLoader({})
        
        env = SandboxedEnvironment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            auto_reload=not self.enable_cache,
            cache_size=400 if self.enable_cache else 0,
            undefined=StrictUndefined
        )
        
        env.filters.update(self._get_custom_filters())
        return env
    
    def _get_custom_filters(self) -> dict:
        """Get custom Jinja2 filters."""
        return {
            'json': json.dumps,
            'from_json': json.loads,
            'datetime_format': lambda dt, fmt='%Y-%m-%d %H:%M:%S': dt.strftime(fmt) if dt else '',
            'truncate_words': lambda text, length=50: ' '.join(text.split()[:length]) + ('...' if len(text.split()) > length else ''),
            'default_if_none': lambda value, default='': default if value is None else value,
            'safe_string': lambda value: str(value) if value is not None else '',
        }
    
    async def render_template(self, template_string: str, variables: dict, 
                            validate: bool = True, required_vars: set = None) -> str:
        """Render a template string with variables."""
        try:
            if validate:
                validation_result = self.validate_template(template_string, required_vars or set())
                if not validation_result.is_valid:
                    raise TemplateError(f"Template validation failed: {validation_result.errors}")
            
            template = self._get_template(template_string)
            context = self.injector.prepare_context(variables)
            rendered = template.render(context)
            return rendered
            
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable in template: {str(e)}") from e
        except TemplateSyntaxError as e:
            raise TemplateError(f"Template syntax error: {e.message} at line {e.lineno}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {str(e)}") from e
    
    def _get_template(self, template_string: str) -> Template:
        """Get template from cache or create new one."""
        cache_key = hash(template_string)
        
        if self.enable_cache and cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        template = self.env.from_string(template_string)
        
        if self.enable_cache:
            self._template_cache[cache_key] = template
        
        return template
    
    def validate_template(self, template_string: str, required_vars: set,
                         available_vars: set = None) -> TemplateValidationResult:
        """Validate template syntax, variables, and security."""
        results = [
            self.validator.validate_syntax(template_string),
            self.validator.validate_variables(template_string, required_vars, available_vars),
            self.validator.validate_security(template_string)
        ]
        
        combined_result = TemplateValidationResult(True)
        for result in results:
            if not result.is_valid:
                combined_result.is_valid = False
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
        
        return combined_result
    
    def extract_variables(self, template_string: str) -> set:
        """Extract all variables used in a template."""
        return self.validator.extract_variables(template_string)
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self._template_cache)


# Test classes
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
        assert "now" in context
        assert "json_dumps" in context
    
    def test_prepare_context_with_datetime(self):
        """Test context preparation with datetime objects."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        variables = {
            "created_at": now,
            "expires_in": timedelta(hours=1)
        }
        
        context = self.injector.prepare_context(variables)
        
        # DateTime objects should be kept as-is for filter compatibility
        assert context["created_at"] == now
        assert context["expires_in"] == timedelta(hours=1)
    
    def test_prepare_context_with_none_values(self):
        """Test context preparation with None values."""
        variables = {
            "name": "John",
            "optional_field": None,
            "empty_list": []
        }
        
        context = self.injector.prepare_context(variables)
        
        assert context["name"] == "John"
        assert context["optional_field"] == ""
        assert context["empty_list"] == []
    
    def test_prepare_context_with_filters(self):
        """Test context preparation with custom filters."""
        variables = {"score": 95}
        filters = {"score": lambda x: f"{x}%"}
        
        context = self.injector.prepare_context(variables, filters)
        
        assert context["score"] == "95%"


class TestTemplateEngine:
    """Test TemplateEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemplateEngine(enable_cache=False)
    
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
    async def test_render_template_validation_errors(self):
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
    
    def test_cache_operations(self):
        """Test template cache operations."""
        engine = TemplateEngine(enable_cache=True)
        
        assert engine.get_cache_size() == 0
        
        template_string = "Hello {{ name }}!"
        template = engine._get_template(template_string)
        assert template is not None
        assert engine.get_cache_size() == 1
        
        template2 = engine._get_template(template_string)
        assert template is template2
        assert engine.get_cache_size() == 1
        
        engine.clear_cache()
        assert engine.get_cache_size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
