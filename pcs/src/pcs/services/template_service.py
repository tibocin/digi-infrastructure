"""
Filepath: pcs/src/pcs/services/template_service.py
Purpose: Template engine service for processing Jinja2 templates with variable injection and validation
Related Components: Jinja2, template validation, variable extraction, security filters
Tags: template-engine, jinja2, variable-injection, validation, security
"""

import re
import json
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from pathlib import Path

from jinja2 import (
    Environment, 
    BaseLoader, 
    DictLoader, 
    FileSystemLoader,
    Template,
    TemplateSyntaxError,
    Undefined,
    UndefinedError,
    select_autoescape,
    meta
)

# Custom Undefined class that raises errors
class StrictUndefined(Undefined):
    """Custom Undefined class that raises errors for undefined variables."""
    def __str__(self):
        raise UndefinedError(f"Variable '{self._undefined_name}' is undefined")
    
    def __repr__(self):
        return f"<StrictUndefined '{self._undefined_name}'>"
from jinja2.sandbox import SandboxedEnvironment

from ..core.exceptions import PCSError


class TemplateError(PCSError):
    """Custom exception for template-related errors."""
    pass


class TemplateValidationResult:
    """Result of template validation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
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
    """
    Template validator for syntax and security validation.
    
    Performs validation on Jinja2 templates to ensure:
    - Valid syntax
    - Required variables are available
    - Security constraints are met
    """
    
    def __init__(self, environment: Environment):
        """Initialize validator with Jinja2 environment."""
        self.env = environment
        
    def validate_syntax(self, template_string: str) -> TemplateValidationResult:
        """
        Validate template syntax.
        
        Args:
            template_string: Template string to validate
            
        Returns:
            TemplateValidationResult with validation results
        """
        result = TemplateValidationResult(True)
        
        try:
            # Parse template to check syntax
            self.env.parse(template_string)
        except TemplateSyntaxError as e:
            result.add_error(f"Template syntax error: {e.message} at line {e.lineno}")
        except Exception as e:
            result.add_error(f"Unexpected template error: {str(e)}")
        
        return result
    
    def validate_variables(
        self, 
        template_string: str, 
        required_vars: Set[str],
        available_vars: Optional[Set[str]] = None
    ) -> TemplateValidationResult:
        """
        Validate template variables.
        
        Args:
            template_string: Template string to validate
            required_vars: Set of required variable names
            available_vars: Set of available variable names (optional)
            
        Returns:
            TemplateValidationResult with validation results
        """
        result = TemplateValidationResult(True)
        
        try:
            # Extract variables from template
            template_vars = self.extract_variables(template_string)
            
            # Check for missing required variables
            missing_required = required_vars - template_vars
            if missing_required:
                for var in missing_required:
                    result.add_error(f"Required variable '{var}' not found in template")
            
            # Check for undefined variables if available_vars provided
            if available_vars is not None:
                undefined_vars = template_vars - available_vars
                if undefined_vars:
                    for var in undefined_vars:
                        result.add_warning(f"Variable '{var}' may not be available at runtime")
        
        except Exception as e:
            result.add_error(f"Error validating variables: {str(e)}")
        
        return result
    
    def extract_variables(self, template_string: str) -> Set[str]:
        """
        Extract all variables used in a template.
        
        Args:
            template_string: Template string to analyze
            
        Returns:
            Set of variable names used in the template
        """
        try:
            ast = self.env.parse(template_string)
            return meta.find_undeclared_variables(ast)
        except Exception as e:
            raise TemplateError(f"Failed to extract variables: {str(e)}") from e
    
    def validate_security(self, template_string: str) -> TemplateValidationResult:
        """
        Validate template for security issues.
        
        Args:
            template_string: Template string to validate
            
        Returns:
            TemplateValidationResult with security validation results
        """
        result = TemplateValidationResult(True)
        
        # Check for potentially dangerous patterns
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
    """
    Variable injector for template context preparation.
    
    Handles variable injection, type conversion, and context preparation
    for template rendering.
    """
    
    def __init__(self):
        """Initialize variable injector."""
        pass
    
    def prepare_context(
        self, 
        variables: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare template context with variables and filters.
        
        Args:
            variables: Dictionary of variables to inject
            filters: Optional additional filters or transformations
            
        Returns:
            Prepared context dictionary
        """
        context = {}
        
        # Process each variable
        for key, value in variables.items():
            # Apply any filters if specified
            if filters and key in filters:
                filter_func = filters[key]
                if callable(filter_func):
                    value = filter_func(value)
            
            # Convert value to template-safe format
            context[key] = self._make_template_safe(value)
        
        # Add utility functions
        context.update(self._get_utility_functions())
        
        return context
    
    def _make_template_safe(self, value: Any) -> Any:
        """
        Convert a value to a template-safe format.
        
        Args:
            value: Value to convert
            
        Returns:
            Template-safe value
        """
        # Handle None values
        if value is None:
            return ""
        
        # Handle datetime objects - preserve for filters
        if isinstance(value, datetime):
            return value
        
        # Handle timedelta objects
        if isinstance(value, timedelta):
            return str(value)
        
        # Handle complex objects by converting to string
        if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
            return str(value)
        
        return value
    
    def _get_utility_functions(self) -> Dict[str, Any]:
        """
        Get utility functions for template context.
        
        Returns:
            Dictionary of utility functions
        """
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
    """
    Core template engine using Jinja2 with security and validation.
    
    Provides template rendering with:
    - Variable injection
    - Custom filters
    - Security validation
    - Template caching
    - Error handling
    """
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None, enable_cache: bool = True):
        """
        Initialize template engine.
        
        Args:
            template_dir: Directory containing template files (optional)
            enable_cache: Whether to enable template caching
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.enable_cache = enable_cache
        
        # Initialize Jinja2 environment with security
        self.env = self._create_environment()
        
        # Initialize components
        self.validator = TemplateValidator(self.env)
        self.injector = VariableInjector()
        
        # Template cache
        self._template_cache: Dict[str, Template] = {}
    
    def _create_environment(self) -> SandboxedEnvironment:
        """
        Create a secure Jinja2 environment.
        
        Returns:
            Configured SandboxedEnvironment
        """
        # Choose loader based on template directory
        if self.template_dir and self.template_dir.exists():
            loader = FileSystemLoader(str(self.template_dir))
        else:
            # Use DictLoader for string templates
            loader = DictLoader({})
        
        # Create sandboxed environment for security
        env = SandboxedEnvironment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            auto_reload=not self.enable_cache,
            cache_size=400 if self.enable_cache else 0,
            undefined=StrictUndefined  # Use custom class that raises errors
        )
        
        # Add custom filters
        env.filters.update(self._get_custom_filters())
        
        return env
    
    def _get_custom_filters(self) -> Dict[str, Any]:
        """
        Get custom Jinja2 filters.
        
        Returns:
            Dictionary of custom filter functions
        """
        return {
            'json': json.dumps,
            'from_json': json.loads,
            'datetime_format': lambda dt, fmt='%Y-%m-%d %H:%M:%S': dt.strftime(fmt) if dt else '',
            'truncate_words': lambda text, length=50: ' '.join(text.split()[:length]) + ('...' if len(text.split()) > length else ''),
            'default_if_none': lambda value, default='': default if value is None else value,
            'safe_string': lambda value: str(value) if value is not None else '',
        }
    
    async def render_template(
        self, 
        template_string: str, 
        variables: Dict[str, Any],
        validate: bool = True,
        required_vars: Optional[Set[str]] = None
    ) -> str:
        """
        Render a template string with variables.
        
        Args:
            template_string: Template string to render
            variables: Variables to inject into template
            validate: Whether to validate template before rendering
            required_vars: Set of required variable names
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateError: If template validation or rendering fails
        """
        try:
            # Validate template if requested
            if validate:
                validation_result = self.validate_template(
                    template_string, 
                    required_vars or set()
                )
                if not validation_result.is_valid:
                    raise TemplateError(f"Template validation failed: {validation_result.errors}")
            
            # Get or create template
            template = self._get_template(template_string)
            
            # Prepare context
            context = self.injector.prepare_context(variables)
            
            # Render template
            rendered = template.render(context)
            
            return rendered
            
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable in template: {str(e)}") from e
        except TemplateSyntaxError as e:
            raise TemplateError(f"Template syntax error: {e.message} at line {e.lineno}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {str(e)}") from e
    
    def _get_template(self, template_string: str) -> Template:
        """
        Get template from cache or create new one.
        
        Args:
            template_string: Template string
            
        Returns:
            Jinja2 Template object
        """
        # Use template string hash as cache key
        cache_key = hash(template_string)
        
        if self.enable_cache and cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Create new template
        template = self.env.from_string(template_string)
        
        # Cache if enabled
        if self.enable_cache:
            self._template_cache[cache_key] = template
        
        return template
    
    def validate_template(
        self, 
        template_string: str, 
        required_vars: Set[str],
        available_vars: Optional[Set[str]] = None
    ) -> TemplateValidationResult:
        """
        Validate template syntax, variables, and security.
        
        Args:
            template_string: Template string to validate
            required_vars: Set of required variable names
            available_vars: Set of available variable names (optional)
            
        Returns:
            TemplateValidationResult with validation results
        """
        # Combine all validation results
        results = [
            self.validator.validate_syntax(template_string),
            self.validator.validate_variables(template_string, required_vars, available_vars),
            self.validator.validate_security(template_string)
        ]
        
        # Combine results
        combined_result = TemplateValidationResult(True)
        for result in results:
            if not result.is_valid:
                combined_result.is_valid = False
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
        
        return combined_result
    
    def extract_variables(self, template_string: str) -> Set[str]:
        """
        Extract all variables used in a template.
        
        Args:
            template_string: Template string to analyze
            
        Returns:
            Set of variable names used in the template
        """
        return self.validator.extract_variables(template_string)
    
    def apply_filters(self, template_string: str, filters: Dict[str, Any]) -> str:
        """
        Apply additional filters to a template.
        
        Args:
            template_string: Template string
            filters: Dictionary of filters to apply
            
        Returns:
            Template string with applied filters
        """
        # This is a simplified implementation
        # In practice, you might want to modify the template string or environment
        for filter_name, filter_func in filters.items():
            if callable(filter_func):
                self.env.filters[filter_name] = filter_func
        
        return template_string
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self._template_cache)
