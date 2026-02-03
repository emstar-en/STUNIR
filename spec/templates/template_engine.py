#!/usr/bin/env python3
"""STUNIR Spec Template Engine

Pipeline Stage: spec -> templates
Issue: #1079

Provides template processing and validation for STUNIR specs.
"""

import json
import hashlib
import re
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of string data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class TemplateEngine:
    """STUNIR Spec Template Engine.
    
    Processes spec templates with variable substitution
    and schema validation.
    """
    
    # Variable pattern: ${variable_name} or ${variable_name:default}
    VAR_PATTERN = re.compile(r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([^}]*))?\}')
    
    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize template engine.
        
        Args:
            template_dir: Directory containing templates. Defaults to spec/templates/.
        """
        self.template_dir = template_dir or Path(__file__).parent
        self._cache: Dict[str, Dict] = {}
    
    def load_template(self, name: str) -> Dict[str, Any]:
        """Load a template by name.
        
        Args:
            name: Template name (without .json extension)
            
        Returns:
            Template dictionary
        """
        if name in self._cache:
            return self._cache[name]
        
        template_path = self.template_dir / f"{name}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {name}")
        
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        self._cache[name] = template
        return template
    
    def substitute(self, template: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in a template.
        
        Args:
            template: Template dictionary
            variables: Variable substitutions
            
        Returns:
            Template with variables substituted
        """
        return self._substitute_recursive(template, variables)
    
    def _substitute_recursive(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute variables."""
        if isinstance(obj, str):
            return self._substitute_string(obj, variables)
        elif isinstance(obj, dict):
            return {k: self._substitute_recursive(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_recursive(item, variables) for item in obj]
        else:
            return obj
    
    def _substitute_string(self, s: str, variables: Dict[str, Any]) -> Any:
        """Substitute variables in a string."""
        def replace(match):
            var_name = match.group(1)
            default = match.group(2)
            
            if var_name in variables:
                value = variables[var_name]
                # If entire string is just the variable, return the value directly
                if match.group(0) == s:
                    return value
                return str(value)
            elif default is not None:
                return default
            else:
                raise ValueError(f"Missing required variable: {var_name}")
        
        # Check if entire string is a variable
        full_match = self.VAR_PATTERN.fullmatch(s)
        if full_match:
            return replace(full_match)
        
        return self.VAR_PATTERN.sub(replace, s)
    
    def validate_template(self, template: Dict[str, Any]) -> List[str]:
        """Validate a template structure.
        
        Args:
            template: Template to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields for STUNIR spec
        required_fields = ['schema', 'name']
        for field in required_fields:
            if field not in template:
                errors.append(f"Missing required field: {field}")
        
        # Schema version check
        if 'schema' in template:
            if not template['schema'].startswith('stunir.'):
                errors.append(f"Invalid schema prefix: {template['schema']}")
        
        return errors
    
    def instantiate(self, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Load a template and instantiate it with variables.
        
        Args:
            template_name: Name of template to load
            variables: Variable substitutions
            
        Returns:
            Instantiated spec
        """
        template = self.load_template(template_name)
        return self.substitute(template, variables)
    
    def get_template_hash(self, template: Dict[str, Any]) -> str:
        """Get deterministic hash of template."""
        return compute_sha256(canonical_json(template))


def main():
    """CLI interface for template engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Spec Template Engine')
    parser.add_argument('template', help='Template name or path')
    parser.add_argument('-v', '--var', action='append', nargs=2, metavar=('NAME', 'VALUE'),
                        help='Variable substitution (can be repeated)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--validate', action='store_true', help='Validate template only')
    
    args = parser.parse_args()
    
    engine = TemplateEngine()
    
    # Load template
    if args.template.endswith('.json'):
        with open(args.template, 'r') as f:
            template = json.load(f)
    else:
        template = engine.load_template(args.template)
    
    # Validate if requested
    if args.validate:
        errors = engine.validate_template(template)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)
        print("Template is valid")
        sys.exit(0)
    
    # Substitute variables
    variables = {}
    if args.var:
        for name, value in args.var:
            # Try to parse as JSON, fallback to string
            try:
                variables[name] = json.loads(value)
            except json.JSONDecodeError:
                variables[name] = value
    
    result = engine.substitute(template, variables)
    
    # Output
    output = canonical_json(result)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
            f.write('\n')
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    
    # Print hash
    print(f"Hash: {compute_sha256(output)}", file=sys.stderr)


if __name__ == '__main__':
    main()
