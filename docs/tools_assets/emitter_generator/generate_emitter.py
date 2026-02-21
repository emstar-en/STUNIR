#!/usr/bin/env python3
"""STUNIR Emitter Generator

Scaffolds new emitters across all 4 STUNIR pipelines simultaneously:
- SPARK (Ada) - DO-178C Level A compliant
- Python - Reference implementation
- Rust - High-performance
- Haskell - Functional

Usage:
    generate_emitter.py --spec=<spec.yaml>
    generate_emitter.py --category=<name> --description=<desc> [options]
    generate_emitter.py --help

Examples:
    # From specification file
    generate_emitter.py --spec=specs/json_emitter.yaml
    
    # From command line
    generate_emitter.py \
        --category=json \
        --description="JSON serialization emitter" \
        --output-types=json,schema \
        --features=pretty_print,validation

Author: STUNIR Team
Version: 1.0.0
License: MIT
"""

import argparse
import sys
import os
import json
import yaml
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil


class EmitterGenerator:
    """Main emitter generator class."""
    
    def __init__(self, repo_root: Path):
        """Initialize generator.
        
        Args:
            repo_root: Path to STUNIR repository root
        """
        self.repo_root = Path(repo_root)
        self.template_dir = self.repo_root / "tools" / "emitter_generator" / "templates"
        self.generated_files: List[Dict[str, Any]] = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def load_spec(self, spec_path: Path) -> Dict[str, Any]:
        """Load emitter specification from YAML or JSON.
        
        Args:
            spec_path: Path to specification file
            
        Returns:
            Specification dictionary
        """
        with open(spec_path, 'r') as f:
            if spec_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif spec_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported spec format: {spec_path.suffix}")
    
    def load_template(self, template_name: str) -> str:
        """Load template file.
        
        Args:
            template_name: Template filename
            
        Returns:
            Template content
        """
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        return template_path.read_text(encoding='utf-8')
    
    def render_template(self, template: str, variables: Dict[str, str]) -> str:
        """Render template with variables.
        
        Args:
            template: Template string
            variables: Variable substitutions
            
        Returns:
            Rendered template
        """
        result = template
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result
    
    def prepare_variables(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """Prepare template variables from specification.
        
        Args:
            spec: Emitter specification
            
        Returns:
            Dictionary of template variables
        """
        category = spec['category']
        
        return {
            'CATEGORY': category.lower(),
            'CATEGORY_UPPER': category.upper(),
            'CATEGORY_TITLE': category.title().replace('_', ''),
            'DESCRIPTION': spec.get('description', f'{category} emitter'),
            'TIMESTAMP': self.timestamp,
            'OUTPUT_TYPES': ', '.join(spec.get('output_types', [category])),
            'FEATURES': ', '.join(spec.get('features', [])),
            'CONFIG_FIELDS': self._generate_config_fields(spec),
            'DEFAULT_CONFIG': self._generate_default_config(spec),
            'MODULE_BODY': self._generate_module_body(spec),
            'CLI_ARGS': self._generate_cli_args(spec),
            'EXT': spec.get('extension', category),
            'TYPE_I32': spec.get('type_map', {}).get('i32', 'int32'),
            'TYPE_I64': spec.get('type_map', {}).get('i64', 'int64'),
            'TYPE_F32': spec.get('type_map', {}).get('f32', 'float'),
            'TYPE_F64': spec.get('type_map', {}).get('f64', 'double'),
            'TYPE_BOOL': spec.get('type_map', {}).get('bool', 'boolean'),
            'TYPE_STRING': spec.get('type_map', {}).get('string', 'string'),
            'CONFIG_DOCS': self._generate_config_docs(spec),
            'OUTPUT_FORMAT': spec.get('output_format', 'Standard format'),
            'EXAMPLE_INPUT': json.dumps(spec.get('example_input', {}), indent=2),
            'EXAMPLE_OUTPUT': spec.get('example_output', '// Example output'),
        }
    
    def _generate_config_fields(self, spec: Dict[str, Any]) -> str:
        """Generate configuration fields."""
        config = spec.get('config', {})
        if not config:
            return "Enable_Feature : Boolean;\n      Max_Size : Positive;"
        
        fields = []
        for name, field_type in config.items():
            ada_name = name.title().replace('_', ' ').title().replace(' ', '_')
            fields.append(f"{ada_name} : {field_type};")
        
        return "\n      ".join(fields)
    
    def _generate_default_config(self, spec: Dict[str, Any]) -> str:
        """Generate default configuration values."""
        defaults = spec.get('config_defaults', {})
        if not defaults:
            return "Enable_Feature => False,\n      Max_Size => 1024"
        
        values = []
        for name, value in defaults.items():
            ada_name = name.title().replace('_', ' ').title().replace(' ', '_')
            ada_value = str(value).title() if isinstance(value, bool) else str(value)
            values.append(f"{ada_name} => {ada_value}")
        
        return ",\n      ".join(values)
    
    def _generate_module_body(self, spec: Dict[str, Any]) -> str:
        """Generate module body placeholder."""
        return f"// {spec['category'].title()} implementation goes here"
    
    def _generate_cli_args(self, spec: Dict[str, Any]) -> str:
        """Generate CLI argument definitions."""
        features = spec.get('features', [])
        if not features:
            return "# No additional CLI arguments"
        
        args = []
        for feature in features:
            args.append(f"parser.add_argument('--{feature}', action='store_true', help='Enable {feature}')")
        
        return "\n    ".join(args)
    
    def _generate_config_docs(self, spec: Dict[str, Any]) -> str:
        """Generate configuration documentation."""
        config = spec.get('config', {})
        if not config:
            return "Default configuration is used."
        
        docs = ["Configuration options:\n"]
        for name, desc in config.items():
            docs.append(f"- `{name}`: {desc}")
        
        return "\n".join(docs)
    
    def generate_spark_emitter(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
        """Generate SPARK (Ada) emitter files.
        
        Args:
            spec: Emitter specification
            variables: Template variables
        """
        category = spec['category']
        spark_dir = self.repo_root / "targets" / "spark" / category
        spark_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  \u2699\ufe0f  Generating SPARK emitter in {spark_dir}...")
        
        # Generate specification (.ads)
        spec_template = self.load_template("spark_spec.ads.template")
        spec_content = self.render_template(spec_template, variables)
        spec_file = spark_dir / f"{category}_emitter.ads"
        spec_file.write_text(spec_content, encoding='utf-8')
        self.generated_files.append({'path': str(spec_file.relative_to(self.repo_root)), 'language': 'spark'})
        
        # Generate body (.adb)
        body_template = self.load_template("spark_body.adb.template")
        body_content = self.render_template(body_template, variables)
        body_file = spark_dir / f"{category}_emitter.adb"
        body_file.write_text(body_content, encoding='utf-8')
        self.generated_files.append({'path': str(body_file.relative_to(self.repo_root)), 'language': 'spark'})
        
        # Generate test
        test_template = self.load_template("test_spark.adb.template")
        test_content = self.render_template(test_template, variables)
        test_file = spark_dir / f"test_{category}_emitter.adb"
        test_file.write_text(test_content, encoding='utf-8')
        self.generated_files.append({'path': str(test_file.relative_to(self.repo_root)), 'language': 'spark'})
        
        print(f"    \u2705 Generated 3 SPARK files")
    
    def generate_python_emitter(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
        """Generate Python emitter files.
        
        Args:
            spec: Emitter specification
            variables: Template variables
        """
        category = spec['category']
        python_dir = self.repo_root / "targets" / category
        python_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  \u2699\ufe0f  Generating Python emitter in {python_dir}...")
        
        # Generate main emitter
        emitter_template = self.load_template("python_emitter.py.template")
        emitter_content = self.render_template(emitter_template, variables)
        emitter_file = python_dir / "emitter.py"
        emitter_file.write_text(emitter_content, encoding='utf-8')
        emitter_file.chmod(0o755)  # Make executable
        self.generated_files.append({'path': str(emitter_file.relative_to(self.repo_root)), 'language': 'python'})
        
        # Generate __init__.py
        init_file = python_dir / "__init__.py"
        init_content = f'"""STUNIR {category.title()} Emitter Package"""\n\nfrom .emitter import {variables["CATEGORY_TITLE"]}Emitter\n\n__all__ = ["{variables["CATEGORY_TITLE"]}Emitter"]\n'
        init_file.write_text(init_content, encoding='utf-8')
        self.generated_files.append({'path': str(init_file.relative_to(self.repo_root)), 'language': 'python'})
        
        # Generate test
        test_template = self.load_template("test_python.py.template")
        test_content = self.render_template(test_template, variables)
        test_file = python_dir / "test_emitter.py"
        test_file.write_text(test_content, encoding='utf-8')
        self.generated_files.append({'path': str(test_file.relative_to(self.repo_root)), 'language': 'python'})
        
        print(f"    \u2705 Generated 3 Python files")
    
    def generate_rust_emitter(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
        """Generate Rust emitter files.
        
        Args:
            spec: Emitter specification
            variables: Template variables
        """
        category = spec['category']
        rust_dir = self.repo_root / "targets" / "rust" / category
        rust_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  \u2699\ufe0f  Generating Rust emitter in {rust_dir}...")
        
        # Generate mod.rs
        emitter_template = self.load_template("rust_emitter.rs.template")
        emitter_content = self.render_template(emitter_template, variables)
        mod_file = rust_dir / "mod.rs"
        mod_file.write_text(emitter_content, encoding='utf-8')
        self.generated_files.append({'path': str(mod_file.relative_to(self.repo_root)), 'language': 'rust'})
        
        print(f"    \u2705 Generated 1 Rust file")
    
    def generate_haskell_emitter(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
        """Generate Haskell emitter files.
        
        Args:
            spec: Emitter specification
            variables: Template variables
        """
        category_title = variables['CATEGORY_TITLE']
        haskell_dir = self.repo_root / "targets" / "haskell" / "src" / "STUNIR" / "Emitters"
        haskell_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  \u2699\ufe0f  Generating Haskell emitter in {haskell_dir}...")
        
        # Generate module file
        emitter_template = self.load_template("haskell_emitter.hs.template")
        emitter_content = self.render_template(emitter_template, variables)
        hs_file = haskell_dir / f"{category_title}.hs"
        hs_file.write_text(emitter_content, encoding='utf-8')
        self.generated_files.append({'path': str(hs_file.relative_to(self.repo_root)), 'language': 'haskell'})
        
        print(f"    \u2705 Generated 1 Haskell file")
    
    def generate_documentation(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
        """Generate README documentation.
        
        Args:
            spec: Emitter specification
            variables: Template variables
        """
        category = spec['category']
        doc_file = self.repo_root / "targets" / category / "README.md"
        
        print(f"  \u2699\ufe0f  Generating documentation...")
        
        readme_template = self.load_template("README.md.template")
        readme_content = self.render_template(readme_template, variables)
        doc_file.write_text(readme_content, encoding='utf-8')
        self.generated_files.append({'path': str(doc_file.relative_to(self.repo_root)), 'language': 'markdown'})
        
        print(f"    \u2705 Generated README.md")

    def update_build_systems(self, spec: Dict[str, Any], pipelines: List[str]) -> None:
        """Update build system files to include new emitter.

        Args:
            spec: Emitter specification
            pipelines: List of target pipelines to update
        """
        category = spec['category']
        category_title = category.title().replace('_', '')

        print(f"  \u2699\ufe0f  Updating build systems...")

        # Update Rust lib.rs
        if 'rust' in pipelines:
            rust_lib = self.repo_root / "targets" / "rust" / "lib.rs"
            if rust_lib.exists():
                content = rust_lib.read_text()
                if f"pub mod {category};" not in content:
                    # Find a good place to insert (after other pub mod declarations)
                    lines = content.split('\n')
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('pub mod '):
                            insert_idx = i + 1

                    lines.insert(insert_idx, f"pub mod {category};")
                    rust_lib.write_text('\n'.join(lines))
                    print(f"    \u2705 Updated Rust lib.rs")

        # Update Haskell cabal file
        if 'haskell' in pipelines:
            cabal_file = self.repo_root / "targets" / "haskell" / "stunir-emitters.cabal"
            if cabal_file.exists():
                content = cabal_file.read_text()
                module_name = f"STUNIR.Emitters.{category_title}"
                if module_name not in content:
                    # Add to exposed-modules
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'exposed-modules:' in line.lower():
                            # Find the last module in the list
                            j = i + 1
                            while j < len(lines) and (lines[j].strip().startswith('STUNIR.') or not lines[j].strip()):
                                if lines[j].strip() and not lines[j].strip().startswith('--'):
                                    j += 1
                                else:
                                    break
                            lines.insert(j, f"    {module_name}")
                            break

                    cabal_file.write_text('\n'.join(lines))
                    print(f"    \u2705 Updated Haskell cabal file")

        print(f"    \u2705 Build systems updated")

    def validate_generated_code(self, spec: Dict[str, Any], pipelines: List[str]) -> bool:
        """Validate generated code compiles/parses.

        Args:
            spec: Emitter specification
            pipelines: List of target pipelines to validate

        Returns:
            True if validation passed
        """
        category = spec['category']

        print(f"  \u2699\ufe0f  Validating generated code...")

        # Validate Python
        if 'python' in pipelines:
            python_file = self.repo_root / "targets" / category / "emitter.py"
            if python_file.exists():
                result = subprocess.run(
                    ['python3', '-m', 'py_compile', str(python_file)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"    \u274c Python validation failed: {result.stderr}")
                    return False
                print(f"    \u2705 Python syntax valid")

        # Validate Rust (syntax check)
        if 'rust' in pipelines:
            rust_file = self.repo_root / "targets" / "rust" / category / "mod.rs"
            if rust_file.exists():
                # Just check basic syntax without full cargo build
                print(f"    \u2705 Rust file generated (run 'cargo check' to validate)")

        # Note: Full SPARK and Haskell compilation would require their toolchains
        if 'spark' in pipelines:
            print(f"    \u2139\ufe0f  SPARK validation requires: gprbuild -P stunir_tools.gpr")

        if 'haskell' in pipelines:
            print(f"    \u2139\ufe0f  Haskell validation requires: cabal build")

        return True
    
    def generate_manifest(self, spec: Dict[str, Any], pipelines: List[str]) -> Dict[str, Any]:
        """Generate manifest of generated files.

        Args:
            spec: Emitter specification
            pipelines: List of selected pipelines

        Returns:
            Manifest dictionary
        """
        return {
            'schema': 'stunir_emitter_generator_manifest_v1',
            'category': spec['category'],
            'description': spec.get('description', ''),
            'timestamp': self.timestamp,
            'generator_version': '1.0.0',
            'files': self.generated_files,
            'pipelines': pipelines,
        }
    
    def generate(self, spec: Dict[str, Any], validate: bool = True, pipelines: Optional[List[str]] = None, skip_docs: bool = False) -> Dict[str, Any]:
        """Generate emitter across selected pipelines.

        Args:
            spec: Emitter specification
            validate: Whether to validate generated code
            pipelines: List of pipelines to generate (default: all)
            skip_docs: Whether to skip documentation generation

        Returns:
            Generation manifest
        """
        category = spec['category']

        selected_pipelines = pipelines if pipelines is not None else ['spark', 'python', 'rust', 'haskell']

        print(f"\n\u2728 Generating {category.upper()} emitter across selected pipelines...\n")

        # Prepare variables
        variables = self.prepare_variables(spec)

        # Generate for each pipeline
        try:
            if 'spark' in selected_pipelines:
                self.generate_spark_emitter(spec, variables)
            if 'python' in selected_pipelines:
                self.generate_python_emitter(spec, variables)
            if 'rust' in selected_pipelines:
                self.generate_rust_emitter(spec, variables)
            if 'haskell' in selected_pipelines:
                self.generate_haskell_emitter(spec, variables)
            if not skip_docs:
                self.generate_documentation(spec, variables)
            self.update_build_systems(spec, selected_pipelines)

            # Validation
            if validate:
                self.validate_generated_code(spec, selected_pipelines)

            # Generate manifest
            manifest = self.generate_manifest(spec, selected_pipelines)

            print(f"\n\u2705 Successfully generated {len(self.generated_files)} files!")
            print(f"\u2139\ufe0f  Category: {category}")
            print(f"\u2139\ufe0f  Pipelines: {', '.join([p.upper() for p in selected_pipelines])}")

            return manifest

        except Exception as e:
            print(f"\n\u274c Generation failed: {e}", file=sys.stderr)
            raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='STUNIR Emitter Generator - Scaffold emitters across all 4 pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From specification file
  %(prog)s --spec=specs/json_emitter.yaml

  # From command line arguments
  %(prog)s --category=json --description="JSON emitter" \\
           --output-types=json,schema --features=validation

  # Without validation
  %(prog)s --spec=specs/xml_emitter.yaml --no-validate
        """
    )

    # Specification input
    spec_group = parser.add_mutually_exclusive_group(required=True)
    spec_group.add_argument('--spec', type=Path, help='Path to emitter specification (YAML/JSON)')
    spec_group.add_argument('--category', help='Emitter category name')

    # Command line specification
    parser.add_argument('--description', help='Emitter description')
    parser.add_argument('--output-types', help='Comma-separated output types')
    parser.add_argument('--features', help='Comma-separated feature list')

    # Options
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).parent.parent.parent,
                        help='STUNIR repository root (default: auto-detect)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation of generated code')
    parser.add_argument('--output-manifest', type=Path,
                        help='Write generation manifest to file')
    parser.add_argument('--pipelines', help='Comma-separated pipelines to generate (spark,python,rust,haskell)')
    parser.add_argument('--skip-docs', action='store_true', help='Skip README generation')

    args = parser.parse_args()

    # Initialize generator
    generator = EmitterGenerator(args.repo_root)

    # Load or build specification
    if args.spec:
        print(f"\u2699\ufe0f  Loading specification from {args.spec}...")
        spec = generator.load_spec(args.spec)
    else:
        if not args.description:
            parser.error("--description is required when using --category")

        spec = {
            'category': args.category,
            'description': args.description,
            'output_types': args.output_types.split(',') if args.output_types else [args.category],
            'features': args.features.split(',') if args.features else [],
        }

    selected_pipelines = None
    if args.pipelines:
        selected_pipelines = [p.strip().lower() for p in args.pipelines.split(',') if p.strip()]

    # Generate emitter
    try:
        manifest = generator.generate(spec, validate=not args.no_validate, pipelines=selected_pipelines, skip_docs=args.skip_docs)

        # Write manifest if requested
        if args.output_manifest:
            args.output_manifest.write_text(json.dumps(manifest, indent=2))
            print(f"\u2139\ufe0f  Manifest written to {args.output_manifest}")

        print(f"\n\u2728 Next steps:")
        print(f"  1. Review generated files")
        print(f"  2. Customize implementation logic")
        print(f"  3. Run tests:")
        print(f"     - Python: python3 -m pytest targets/{spec['category']}/test_emitter.py")
        print(f"     - Rust: cd targets/rust && cargo test {spec['category']}")
        print(f"     - SPARK: cd targets/spark/{spec['category']} && gprbuild test_{spec['category']}_emitter.adb")
        print(f"  4. Commit and push to devsite branch")
        
        return 0
        
    except Exception as e:
        print(f"\u274c Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
