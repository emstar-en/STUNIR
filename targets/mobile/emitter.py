#!/usr/bin/env python3
"""STUNIR Mobile Emitter - Emit Swift/Kotlin bindings.

This tool is part of the targets → mobile pipeline stage.
It converts STUNIR IR to mobile platform bindings.

Usage:
    emitter.py <ir.json> --output=<dir> [--platform=ios|android|both]
    emitter.py --help
"""

import json
import hashlib
import time
import sys
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class MobileEmitter:
    """Emitter for mobile platforms (iOS/Android)."""
    
    # Type mappings
    TYPE_MAP_SWIFT = {
        'i32': 'Int32', 'i64': 'Int64', 'f32': 'Float', 'f64': 'Double',
        'int': 'Int', 'long': 'Int64', 'float': 'Float', 'double': 'Double',
        'void': 'Void', 'bool': 'Bool', 'byte': 'UInt8', 'string': 'String'
    }
    
    TYPE_MAP_KOTLIN = {
        'i32': 'Int', 'i64': 'Long', 'f32': 'Float', 'f64': 'Double',
        'int': 'Int', 'long': 'Long', 'float': 'Float', 'double': 'Double',
        'void': 'Unit', 'bool': 'Boolean', 'byte': 'Byte', 'string': 'String'
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize mobile emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.platform = options.get('platform', 'both') if options else 'both'
        self.generated_files = []
        self.epoch = int(time.time())
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def _map_type_swift(self, ir_type):
        """Map IR type to Swift type."""
        return self.TYPE_MAP_SWIFT.get(ir_type, 'Int')
    
    def _map_type_kotlin(self, ir_type):
        """Map IR type to Kotlin type."""
        return self.TYPE_MAP_KOTLIN.get(ir_type, 'Int')
    
    def _emit_statement_swift(self, stmt, indent='        '):
        """Convert IR statement to Swift code."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                swift_type = self._map_type_swift(stmt.get('var_type', 'i32'))
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                return f'{indent}var {var_name}: {swift_type} = {init}'
            elif stmt_type == 'return':
                return f'{indent}return {stmt.get("value", "0")}'
            elif stmt_type == 'assign':
                return f'{indent}{stmt.get("target", "v0")} = {stmt.get("value", "0")}'
            elif stmt_type in ('add', 'sub', 'mul', 'div'):
                ops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
                op = ops.get(stmt_type, '+')
                return f'{indent}{stmt.get("dest", "v0")} = {stmt.get("left", "0")} {op} {stmt.get("right", "0")}'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'{indent}{func}({args})'
        return f'{indent}// nop'
    
    def _emit_statement_kotlin(self, stmt, indent='        '):
        """Convert IR statement to Kotlin code."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                kotlin_type = self._map_type_kotlin(stmt.get('var_type', 'i32'))
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                return f'{indent}var {var_name}: {kotlin_type} = {init}'
            elif stmt_type == 'return':
                return f'{indent}return {stmt.get("value", "0")}'
            elif stmt_type == 'assign':
                return f'{indent}{stmt.get("target", "v0")} = {stmt.get("value", "0")}'
            elif stmt_type in ('add', 'sub', 'mul', 'div'):
                ops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
                op = ops.get(stmt_type, '+')
                return f'{indent}{stmt.get("dest", "v0")} = {stmt.get("left", "0")} {op} {stmt.get("right", "0")}'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'{indent}{func}({args})'
        return f'{indent}// nop'
    
    def _emit_function_swift(self, func):
        """Emit Swift function."""
        name = func.get('name', 'func0')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        ret_type = self._map_type_swift(returns)
        param_str = ', '.join([
            f"{p.get('name', f'arg{i}')}: {self._map_type_swift(p.get('type', 'i32'))}"
            if isinstance(p, dict) else f'arg{i}: Int'
            for i, p in enumerate(params)
        ])
        
        ret_sig = '' if returns == 'void' else f' -> {ret_type}'
        
        lines = [
            f'    /// {name}',
            f'    func {name}({param_str}){ret_sig} {{'
        ]
        
        for stmt in body:
            lines.append(self._emit_statement_swift(stmt))
        
        if returns == 'void':
            pass
        elif not any(isinstance(s, dict) and s.get('type') == 'return' for s in body):
            lines.append('        return 0')
        
        lines.append('    }')
        return '\n'.join(lines)
    
    def _emit_function_kotlin(self, func):
        """Emit Kotlin function."""
        name = func.get('name', 'func0')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        ret_type = self._map_type_kotlin(returns)
        param_str = ', '.join([
            f"{p.get('name', f'arg{i}')}: {self._map_type_kotlin(p.get('type', 'i32'))}"
            if isinstance(p, dict) else f'arg{i}: Int'
            for i, p in enumerate(params)
        ])
        
        ret_sig = '' if returns == 'void' else f': {ret_type}'
        
        lines = [
            f'    /** {name} */',
            f'    fun {name}({param_str}){ret_sig} {{'
        ]
        
        for stmt in body:
            lines.append(self._emit_statement_kotlin(stmt))
        
        if returns == 'void':
            pass
        elif not any(isinstance(s, dict) and s.get('type') == 'return' for s in body):
            lines.append('        return 0')
        
        lines.append('    }')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit mobile platform files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        
        class_name = ''.join(w.capitalize() for w in module_name.split('_'))
        
        if self.platform in ('ios', 'both'):
            self._emit_ios(module_name, class_name, functions)
        
        if self.platform in ('android', 'both'):
            self._emit_android(module_name, class_name, functions)
        
        # Cross-platform interface
        self._emit_interface(module_name, class_name, functions)
        
        # README
        self._write_file('README.md', self._emit_readme(module_name, len(functions)))
        
        return f"Generated mobile bindings for {self.platform}"
    
    def _emit_ios(self, module_name, class_name, functions):
        """Emit iOS/Swift files."""
        lines = [
            f'// STUNIR Mobile Module: {module_name}',
            f'// Platform: iOS (Swift)',
            f'// Schema: stunir.mobile.ios.v1',
            f'// Epoch: {self.epoch}',
            '',
            'import Foundation',
            '',
            f'/// STUNIR Generated Module',
            f'public class {class_name} {{',
            '',
            '    public init() {}',
            ''
        ]
        
        for func in functions:
            lines.append(self._emit_function_swift(func))
            lines.append('')
        
        lines.append('}')
        
        swift_content = '\n'.join(lines)
        self._write_file(f'ios/{class_name}.swift', swift_content)
        
        # Package.swift for SPM
        package_swift = f"""// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "{class_name}",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "{class_name}", targets: ["{class_name}"])
    ],
    targets: [
        .target(name: "{class_name}", path: ".")
    ]
)
"""
        self._write_file('ios/Package.swift', package_swift)
    
    def _emit_android(self, module_name, class_name, functions):
        """Emit Android/Kotlin files."""
        package_name = f'com.stunir.{module_name.lower()}'
        
        lines = [
            f'// STUNIR Mobile Module: {module_name}',
            f'// Platform: Android (Kotlin)',
            f'// Schema: stunir.mobile.android.v1',
            f'// Epoch: {self.epoch}',
            '',
            f'package {package_name}',
            '',
            f'/** STUNIR Generated Module */',
            f'class {class_name} {{',
            ''
        ]
        
        for func in functions:
            lines.append(self._emit_function_kotlin(func))
            lines.append('')
        
        lines.append('}')
        
        kotlin_content = '\n'.join(lines)
        self._write_file(f'android/{class_name}.kt', kotlin_content)
        
        # Build.gradle.kts
        gradle = f"""plugins {{
    kotlin("jvm") version "1.9.0"
}}

group = "{package_name}"
version = "1.0.0"

repositories {{
    mavenCentral()
}}

dependencies {{
    implementation(kotlin("stdlib"))
}}
"""
        self._write_file('android/build.gradle.kts', gradle)
    
    def _emit_interface(self, module_name, class_name, functions):
        """Emit cross-platform interface definition."""
        interface = {
            'schema': 'stunir.mobile.interface.v1',
            'module': module_name,
            'class': class_name,
            'epoch': self.epoch,
            'functions': [
                {
                    'name': f.get('name', 'func0'),
                    'params': f.get('params', []),
                    'returns': f.get('returns', 'void')
                }
                for f in functions
            ]
        }
        self._write_file('interface.json', canonical_json(interface))
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        return f"""# {module_name} (Mobile)

Generated by STUNIR Mobile Emitter.

## Platforms

{self.platform.upper()}

## Files

### iOS
- `ios/{module_name.title()}.swift` - Swift implementation
- `ios/Package.swift` - Swift Package Manager config

### Android
- `android/{module_name.title()}.kt` - Kotlin implementation
- `android/build.gradle.kts` - Gradle build config

### Cross-Platform
- `interface.json` - Platform-agnostic interface definition

## iOS Integration

```swift
import {module_name.title()}

let module = {module_name.title()}()
let result = module.yourFunction()
```

## Android Integration

```kotlin
import com.stunir.{module_name.lower()}.{module_name.title()}

val module = {module_name.title()}()
val result = module.yourFunction()
```

## Statistics

- Functions: {func_count}
- Epoch: {self.epoch}

## Schema

stunir.mobile.ios.v1 / stunir.mobile.android.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.mobile.{self.platform}.manifest.v1',
            'epoch': self.epoch,
            'platform': self.platform,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.mobile.{self.platform}.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


def main():
    args = {'output': None, 'input': None, 'platform': 'both'}
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--platform='):
            args['platform'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'mobile_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = MobileEmitter(ir_data, out_dir, {'platform': args['platform']})
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"Mobile bindings emitted to {out_dir}/", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
