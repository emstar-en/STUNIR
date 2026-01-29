#!/usr/bin/env python3
"""STUNIR Code Emitter - Dispatches code generation to target emitters.

This tool is part of the tools → emitters pipeline stage.
It coordinates code generation for various target languages.

Phase 1 Enhancement Integration:
- Optionally runs enhancement pipeline to produce EnhancementContext
- Passes context to emitters that support it
- Maintains backward compatibility with emitters that don't use context

Usage:
    emit_code.py <ir.json> --target=<lang> [--output=<file>] [--enhance] [--opt-level=O2]

Supported targets: python, rust, c, go, haskell, java, node, wasm
"""

import json
import sys
import os
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Supported targets and their emitter modules
TARGET_EMITTERS = {
    'python': 'ir_to_python',
    'rust': 'ir_to_rust',
    'c': 'ir_to_c',
    'c89': 'ir_to_c',
    'c99': 'ir_to_c',
    'cpp': 'ir_to_cpp',
    'go': 'ir_to_go',
    'haskell': 'ir_to_haskell',
    'java': 'ir_to_java',
    'node': 'ir_to_node',
    'wasm': 'ir_to_wasm',
    'ruby': 'ir_to_ruby',
    'php': 'ir_to_php',
    'csharp': 'ir_to_csharp',
    'dotnet': 'ir_to_dotnet',
    'erlang': 'ir_to_erlang',
    'prolog': 'ir_to_prolog',
    'lisp': 'ir_to_lisp',
    'smt2': 'ir_to_smt2',
    'asm': 'ir_to_asm'
}


def load_emitter(target):
    """Load the emitter module for a target language."""
    if target not in TARGET_EMITTERS:
        return None
    
    module_name = TARGET_EMITTERS[target]
    tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(tools_dir, f"{module_name}.py")
    
    if not os.path.exists(module_path):
        return None
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Failed to load {module_name}: {e}")
        return None


def run_enhancement_pipeline(ir_data, target, opt_level='O2'):
    """Run the enhancement pipeline to produce EnhancementContext.
    
    Args:
        ir_data: The IR data dictionary.
        target: Target language.
        opt_level: Optimization level (O0, O1, O2, O3).
        
    Returns:
        EnhancementContext or None if pipeline not available.
    """
    try:
        from tools.integration import create_pipeline, PipelineConfig
        
        config = PipelineConfig(
            enable_control_flow=True,
            enable_type_analysis=True,
            enable_semantic_analysis=True,
            enable_memory_analysis=True,
            enable_optimization=(opt_level != 'O0'),
            optimization_level=opt_level
        )
        
        pipeline = create_pipeline(target, opt_level)
        pipeline.config = config
        
        logger.info(f"Running enhancement pipeline for target={target}, opt={opt_level}")
        context = pipeline.run_all_enhancements(ir_data)
        
        # Log status summary
        summary = context.get_status_summary()
        for enhancement, status in summary.items():
            logger.info(f"  {enhancement}: {status}")
        
        return context
        
    except ImportError as e:
        logger.warning(f"Enhancement pipeline not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Enhancement pipeline failed: {e}")
        return None


def emit_code_with_codegen(ir_data, target, enhancement_context=None):
    """Generate code using the Phase 2 code generators.
    
    Args:
        ir_data: IR data dictionary.
        target: Target language.
        enhancement_context: Optional EnhancementContext.
        
    Returns:
        Generated code string.
    """
    try:
        from tools.codegen import get_generator, get_supported_targets
        
        if target.lower() in [t.lower() for t in get_supported_targets()]:
            generator = get_generator(target, enhancement_context)
            return generator.generate_module(ir_data)
    except ImportError:
        logger.debug("Code generators not available, using fallback")
    except Exception as e:
        logger.warning(f"Code generator failed: {e}, using fallback")
    
    return None  # Signal to use fallback


def emit_code_fallback(ir_data, target, enhancement_context=None):
    """Fallback code emitter when specific emitter is not available.
    
    Args:
        ir_data: IR data dictionary.
        target: Target language.
        enhancement_context: Optional EnhancementContext.
        
    Returns:
        Generated code string.
    """
    # Try the new code generators first
    codegen_result = emit_code_with_codegen(ir_data, target, enhancement_context)
    if codegen_result is not None:
        return codegen_result
    
    module_name = ir_data.get('ir_module', 'module')
    functions = ir_data.get('ir_functions', [])
    
    lines = [
        f"# STUNIR Generated Code for {target}",
        f"# Module: {module_name}",
        f"# Schema: {ir_data.get('schema', 'unknown')}",
    ]
    
    # Add enhancement context info if available
    if enhancement_context is not None:
        lines.append(f"# Enhancement Context: Available")
        summary = enhancement_context.get_status_summary()
        for key, status in summary.items():
            lines.append(f"#   {key}: {status}")
    else:
        lines.append(f"# Enhancement Context: Not available")
    
    lines.append("")
    
    for func in functions:
        func_name = func.get('name', 'unnamed')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        
        param_str = ', '.join(str(p) for p in params) if params else ''
        lines.append(f"# Function: {func_name}({param_str}) -> {returns}")
        
        # If context available, add enhancement info
        if enhancement_context is not None:
            cfg = enhancement_context.get_function_cfg(func_name)
            if cfg:
                lines.append(f"#   CFG: Available")
            loops = enhancement_context.get_loops(func_name)
            if loops:
                lines.append(f"#   Loops: {len(loops)} detected")
        
        lines.append(f"# Body: {func.get('body', [])}")
        lines.append("")
    
    return '\n'.join(lines)


def emit_code_with_enhancements(ir_data, target, enhancement_context, output_dir=None):
    """Emit code using enhanced emitter if available.
    
    Args:
        ir_data: IR data dictionary.
        target: Target language.
        enhancement_context: EnhancementContext from pipeline.
        output_dir: Optional output directory.
        
    Returns:
        Generated code or manifest.
    """
    try:
        from tools.emitters.base_emitter import BaseEmitter
        
        # Try to find enhanced emitter class
        emitter_module = load_emitter(target)
        
        if emitter_module is None:
            return emit_code_fallback(ir_data, target, enhancement_context)
        
        # Check if module has an emitter class that accepts context
        for attr_name in dir(emitter_module):
            attr = getattr(emitter_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseEmitter) and 
                attr is not BaseEmitter):
                # Found enhanced emitter class
                import tempfile
                out_dir = output_dir or tempfile.mkdtemp(prefix='stunir_emit_')
                emitter = attr(ir_data, out_dir, enhancement_context)
                return emitter.emit()
        
        # Fall back to module-level emit function
        if hasattr(emitter_module, 'emit'):
            # Try to call with context if signature supports it
            import inspect
            sig = inspect.signature(emitter_module.emit)
            if 'enhancement_context' in sig.parameters or 'context' in sig.parameters:
                return emitter_module.emit(ir_data, enhancement_context=enhancement_context)
            else:
                return emitter_module.emit(ir_data)
        
        return emit_code_fallback(ir_data, target, enhancement_context)
        
    except Exception as e:
        logger.warning(f"Enhanced emission failed, using fallback: {e}")
        return emit_code_fallback(ir_data, target, enhancement_context)


def parse_args(argv):
    """Parse command line arguments."""
    args = {
        'target': None, 
        'output': None, 
        'input': None,
        'enhance': False,
        'opt_level': 'O2',
        'verbose': False
    }
    
    for arg in argv[1:]:
        if arg.startswith('--target='):
            args['target'] = arg.split('=', 1)[1].lower()
        elif arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--opt-level='):
            args['opt_level'] = arg.split('=', 1)[1].upper()
        elif arg == '--enhance':
            args['enhance'] = True
        elif arg in ('-v', '--verbose'):
            args['verbose'] = True
        elif arg in ('-h', '--help'):
            args['help'] = True
        elif not arg.startswith('--'):
            args['input'] = arg
    
    return args


def print_usage():
    """Print usage information."""
    print(f"Usage: {sys.argv[0]} <ir.json> --target=<lang> [options]", file=sys.stderr)
    print("\nOptions:", file=sys.stderr)
    print("  --target=LANG    Target language (required)", file=sys.stderr)
    print("  --output=FILE    Output file path", file=sys.stderr)
    print("  --enhance        Run enhancement pipeline", file=sys.stderr)
    print("  --opt-level=Ox   Optimization level (O0, O1, O2, O3)", file=sys.stderr)
    print("  -v, --verbose    Verbose output", file=sys.stderr)
    print("  -h, --help       Show this help", file=sys.stderr)
    print("\nSupported targets:", file=sys.stderr)
    for target in sorted(TARGET_EMITTERS.keys()):
        print(f"  - {target}", file=sys.stderr)
    print("\nEnhancement Pipeline:", file=sys.stderr)
    print("  The --enhance flag runs the STUNIR enhancement pipeline", file=sys.stderr)
    print("  which provides control flow, type, semantic, memory, and", file=sys.stderr)
    print("  optimization analysis to emitters for better code generation.", file=sys.stderr)


def main():
    args = parse_args(sys.argv)
    
    if args.get('help'):
        print_usage()
        sys.exit(0)
    
    if args.get('verbose'):
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args['input'] or not args['target']:
        print_usage()
        sys.exit(1)
    
    try:
        # Read IR file
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        enhancement_context = None
        
        # Run enhancement pipeline if requested
        if args['enhance']:
            enhancement_context = run_enhancement_pipeline(
                ir_data, 
                args['target'], 
                args['opt_level']
            )
        
        # Try to load specific emitter
        emitter = load_emitter(args['target'])
        
        if enhancement_context is not None:
            # Use enhanced emission
            code = emit_code_with_enhancements(
                ir_data, 
                args['target'], 
                enhancement_context,
                os.path.dirname(args['output']) if args['output'] else None
            )
        elif emitter and hasattr(emitter, 'emit'):
            code = emitter.emit(ir_data)
        elif emitter and hasattr(emitter, 'main'):
            # Some emitters use different entry point
            code = emit_code_fallback(ir_data, args['target'], enhancement_context)
        else:
            code = emit_code_fallback(ir_data, args['target'], enhancement_context)
        
        # Handle different return types
        if isinstance(code, dict):
            # Manifest returned - convert to JSON
            code = json.dumps(code, indent=2)
        
        # Output
        if args['output']:
            with open(args['output'], 'w') as f:
                f.write(code)
            logger.info(f"Code emitted to {args['output']}")
        else:
            print(code)
        
        logger.info(f"Target: {args['target']}")
        logger.info(f"Lines: {len(code.splitlines())}")
        if enhancement_context:
            logger.info(f"Enhancements: enabled")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
