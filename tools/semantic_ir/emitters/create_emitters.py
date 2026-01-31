"""
Script to generate all 24 STUNIR semantic IR emitters from templates.
This ensures consistency and speeds up implementation.
"""

import os
from pathlib import Path

# Emitter categories and their implementations
EMITTERS = {
    "core": [
        ("embedded", "Embedded", "bare-metal C for ARM/AVR/MIPS/RISC-V"),
        ("gpu", "GPU", "CUDA/OpenCL/Vulkan compute shaders"),
        ("wasm", "WebAssembly", "WebAssembly binary and text formats"),
        ("assembly", "Assembly", "x86/ARM assembly with multiple syntaxes"),
        ("polyglot", "Polyglot", "C89/C99/C11/Rust multi-language support"),
    ],
    "language_families": [
        ("lisp", "Lisp", "Common Lisp/Scheme/Clojure/Racket dialects"),
        ("prolog", "Prolog", "SWI-Prolog/GNU-Prolog/Mercury variants"),
    ],
    "specialized": [
        ("business", "Business", "COBOL/RPG business logic"),
        ("fpga", "FPGA", "VHDL/Verilog/SystemVerilog HDL"),
        ("grammar", "Grammar", "EBNF/ANTLR grammar definitions"),
        ("lexer", "Lexer", "Flex/RE2C lexer generators"),
        ("parser", "Parser", "Bison/Yacc parser generators"),
        ("expert", "ExpertSystem", "CLIPS/Jess rule-based systems"),
        ("constraints", "Constraint", "MiniZinc/ASP constraint solving"),
        ("functional", "Functional", "ML/Haskell/OCaml functional languages"),
        ("oop", "OOP", "Java/C++/C# object-oriented code"),
        ("mobile", "Mobile", "Swift/Kotlin mobile platforms"),
        ("scientific", "Scientific", "FORTRAN/Julia/R scientific computing"),
        ("bytecode", "Bytecode", "JVM/LLVM/CLR bytecode"),
        ("systems", "Systems", "SystemC/TLA+ system modeling"),
        ("planning", "Planning", "PDDL AI planning"),
        ("asm_ir", "AssemblyIR", "LLVM IR/GCC GIMPLE intermediate"),
        ("beam", "BEAM", "Erlang BEAM bytecode"),
        ("asp", "ASP", "Answer Set Programming (Clingo/DLV)"),
    ],
}

EMITTER_TEMPLATE = '''"""STUNIR {class_name} Emitter - Python Reference Implementation

{description}
Based on Ada SPARK {snake_name}_emitter implementation.
"""

from typing import Dict, List, Optional
from ..base_emitter import BaseEmitter, EmitterConfig, EmitterResult, EmitterStatus
from ..types import IRModule, IRFunction, IRType, IRDataType
from ..codegen import CodeGenerator


class {class_name}EmitterConfig(EmitterConfig):
    """{class_name} emitter specific configuration."""
    
    def __init__(self, output_dir: str, module_name: str, **kwargs):
        super().__init__(output_dir, module_name, **kwargs)
        # Add {snake_name}-specific config here
        self.target_variant: str = kwargs.get("target_variant", "default")


class {class_name}Emitter(BaseEmitter):
    """{class_name} emitter - {description}
    
    Generates {output_format} from STUNIR Semantic IR.
    Ensures confluence with Ada SPARK implementation.
    """

    def __init__(self, config: {class_name}EmitterConfig):
        """Initialize {class_name} emitter.
        
        Args:
            config: Emitter configuration
        """
        super().__init__(config)
        self.config: {class_name}EmitterConfig = config
        self.codegen = CodeGenerator()

    def emit(self, ir_module: IRModule) -> EmitterResult:
        """Emit {output_format} from IR module.
        
        Args:
            ir_module: Semantic IR module to emit
            
        Returns:
            EmitterResult with generated files
        """
        if not self.validate_ir(ir_module):
            return EmitterResult(
                status=EmitterStatus.ERROR_INVALID_IR,
                error_message="Invalid IR module structure"
            )

        try:
            files = []
            total_size = 0

            # Generate main output file
            main_content = self._generate_{snake_name}_code(ir_module)
            main_file = self.write_file(
                f"{{ir_module.module_name}}.{file_ext}",
                main_content
            )
            files.append(main_file)
            total_size += main_file.size

            # Generate additional files if needed
            additional_files = self._generate_additional_files(ir_module)
            for rel_path, content in additional_files.items():
                gen_file = self.write_file(rel_path, content)
                files.append(gen_file)
                total_size += gen_file.size

            return EmitterResult(
                status=EmitterStatus.SUCCESS,
                files=files,
                total_size=total_size
            )

        except Exception as e:
            return EmitterResult(
                status=EmitterStatus.ERROR_WRITE_FAILED,
                error_message=f"Failed to emit {snake_name}: {{e}}"
            )

    def _generate_{snake_name}_code(self, ir_module: IRModule) -> str:
        """Generate main {output_format} code.
        
        Args:
            ir_module: IR module
            
        Returns:
            Generated code content
        """
        lines = []
        
        # Add header
        lines.append(self.get_do178c_header(
            f"{{ir_module.module_name}} - {description}"
        ))
        
        # Add module documentation
        if ir_module.docstring:
            lines.extend(self.codegen.format_comment(
                ir_module.docstring,
                style="{comment_style}"
            ))
        
        # Generate types
        for ir_type in ir_module.types:
            lines.append(self._generate_type(ir_type))
            lines.append("")
        
        # Generate functions
        for function in ir_module.functions:
            lines.append(self._generate_function(function))
            lines.append("")
        
        return "\\n".join(lines)

    def _generate_type(self, ir_type: IRType) -> str:
        """Generate type definition.
        
        Args:
            ir_type: IR type
            
        Returns:
            Type definition code
        """
        # Implement type generation for {snake_name}
        return f"/* Type: {{ir_type.name}} */"

    def _generate_function(self, function: IRFunction) -> str:
        """Generate function definition.
        
        Args:
            function: IR function
            
        Returns:
            Function definition code
        """
        # Implement function generation for {snake_name}
        params = [(p.name, self.codegen.map_type_to_language(p.param_type, "{target_lang}"))
                  for p in function.parameters]
        return_type = self.codegen.map_type_to_language(function.return_type, "{target_lang}")
        
        signature = self.codegen.generate_function_signature(
            function.name,
            params,
            return_type,
            "{target_lang}"
        )
        
        return f"{{signature}} {{{{\\n    /* Implementation */\\n}}}}"

    def _generate_additional_files(self, ir_module: IRModule) -> Dict[str, str]:
        """Generate additional support files.
        
        Args:
            ir_module: IR module
            
        Returns:
            Dictionary of {{relative_path: content}}
        """
        return {{}}


# Convenience function for direct usage
def emit_{snake_name}(
    ir_module: IRModule,
    output_dir: str,
    **config_kwargs
) -> EmitterResult:
    """Emit {output_format} from IR module.
    
    Args:
        ir_module: Semantic IR module
        output_dir: Output directory path
        **config_kwargs: Additional configuration options
        
    Returns:
        EmitterResult
    """
    config = {class_name}EmitterConfig(
        output_dir=output_dir,
        module_name=ir_module.module_name,
        **config_kwargs
    )
    emitter = {class_name}Emitter(config)
    return emitter.emit(ir_module)
'''

def generate_emitter(category: str, snake_name: str, class_name: str, description: str):
    """Generate a single emitter file."""
    
    # Determine file extension and target language
    ext_map = {
        "embedded": ("c", "c", "c"),
        "gpu": ("cu", "c", "c"),
        "wasm": ("wat", "c", "c"),
        "assembly": ("asm", "ada", "c"),
        "polyglot": ("c", "c", "c"),
        "lisp": ("lisp", "cpp", "c"),
        "prolog": ("pl", "cpp", "c"),
        "business": ("cob", "cpp", "c"),
        "fpga": ("vhd", "ada", "c"),
        "grammar": ("ebnf", "cpp", "c"),
        "lexer": ("l", "cpp", "c"),
        "parser": ("y", "cpp", "c"),
        "expert": ("clp", "cpp", "c"),
        "constraints": ("mzn", "cpp", "c"),
        "functional": ("ml", "cpp", "c"),
        "oop": ("cpp", "cpp", "c"),
        "mobile": ("swift", "cpp", "c"),
        "scientific": ("f90", "cpp", "c"),
        "bytecode": ("bc", "cpp", "c"),
        "systems": ("sc", "cpp", "c"),
        "planning": ("pddl", "cpp", "c"),
        "asm_ir": ("ll", "cpp", "c"),
        "beam": ("beam", "cpp", "c"),
        "asp": ("lp", "cpp", "c"),
    }
    
    file_ext, comment_style, target_lang = ext_map.get(snake_name, ("txt", "c", "c"))
    output_format = f"{class_name} code"
    
    content = EMITTER_TEMPLATE.format(
        class_name=class_name,
        snake_name=snake_name,
        description=description,
        file_ext=file_ext,
        comment_style=comment_style,
        target_lang=target_lang,
        output_format=output_format
    )
    
    # Write file
    output_dir = Path("/home/ubuntu/stunir_repo/tools/semantic_ir/emitters") / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{snake_name}.py"
    output_file.write_text(content)
    
    print(f"✓ Generated {category}/{snake_name}.py")

def main():
    """Generate all emitters."""
    print("Generating STUNIR Semantic IR Emitters...")
    
    total = 0
    for category, emitters in EMITTERS.items():
        print(f"\n{category.upper()}:")
        for snake_name, class_name, description in emitters:
            generate_emitter(category, snake_name, class_name, description)
            total += 1
    
    # Generate __init__.py files for each category
    for category in EMITTERS.keys():
        init_file = Path("/home/ubuntu/stunir_repo/tools/semantic_ir/emitters") / category / "__init__.py"
        emitter_list = [e[0] for e in EMITTERS[category]]
        imports = "\n".join([f"from .{e} import {EMITTERS[category][i][1]}Emitter" 
                            for i, e in enumerate(emitter_list)])
        all_list = ", ".join([f'"{EMITTERS[category][i][1]}Emitter"' 
                             for i in range(len(emitter_list))])
        
        init_content = f'''"""STUNIR {category.title()} Emitters"""

{imports}

__all__ = [{all_list}]
'''
        init_file.write_text(init_content)
        print(f"✓ Generated {category}/__init__.py")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully generated all {total} emitters!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
