//! Tests for specialized emitters

use stunir_emitters::base::{BaseEmitter, EmitterConfig, EmitterStatus};
use stunir_emitters::specialized::*;
use stunir_emitters::types::*;
use tempfile::TempDir;

macro_rules! test_specialized_emitter {
    ($name:ident, $config_type:ty, $emitter_type:ty, $variant:expr) => {
        #[test]
        fn $name() {
            let temp_dir = TempDir::new().unwrap();
            let base_config = EmitterConfig::new(temp_dir.path(), stringify!($name));
            let config = <$config_type>::new(base_config, $variant);
            let emitter = <$emitter_type>::new(config);

            let ir_module = create_test_module(stringify!($name));
            let result = emitter.emit(&ir_module).unwrap();

            assert_eq!(result.status, EmitterStatus::Success);
            assert!(result.files.len() >= 1);
        }
    };
}

test_specialized_emitter!(
    test_business,
    BusinessConfig,
    BusinessEmitter,
    BusinessVariant::COBOL
);
test_specialized_emitter!(test_fpga, FpgaConfig, FpgaEmitter, FpgaVariant::VHDL);
test_specialized_emitter!(
    test_grammar,
    GrammarConfig,
    GrammarEmitter,
    GrammarVariant::ANTLR
);
test_specialized_emitter!(test_lexer, LexerConfig, LexerEmitter, LexerVariant::Flex);
test_specialized_emitter!(
    test_parser,
    ParserConfig,
    ParserEmitter,
    ParserVariant::Yacc
);
test_specialized_emitter!(
    test_expert,
    ExpertConfig,
    ExpertEmitter,
    ExpertVariant::CLIPS
);
test_specialized_emitter!(
    test_constraints,
    ConstraintsConfig,
    ConstraintsEmitter,
    ConstraintsVariant::MiniZinc
);
test_specialized_emitter!(
    test_functional,
    FunctionalConfig,
    FunctionalEmitter,
    FunctionalVariant::Haskell
);
test_specialized_emitter!(test_oop, OopConfig, OopEmitter, OopVariant::Java);
test_specialized_emitter!(
    test_mobile,
    MobileConfig,
    MobileEmitter,
    MobileVariant::iOS_Swift
);
test_specialized_emitter!(
    test_scientific,
    ScientificConfig,
    ScientificEmitter,
    ScientificVariant::MATLAB
);
test_specialized_emitter!(
    test_bytecode,
    BytecodeConfig,
    BytecodeEmitter,
    BytecodeVariant::JVM
);
test_specialized_emitter!(
    test_systems,
    SystemsConfig,
    SystemsEmitter,
    SystemsVariant::Ada
);
test_specialized_emitter!(
    test_planning,
    PlanningConfig,
    PlanningEmitter,
    PlanningVariant::PDDL
);
test_specialized_emitter!(
    test_asm_ir,
    Asm_irConfig,
    Asm_irEmitter,
    Asm_irVariant::LLVM_IR
);
test_specialized_emitter!(test_beam, BeamConfig, BeamEmitter, BeamVariant::Erlang_BEAM);
test_specialized_emitter!(test_asp, AspConfig, AspEmitter, AspVariant::Clingo);

fn create_test_module(name: &str) -> IRModule {
    IRModule {
        ir_version: "1.0".to_string(),
        module_name: name.to_string(),
        types: vec![],
        functions: vec![IRFunction {
            name: "test_function".to_string(),
            return_type: IRDataType::I32,
            parameters: vec![],
            statements: vec![],
            docstring: Some("Test function".to_string()),
        }],
        docstring: None,
    }
}
