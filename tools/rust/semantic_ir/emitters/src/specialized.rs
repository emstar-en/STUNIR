//! STUNIR Specialized Emitters - Rust Implementation
//!
//! Specialized emitters for domain-specific languages and platforms.

pub mod asm_ir;
pub mod asp;
pub mod beam;
pub mod business;
pub mod bytecode;
pub mod constraints;
pub mod expert;
pub mod fpga;
pub mod functional;
pub mod grammar;
pub mod lexer;
pub mod mobile;
pub mod oop;
pub mod parser;
pub mod planning;
pub mod scientific;
pub mod systems;

// Re-export emitter types
pub use asm_ir::{Asm_irConfig, Asm_irEmitter, Asm_irVariant};
pub use asp::{AspConfig, AspEmitter, AspVariant};
pub use beam::{BeamConfig, BeamEmitter, BeamVariant};
pub use business::{BusinessConfig, BusinessEmitter, BusinessVariant};
pub use bytecode::{BytecodeConfig, BytecodeEmitter, BytecodeVariant};
pub use constraints::{ConstraintsConfig, ConstraintsEmitter, ConstraintsVariant};
pub use expert::{ExpertConfig, ExpertEmitter, ExpertVariant};
pub use fpga::{FpgaConfig, FpgaEmitter, FpgaVariant};
pub use functional::{FunctionalConfig, FunctionalEmitter, FunctionalVariant};
pub use grammar::{GrammarConfig, GrammarEmitter, GrammarVariant};
pub use lexer::{LexerConfig, LexerEmitter, LexerVariant};
pub use mobile::{MobileConfig, MobileEmitter, MobileVariant};
pub use oop::{OopConfig, OopEmitter, OopVariant};
pub use parser::{ParserConfig, ParserEmitter, ParserVariant};
pub use planning::{PlanningConfig, PlanningEmitter, PlanningVariant};
pub use scientific::{ScientificConfig, ScientificEmitter, ScientificVariant};
pub use systems::{SystemsConfig, SystemsEmitter, SystemsVariant};
