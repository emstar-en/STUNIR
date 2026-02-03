//! Emitter benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stunir_emitters::base::{BaseEmitter, EmitterConfig};
use stunir_emitters::core::*;
use stunir_emitters::types::*;
use tempfile::TempDir;

fn benchmark_embedded_emitter(c: &mut Criterion) {
    c.bench_function("embedded_arm", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let base_config = EmitterConfig::new(temp_dir.path(), "bench_module");
            let config = EmbeddedConfig::new(base_config, Architecture::ARM);
            let emitter = EmbeddedEmitter::new(config);

            let ir_module = create_benchmark_module();
            black_box(emitter.emit(&ir_module).unwrap());
        });
    });
}

fn benchmark_gpu_emitter(c: &mut Criterion) {
    c.bench_function("gpu_cuda", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let base_config = EmitterConfig::new(temp_dir.path(), "bench_gpu");
            let config = GPUConfig::new(base_config, GPUPlatform::CUDA);
            let emitter = GPUEmitter::new(config);

            let ir_module = create_benchmark_module();
            black_box(emitter.emit(&ir_module).unwrap());
        });
    });
}

fn benchmark_wasm_emitter(c: &mut Criterion) {
    c.bench_function("wasm_core", |b| {
        b.iter(|| {
            let temp_dir = TempDir::new().unwrap();
            let base_config = EmitterConfig::new(temp_dir.path(), "bench_wasm");
            let config = WasmConfig::new(base_config, WasmTarget::Core);
            let emitter = WasmEmitter::new(config);

            let ir_module = create_benchmark_module();
            black_box(emitter.emit(&ir_module).unwrap());
        });
    });
}

fn create_benchmark_module() -> IRModule {
    IRModule {
        ir_version: "1.0".to_string(),
        module_name: "benchmark".to_string(),
        types: vec![],
        functions: vec![
            IRFunction {
                name: "add".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![
                    IRParameter {
                        name: "a".to_string(),
                        param_type: IRDataType::I32,
                    },
                    IRParameter {
                        name: "b".to_string(),
                        param_type: IRDataType::I32,
                    },
                ],
                statements: vec![IRStatement {
                    stmt_type: IRStatementType::Add,
                    data_type: Some(IRDataType::I32),
                    target: Some("result".to_string()),
                    value: None,
                    left_op: Some("a".to_string()),
                    right_op: Some("b".to_string()),
                }],
                docstring: Some("Add two numbers".to_string()),
            },
            IRFunction {
                name: "multiply".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![
                    IRParameter {
                        name: "x".to_string(),
                        param_type: IRDataType::I32,
                    },
                    IRParameter {
                        name: "y".to_string(),
                        param_type: IRDataType::I32,
                    },
                ],
                statements: vec![IRStatement {
                    stmt_type: IRStatementType::Mul,
                    data_type: Some(IRDataType::I32),
                    target: Some("result".to_string()),
                    value: None,
                    left_op: Some("x".to_string()),
                    right_op: Some("y".to_string()),
                }],
                docstring: Some("Multiply two numbers".to_string()),
            },
        ],
        docstring: Some("Benchmark module".to_string()),
    }
}

criterion_group!(
    benches,
    benchmark_embedded_emitter,
    benchmark_gpu_emitter,
    benchmark_wasm_emitter
);
criterion_main!(benches);
