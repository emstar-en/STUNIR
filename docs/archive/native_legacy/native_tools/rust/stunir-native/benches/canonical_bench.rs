//! Benchmarks for STUNIR JSON Canonicalization
//!
//! Run benchmarks with: `cargo bench`
//! View HTML report in: `target/criterion/report/index.html`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

/// Generate test JSON objects of various complexities
fn generate_json(depth: usize, keys_per_level: usize) -> String {
    fn build_object(depth: usize, keys: usize, prefix: &str) -> String {
        if depth == 0 {
            return format!(r#""value_{}""#, prefix);
        }
        
        let entries: Vec<String> = (0..keys)
            .map(|i| {
                let key = format!("key_{}_{}", prefix, i);
                let value = build_object(depth - 1, keys, &format!("{}_{}", prefix, i));
                format!(r#""{}":{}"#, key, value)
            })
            .collect();
        
        format!(r#"{{{}}}"#, entries.join(","))
    }
    
    build_object(depth, keys_per_level, "root")
}

/// Benchmark JSON normalization
fn bench_normalization(c: &mut Criterion) {
    let test_cases = vec![
        ("small_flat", r#"{"z":1,"a":2,"m":3}"#.to_string()),
        ("medium_flat", (0..20).map(|i| format!(r#""key_{}":{}"#, i, i)).collect::<Vec<_>>().join(",").pipe(|s| format!("{{{}}}" , s))),
        ("nested_2", generate_json(2, 3)),
        ("nested_3", generate_json(3, 3)),
        ("nested_4", generate_json(4, 2)),
    ];
    
    let mut group = c.benchmark_group("json_normalization");
    
    for (name, json) in test_cases {
        group.throughput(Throughput::Bytes(json.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &json, |b, json| {
            b.iter(|| {
                stunir_native::canonical::normalize(black_box(json)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark normalize_and_hash
fn bench_normalize_and_hash(c: &mut Criterion) {
    let test_cases = vec![
        ("simple", r#"{"key":"value"}"#),
        ("numbers", r#"{"a":1,"b":2,"c":3,"d":4,"e":5}"#),
        ("nested", r#"{"outer":{"inner":{"deep":"value"}}}"#),
    ];
    
    let mut group = c.benchmark_group("normalize_and_hash");
    
    for (name, json) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &json, |b, json| {
            b.iter(|| {
                stunir_native::canonical::normalize_and_hash(black_box(json)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark json_equal comparisons
fn bench_json_equal(c: &mut Criterion) {
    let pairs = vec![
        ("equal_simple", r#"{"a":1}"#, r#"{"a":1}"#),
        ("equal_reordered", r#"{"z":1,"a":2}"#, r#"{"a":2,"z":1}"#),
        ("different", r#"{"a":1}"#, r#"{"a":2}"#),
    ];
    
    let mut group = c.benchmark_group("json_equal");
    
    for (name, a, b) in pairs {
        group.bench_with_input(BenchmarkId::from_parameter(name), &(a, b), |bench, (a, b)| {
            bench.iter(|| {
                stunir_native::canonical::json_equal(black_box(a), black_box(b)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark serialization round-trip
fn bench_serialization_roundtrip(c: &mut Criterion) {
    use stunir_native::ir_v1::{IrV1, IrFunction, IrInstruction};
    
    // Create IR of various sizes
    let create_ir = |func_count: usize, inst_count: usize| -> IrV1 {
        let mut ir = IrV1::new("benchmark_module");
        for f in 0..func_count {
            ir.add_function(IrFunction {
                name: format!("func_{}", f),
                body: (0..inst_count)
                    .map(|i| IrInstruction {
                        op: "call".to_string(),
                        args: vec![format!("arg_{}", i)],
                    })
                    .collect(),
            });
        }
        ir
    };
    
    let test_cases = vec![
        ("small_ir", create_ir(5, 10)),
        ("medium_ir", create_ir(20, 50)),
        ("large_ir", create_ir(50, 100)),
    ];
    
    let mut group = c.benchmark_group("serialization_roundtrip");
    
    for (name, ir) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &ir, |b, ir| {
            b.iter(|| {
                let json = serde_json::to_string(black_box(ir)).unwrap();
                let _: IrV1 = serde_json::from_str(&json).unwrap();
            })
        });
    }
    
    group.finish();
}

/// Extension trait for pipe syntax
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

impl<T> Pipe for T {}

criterion_group!(benches, bench_normalization, bench_normalize_and_hash, bench_json_equal, bench_serialization_roundtrip);
criterion_main!(benches);
