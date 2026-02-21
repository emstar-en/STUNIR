//! Benchmarks for STUNIR Cryptographic Operations
//!
//! Run benchmarks with: `cargo bench`
//! View HTML report in: `target/criterion/report/index.html`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::fs::{self, File};
use std::io::Write;
use tempfile::TempDir;

/// Benchmark file hashing for various file sizes
fn bench_file_hashing(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    
    // Create test files of various sizes
    let sizes = vec![
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
    ];
    
    let mut group = c.benchmark_group("file_hashing");
    
    for (name, size) in sizes {
        let file_path = temp_dir.path().join(format!("test_{}.bin", name));
        let mut file = File::create(&file_path).unwrap();
        
        // Write random-ish data
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        file.write_all(&data).unwrap();
        
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &file_path, |b, path| {
            b.iter(|| {
                stunir_native::crypto::hash_file(black_box(path)).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark directory hashing
fn bench_directory_hashing(c: &mut Criterion) {
    let temp_dir = TempDir::new().unwrap();
    
    // Create directory structure with multiple files
    let file_counts = vec![
        ("10_files", 10),
        ("50_files", 50),
        ("100_files", 100),
    ];
    
    let mut group = c.benchmark_group("directory_hashing");
    
    for (name, count) in file_counts {
        let dir_path = temp_dir.path().join(name);
        fs::create_dir_all(&dir_path).unwrap();
        
        // Create files
        for i in 0..count {
            let file_path = dir_path.join(format!("file_{}.txt", i));
            fs::write(&file_path, format!("Content of file {}", i)).unwrap();
        }
        
        group.bench_with_input(BenchmarkId::from_parameter(name), &dir_path, |b, path| {
            b.iter(|| {
                stunir_native::crypto::hash_directory(black_box(path), 0).unwrap()
            })
        });
    }
    
    group.finish();
}

/// Benchmark raw SHA-256 hashing
fn bench_sha256_raw(c: &mut Criterion) {
    use sha2::{Sha256, Digest};
    
    let sizes = vec![
        ("64B", 64),
        ("256B", 256),
        ("1KB", 1024),
        ("4KB", 4096),
        ("64KB", 65536),
    ];
    
    let mut group = c.benchmark_group("sha256_raw");
    
    for (name, size) in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let mut hasher = Sha256::new();
                hasher.update(black_box(data));
                hex::encode(hasher.finalize())
            })
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_file_hashing, bench_directory_hashing, bench_sha256_raw);
criterion_main!(benches);
