//! Micro benchmarks

use criterion::{criterion_group, criterion_main, Criterion};

fn round_floor(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("round_floor");

    group.bench_function("round", |b| {
        b.iter_batched(
            rand::random,
            |input: f32| input.round() as usize,
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("round_even", |b| {
        b.iter_batched(
            rand::random,
            |input: f32| input.round_ties_even() as usize,
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("floor", |b| {
        b.iter_batched(
            rand::random,
            |input: f32| input.floor() as usize,
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("ceil", |b| {
        b.iter_batched(
            rand::random,
            |input: f32| input.ceil() as usize,
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("manual round", |b| {
        b.iter_batched(
            rand::random,
            |input: f32| (input + 0.5).floor() as usize,
            criterion::BatchSize::SmallInput,
        )
    });
}

fn hash_speed(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("hash speed for usize");

    use std::hash::BuildHasher;

    group.bench_function("std::hash", |b| {
        b.iter_batched(
            rand::random,
            |input: usize| std::hash::RandomState::new().hash_one(input),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("ahash", |b| {
        b.iter_batched(
            rand::random,
            |input: usize| ahash::RandomState::new().hash_one(input),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("wy2hash", |b| {
        b.iter_batched(
            rand::random,
            |input: usize| wyhash2::WyHash::default().hash_one(input),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("xxhash", |b| {
        b.iter_batched(
            rand::random,
            |input: usize| xxhash_rust::xxh64::Xxh64Builder::default().hash_one(input),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("fxhash", |b| {
        b.iter_batched(
            rand::random,
            |input: usize| fxhash::FxBuildHasher::default().hash_one(input),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn dist(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("distribute");

    let size: usize = 123141489724;

    group.bench_function("mod", |b| {
        b.iter_batched(
            rand::random,
            |seed: u64| seed as usize % size >= 1 << size.ilog2(),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("fast dist", |b| {
        b.iter_batched(
            rand::random,
            |seed: u64| ((seed as u128 * size as u128) >> 64) as usize >= 1 << size.ilog2(),
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("and", |b| {
        b.iter_batched(
            rand::random,
            |seed: u64| (seed & 1) as usize,
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    round_floor,
    hash_speed,
    dist
);
criterion_main!(benches);
