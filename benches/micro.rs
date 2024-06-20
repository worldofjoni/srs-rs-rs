//! Micro benchmarks

use std::cell::Cell;

use ahash::HashSet;
use bitvec::vec::BitVec;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use recsplit::{mvp::MvpBuilder, LooseRecSplit, RecHasher};

fn check_hash_function(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("check_hash_function");

    let hasher = RecHasher(ahash::RandomState::new());

    let input = [
        123123_usize,
        5634562345,
        574534562345,
        1242345,
        12347234,
        7834538,
        2353571245,
        865453262436,
        12447612354,
        34735378,
        156475612,
        2463727,
    ];

    group.bench_with_input("is_bijection", &input, |b, input| {
        b.iter_batched(
            rand::random,
            |seed| {
                hasher.is_bijection(seed, input);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_with_input("is_split_1", &input, |b, input| {
        b.iter_batched(
            rand::random,
            |seed| {
                hasher.is_split(seed, 1, input);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn find_hash_function(c: &mut Criterion) {
    let mut group: criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> =
        c.benchmark_group("find_hash_function");

    let hasher = RecHasher(ahash::RandomState::new());

    group.bench_function("find_split_seed", |b| {
        b.iter_batched(
            || loop {
                let input: [usize; 12] = rand::random();
                if input.iter().collect::<HashSet<_>>().len() == 12 {
                    return input;
                }
            },
            |input| {
                hasher.find_split_seed(1, &input);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // let input = [
    //     123123_usize,
    //     5634562345,
    //     574534562345,
    //     1242345,
    //     12347234,
    //     7834538,
    //     2353571245,
    //     865453262436,
    //     12447612354,
    //     34735378,
    //     156475612,
    //     2463727,
    // ];

    // let input = [input.as_slice()];
    // let mut slice = BitVec::new();

    // group.bench_function("is_split_1", |b| {
    //     b.iter_batched(
    //         || {
    //             let mut mvp = MvpBuilder::new(&input, &hasher, &mut slice);
    //             (mvp, rand::random())
    //         },
    //         |(mvp, seed)| {
    //             mvp.find_seed_bucket(1, seed);
    //         },
    //         criterion::BatchSize::SmallInput,
    //     )
    // });
}

criterion_group!(benches, check_hash_function, find_hash_function);
criterion_main!(benches);
