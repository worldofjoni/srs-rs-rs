use std::time::Duration;

use ahash::HashSet;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use rand::{
    distributions::{Alphanumeric, DistString},
    random,
};
use recsplit::mphf::{determine_mvp_bits_per_key, SrsMphf};

fn create_mphf_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("create single srs mphf (size = 1024)");

    const SIZE: usize = 1 << 10;
    const OVERHEAD: f64 = 0.01;

    group.warm_up_time(Duration::from_secs(10));

    for i in 0..2 {
        group.bench_function(BenchmarkId::from_parameter(i), |b| {
            b.iter_batched(
                || gen_input(SIZE),
                |input| {
                    SrsMphf::new_random(input.as_slice(), OVERHEAD);
                },
                criterion::BatchSize::LargeInput,
            )
        });

        group.warm_up_time(Duration::from_secs(1));
    }

    group.finish();
}

fn create_mphf_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("create single large srs mphf (size = 65536)");

    const SIZE: usize = 1 << 14;
    const OVERHEAD: f64 = 0.01;

    group.warm_up_time(Duration::from_secs(10));
    group.sample_size(100);

    group.bench_function("create", |b| {
        b.iter_batched(
            || gen_input(SIZE),
            |input| {
                SrsMphf::new_random(input.as_slice(), OVERHEAD);
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn create_many_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("crate multiple mphf with different sizes");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    const OVERHEAD: f64 = 0.01;
    const MAX: usize = 16;

    for size in (4..MAX).map(|i| 1 << i) {
        group.throughput(criterion::Throughput::Elements(size));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_batched(
                || gen_input(size as usize),
                |input| {
                    SrsMphf::new_random(&input, OVERHEAD);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

fn create_many_eps(c: &mut Criterion) {
    let mut group = c.benchmark_group("crate multiple mphfs with different overheads");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    const SIZE: usize = 1 << 10;

    for overhead in [2., 1., 0.5, 0.1, 0.01, 0.001, 0.0001] {
        group.bench_function(BenchmarkId::from_parameter(1. / overhead), |b| {
            b.iter_batched(
                || gen_input(SIZE),
                |input| {
                    SrsMphf::new_random(input.as_slice(), overhead);
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

fn hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash srs mphf");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let size = 1 << 12;
    let overhead = 0.01;
    let data = &(0..size).collect::<Vec<_>>();
    let mphf = SrsMphf::new_random(data, overhead);

    group.bench_with_input("hash", &mphf, |b, mphf| {
        b.iter_batched(random, |i| mphf.hash(&i), criterion::BatchSize::SmallInput)
    });
}

fn pareto(c: &mut Criterion) {
    let mut group = c.benchmark_group("mphf pareto");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    const SIZE: usize = 1 << 15;

    let gen_input = || loop {
        let data = (0..SIZE)
            .map(|_| Alphanumeric.sample_string(&mut rand::thread_rng(), 16))
            .collect::<Vec<_>>();
        if data.iter().collect::<HashSet<_>>().len() == SIZE {
            return data;
        }
    };

    group.throughput(criterion::Throughput::Elements(SIZE as u64));
    for overhead in [1., 0.5, 0.2, 0.1, 0.01, 0.001] {
        group.bench_function(
            BenchmarkId::new(SIZE.to_string(), determine_mvp_bits_per_key(SIZE, overhead)),
            |b| {
                b.iter_batched(
                    gen_input,
                    |input| {
                        SrsMphf::new_random(&input, overhead);
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

fn different_hashers(c: &mut Criterion) {
    let mut group = c.benchmark_group("different_hashers");

    const SIZE: usize = 1 << 10;
    let overhead = 0.1;

    let random = random();
    // group.sample_size(10);

    group.bench_function("std::hash", |b| {
        b.iter_batched(
            || gen_input(SIZE),
            |input| {
                SrsMphf::with_state(
                    input.as_slice(),
                    overhead,
                    std::hash::RandomState::default(),
                );
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.bench_function("ahash", |b| {
        b.iter_batched(
            || gen_input(SIZE),
            |input| {
                SrsMphf::with_state(
                    input.as_slice(),
                    overhead,
                    ahash::RandomState::with_seeds(random, random, random, random),
                );
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.bench_function("wy2hash", |b| {
        b.iter_batched(
            || gen_input(SIZE),
            |input| {
                SrsMphf::with_state(input.as_slice(), overhead, wyhash2::WyHash::default());
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.bench_function("xxhash", |b| {
        b.iter_batched(
            || gen_input(SIZE),
            |input| {
                SrsMphf::with_state(
                    input.as_slice(),
                    overhead,
                    xxhash_rust::xxh64::Xxh64Builder::default(),
                );
            },
            criterion::BatchSize::LargeInput,
        )
    });

    // does not terminate, too bad quality?
    // group.bench_function("fxhash", |b| {
    //     b.iter_batched(
    //         gen_input,
    //         |input| {
    //             SrsMphf::with_state(&input, overhead, fxhash::FxBuildHasher::default());
    //         },
    //         criterion::BatchSize::LargeInput,
    //     )
    // });
}

fn gen_input(size: usize) -> Vec<[usize; 8]> {
    loop {
        let mut data = vec![[0; 8]; size];
        data.iter_mut().for_each(|s| *s = random());
        if data.iter().collect::<HashSet<_>>().len() == size {
            return data;
        }
    }
}

criterion_group!(
    benches,
    create_mphf_single,
    create_mphf_large,
    create_many_sizes,
    create_many_eps,
    hash,
    different_hashers,
    pareto,
);
criterion_main!(benches);
