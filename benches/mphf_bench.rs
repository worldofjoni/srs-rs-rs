use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use rand::random;
use recsplit::mphf::{determine_mvp_bits_per_key,  SrsMphf};

fn create_mphf_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("create single srs mphf (size = 1024)");

    const SIZE: usize = 1 << 10;
    const OVERHEAD: f64 = 0.01;

    group.warm_up_time(Duration::from_secs(10));

    for i in 0..2 {
        let data = &(0..SIZE).collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(i), &data, |b, input| {
            b.iter(|| {
                SrsMphf::new_random(input, OVERHEAD);
            })
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

    let data = &(0..SIZE).collect::<Vec<_>>();

    group.bench_with_input("create", &data, |b, input| {
        b.iter(|| {
            SrsMphf::new_random(input, OVERHEAD);
        })
    });

    group.finish();
}

fn create_many_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("crate multiple mphf with different sizes");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    const OVERHEAD: f64 = 0.01;

    for size in (4..16).map(|i| 1 << i) {
        let data = &(0..size).collect::<Vec<_>>();

        group.throughput(criterion::Throughput::Elements(size));
        group.bench_with_input(BenchmarkId::from_parameter(size), data, |b, input| {
            b.iter(|| {
                SrsMphf::new_random(input, OVERHEAD);
            })
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

    let size = 1 << 10;
    let data = &(0..size).collect::<Vec<_>>();

    for overhead in [2., 1., 0.5, 0.1, 0.01, 0.001, 0.0001] {
        group.bench_with_input(
            BenchmarkId::from_parameter(1. / overhead),
            data,
            |b, input| {
                b.iter(|| {
                    SrsMphf::new_random(input, overhead);
                })
            },
        );
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

    let size = 1 << 15;
    let data = &(0..size).map(|i| i.to_string()).collect::<Vec<_>>();

    group.throughput(criterion::Throughput::Elements(size as u64));
    for overhead in [1., 0.5, 0.2, 0.1, 0.01, 0.001] {
        group.bench_with_input(
            BenchmarkId::new(size.to_string(), determine_mvp_bits_per_key(size, overhead)),
            data,
            |b, input| {
                b.iter(|| {
                    SrsMphf::new_random(input, overhead);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    create_mphf_single,
    create_mphf_large,
    create_many_sizes,
    create_many_eps,
    hash,
    pareto,
);
criterion_main!(benches);
