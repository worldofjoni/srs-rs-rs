use std::{
    cell::RefCell,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    num::NonZeroU32,
};

use bitvec::{field::BitField, order::Msb0, vec::BitVec};

use crate::RecHasher;

type Word = usize;
type Float = f64;
type DefaultHash = wyhash2::WyHash;

pub struct SrsMphf<T: Hash, H: BuildHasher + Clone = DefaultHash> {
    _phantom: PhantomData<T>,
    hasher: RecHasher<H>,
    /// includes root seed
    information: BitVec<Word, Msb0>,
    size: usize,
    overhead: Float,
    #[cfg(feature = "debug_output")]
    stats: (),
}

impl<T: Hash, H: BuildHasher + Clone> SrsMphf<T, H> {
    pub fn with_state(data: &[T], overhead: Float, state: H) -> Self {
        MphfBuilder::new(&mut data.iter().collect::<Vec<_>>(), overhead, state).build()
    }
}

impl<T: Hash> SrsMphf<T> {
    pub fn new(data: &[T], overhead: Float) -> Self {
        Self::with_state(data, overhead, DefaultHash::default())
    }

    pub fn hash(&self, value: &T) -> usize {
        let mut result = 0;
        let mut cum = CumulativeSigma::new(self.size, self.overhead);
        for i in 0..self.size.ilog2() {
            let j_1 = (1 << i) as usize + result;

            let start = cum.calc_to(j_1).ceil() as Word;
            // + Word::BITS for root - Word::BITS for start

            let seed = self.information[start..][..Word::BITS as usize].load_be();
            // println!("level {i}, result {result:b}, accessing seed at {start}");
            // println!("  seed {seed:b}");

            result <<= 1;
            let hash = self.hasher.hash_binary(seed, value);
            // println!("  gotten hash {hash}");
            result |= hash;
        }
        // println!("done {result}");

        result
    }

    pub fn bit_size(&self) -> usize {
        self.information.len() - self.information.leading_zeros()
    }

    pub fn bit_per_key(&self) -> f64 {
        self.bit_size() as f64 / self.size as f64
    }
}

/// determine how many bits would be used
pub fn determine_mvp_space_usage(num_elements: usize, overhead: Float) -> usize {
    sigma(num_elements, overhead, num_elements).ceil() as usize + Word::BITS as usize
}
/// determine how many bits per key would be used
pub fn determine_mvp_bits_per_key(num_elements: usize, overhead: Float) -> Float {
    (sigma(num_elements, overhead, num_elements).ceil() + Word::BITS as Float)
        / num_elements as Float
}

pub struct MphfBuilder<'a, T: Hash, H: BuildHasher + Clone> {
    data: &'a mut [&'a T],
    /// extra bits per task: overhead=log(1+eps)
    overhead: Float,
    hasher: RecHasher<H>,
    information: BitVec<Word, Msb0>,
    #[cfg(feature = "debug_output")]
    stats: (),
    #[cfg(feature = "progress")]
    progress_bar: indicatif::ProgressBar,
    #[cfg(feature = "progress")]
    saved_progress: u64,
}

#[derive(Copy, Clone)]
struct TaskState {
    parent_seed: Word, // todo avoidable?
    fractional_accounted_bits: Float,
    started: Option<Started>,
}

type Index = u64;
/// Information only available after started
#[derive(Copy, Clone)]
struct Started {
    task_bit_count: NonZeroU32,
    current_index: Index,
}

impl<'a, T: Hash, H: BuildHasher + Clone> MphfBuilder<'a, T, H> {
    fn new(data: &'a mut [&'a T], overhead: Float, random_state: H) -> Self {
        let size: usize = data.len();
        assert!(size.is_power_of_two());
        let max_bit_task = targeted_bits_on_layer(0, overhead, size);
        assert!(
            max_bit_task <= Index::BITS as f64,
            "{max_bit_task}<={} required for impl",
            Index::BITS
        );

        Self {
            overhead,
            data,
            hasher: RecHasher(random_state),
            information: BitVec::EMPTY,
            #[cfg(feature = "debug_output")]
            stats: (),
            #[cfg(feature = "progress")]
            progress_bar: indicatif::ProgressBar::new(size.ilog2() as u64),
            #[cfg(feature = "progress")]
            saved_progress: 0,
        }
    }

    pub fn build(mut self) -> SrsMphf<T, H> {
        let total_bits = Word::BITS as usize
            + sigma(self.data.len() - 1, self.overhead, self.data.len()).ceil() as usize;
        self.information.resize(total_bits, false);

        let mut stack = Vec::<TaskState>::with_capacity(self.data.len() - 1);
        for root_seed in 0.. {
            if self.srs_search_iterative(root_seed, &mut stack) {
                // #[cfg(feature = "debug_output")]
                // {
                //     println!(
                //         "bijections found in total {}/{num_buckets}, for each bucket: {:?}",
                //         self.stats.num_bijections_found.iter().sum::<usize>(),
                //         self.stats.num_bijections_found
                //     );
                //     println!("bij tested: {}", self.stats.bij_tests);
                // }

                self.information[..Word::BITS as usize].store_be(root_seed);

                #[cfg(feature = "debug_output")]
                println!(
                    "data: {} bit, {} per key, {} bit 'unused', {} per key therewith \n\t{}",
                    self.information.len(),
                    self.information.len() as f64 / self.data.len() as f64,
                    self.information.first_one().unwrap(),
                    (self.information.len() - self.information.first_one().unwrap()) as f64
                        / self.data.len() as f64,
                    self.information
                        .iter()
                        .map(|b| usize::from(*b).to_string())
                        .collect::<String>()
                );
                return SrsMphf {
                    size: self.data.len(),
                    _phantom: PhantomData,
                    hasher: self.hasher,
                    information: self.information,
                    overhead: self.overhead,
                    #[cfg(feature = "debug_output")]
                    stats: self.stats,
                };
            }
        }
        panic!("No MPHF found!")
    }

    fn srs_search_iterative(&mut self, root_seed: Word, stack: &mut Vec<TaskState>) -> bool {
        let num_tasks = self.data.len() - 1;
        stack.clear();

        stack.push(TaskState {
            parent_seed: root_seed,
            fractional_accounted_bits: 0.,
            started: None,
        });

        'main: while let Some(frame) = stack.last().copied() {
            let task_idx_1 = stack.len();

            let layer = task_idx_1.ilog2();
            let chunk = task_idx_1 - (1 << layer);
            let required_bits = targeted_bits_on_layer(layer, self.overhead, self.data.len());

            let required_bits = required_bits - frame.fractional_accounted_bits;
            let task_bit_count = required_bits.ceil() as usize; // log2 k
            let new_fractional_accounted_bits = required_bits.ceil() - required_bits;

            // println!("task {task_idx_1}, required {required_bits} bit, task bits {task_bit_count}, resuming after? {:?}",
            // frame.started.map(|s| s.current_index));

            #[cfg(feature = "progress")]
            {
                if self.saved_progress != layer as u64 {
                    self.progress_bar.set_position(layer as u64);
                    self.saved_progress = layer as u64;
                }
            }

            for seed in ((frame.parent_seed << task_bit_count)..)
                .take(1 << task_bit_count)
                .skip(
                    frame
                        .started
                        .map(|s| s.current_index as usize + 1)
                        .unwrap_or(0),
                )
            {
                let layer_size = self.data.len() >> layer;
                let data_slice = &mut self.data[chunk * layer_size..][..layer_size];

                if self.hasher.is_binary_split(seed, data_slice) {
                    data_slice.select_nth_unstable_by_key(layer_size / 2, |v| {
                        self.hasher.hash_binary(seed, v)
                    });

                    let index = seed - (frame.parent_seed << task_bit_count);
                    let last = stack.last_mut().expect("exists");
                    last.started = Some(Started {
                        current_index: index.try_into().expect("indices are small enough"),
                        task_bit_count: task_bit_count
                            .try_into()
                            .ok()
                            .and_then(NonZeroU32::new)
                            .expect("task bit count in expected range"),
                    });

                    // println!("  found at index {index}");

                    if task_idx_1 == num_tasks {
                        self.load_indices_from_stack(stack);
                        return true;
                    }

                    // recurse
                    stack.push(TaskState {
                        parent_seed: seed,
                        fractional_accounted_bits: new_fractional_accounted_bits,
                        started: None,
                    });
                    continue 'main;
                }
            }

            // backtrack
            stack.pop();
        }

        false
    }

    fn load_indices_from_stack(&mut self, stack: &[TaskState]) {
        assert_eq!(stack.len(), self.data.len() - 1);

        let mut working_slice = &mut self.information[Word::BITS as usize..];

        for TaskState { started, .. } in stack {
            let started = started.expect("all started");
            let task_bit_count = started.task_bit_count.get() as usize;
            working_slice[..task_bit_count].store_be(started.current_index);
            working_slice = &mut working_slice[task_bit_count..];
        }
    }
}

/// n: size where to search binary splitting
fn calc_log_p(n: usize) -> Float {
    assert!(n.is_power_of_two(), "n={n}");
    // todo stirling good enough? -> improve
    (1..=n / 2)
        .map(|i| (n as Float / (8. * i as Float)) + 0.25)
        .map(Float::log2)
        .sum()
}

fn get_log_p(n: usize) -> Float {
    assert!(n.is_power_of_two(), "n={n}");
    let power = n.ilog2() as usize;

    thread_local! {
        static LOG_P_LAYER: RefCell<Vec<Float>> = const {RefCell::new(Vec::new())};
    }

    LOG_P_LAYER.with_borrow_mut(|cache| {
        if power >= cache.len() {
            cache.reserve(power - cache.len() + 1);
            for i in cache.len()..=power {
                cache.push(calc_log_p(1 << i));
            }
        }
        cache[power]
    })
}

fn sigma(j: usize, overhead: Float, size: usize) -> Float {
    if j == 0 {
        return 0.;
    }

    let layer = j.ilog2();
    let chunk = j - (1 << layer); // starts with 0

    (0..layer)
        .map(|i| (1 << i) as Float * targeted_bits_on_layer(i, overhead, size))
        .sum::<Float>()
        + (chunk + 1) as Float * targeted_bits_on_layer(layer, overhead, size)
}

#[derive(Default)]
struct CumulativeSigma {
    size: usize,
    overhead: Float,
    index: usize,
    value: Float,
}

impl CumulativeSigma {
    fn new(size: usize, overhead: Float) -> Self {
        Self {
            size,
            overhead,
            index: 0,
            value: 0.,
        }
    }
    fn calc_to(&mut self, idx: usize) -> Float {
        debug_assert_eq!(
            idx.ilog2().checked_sub(1),
            self.index.checked_ilog2(),
            "idx needs to be on the next layer."
        );

        if let Some(old_layer) = self.index.checked_ilog2() {
            let old_chunk = self.index - (1 << old_layer); // starts with 0

            self.value += ((1 << old_layer) - old_chunk - 1) as Float
                * targeted_bits_on_layer(old_layer, self.overhead, self.size);
        }

        let new_layer = idx.ilog2();
        let new_chunk = idx - (1 << new_layer); // starts with 0

        self.value +=
            (new_chunk + 1) as Float * targeted_bits_on_layer(new_layer, self.overhead, self.size);

        self.index = idx;
        self.value
    }
}

fn targeted_bits_on_layer(layer: u32, overhead: Float, size: usize) -> Float {
    overhead * ((size >> layer) as Float).sqrt() - get_log_p(size >> layer)
}

#[cfg(test)]
mod test {

    use std::{collections::HashSet, hint::black_box, time};

    use float_cmp::{assert_approx_eq, F64Margin};
    use rand::distributions::{Alphanumeric, DistString};

    use crate::mphf::{
        calc_log_p, determine_mvp_bits_per_key, determine_mvp_space_usage, sigma, Float,
    };

    use super::{get_log_p, CumulativeSigma, SrsMphf};

    #[test]
    fn test_calc_log_p() {
        assert_approx_eq!(Float, 0., calc_log_p(1));
        assert_approx_eq!(Float, -1., calc_log_p(2));
        assert_approx_eq!(Float, -1.415037499278844, calc_log_p(4));
        assert_approx_eq!(Float, -1.8707169830550336, calc_log_p(8));
        assert_approx_eq!(Float, -2.348275566891936, calc_log_p(16));
    }

    #[test]
    fn test_get_log_p() {
        for i in [0, 2, 5, 4, 7, 9] {
            get_log_p(1 << i);
        }
    }

    #[test]
    fn test_sigma() {
        let size = 1 << 4;
        let overhead = 0.01;
        assert_approx_eq!(Float, 0., sigma(0, overhead, size));
        assert_approx_eq!(Float, 2.358275566891936, sigma(1, overhead, size));
        assert_approx_eq!(Float, 4.238992549946969, sigma(2, overhead, size));
        assert_approx_eq!(Float, 16., sigma(11, overhead, size).ceil());
    }

    #[test]
    fn test_sigma_large() {
        let size = 1 << 7;
        let overhead = 0.01;

        let mut extra_fractional = 0.;
        let mut bits_so_far = 0;
        for i in 1usize..size {
            println!("i={i}");
            let targeted_bits = overhead - calc_log_p(size >> i.ilog2());
            let targeted_bits = targeted_bits - extra_fractional;
            let task_bits = (targeted_bits).ceil() as usize;
            println!("  {task_bits} bit");
            extra_fractional = targeted_bits.ceil() - targeted_bits;
            bits_so_far += task_bits;

            assert_approx_eq!(
                Float,
                sigma(i, overhead, size),
                bits_so_far as Float - extra_fractional,
                F64Margin::zero().epsilon(1e-10)
            );
        }
    }

    #[test]
    fn test_sigma_cumulative() {
        let size = 1 << 10;
        let overhead = 0.001;
        let mut cum = CumulativeSigma::new(size, overhead);

        for j in [1, 2, 7, 8, 18] {
            println!("{j}");
            assert_approx_eq!(Float, sigma(j, overhead, size), cum.calc_to(j));
        }
    }

    #[test]
    fn test_bit_precalcs() {
        let size = 1 << 10;
        let overhead = 0.1;
        assert_eq!(
            SrsMphf::new(&(0..size).collect::<Vec<_>>(), overhead).bit_size(),
            determine_mvp_space_usage(size, overhead)
        );
        assert_approx_eq!(
            Float,
            SrsMphf::new(&(0..size).collect::<Vec<_>>(), overhead).bit_per_key(),
            determine_mvp_bits_per_key(size, overhead)
        );
    }

    #[test]
    fn test_create_mphf() {
        let size = 1 << 4;
        let overhead = 0.1;
        let data = (0..size).collect::<Vec<_>>();

        let mphf = SrsMphf::new(&data, overhead);
        println!(
            "done building! uses {} bits, {} per key",
            mphf.bit_size(),
            mphf.bit_per_key()
        );

        let hashes = (0..size).map(|v| mphf.hash(&v)).collect::<HashSet<_>>();
        assert_eq!(hashes.len(), size);
    }

    #[test]
    fn test_create_large_mphf() {
        let size = 1 << 15;
        let overhead = 0.001;
        let data = (0..size).collect::<Vec<_>>();

        let start = time::Instant::now();
        let mphf = SrsMphf::new(&data, overhead);
        let took = start.elapsed();

        println!(
            "done building in {:?} for size {size} uses {} bits, {} per key",
            took,
            mphf.bit_size(),
            mphf.bit_per_key()
        );
        println!(
            "data: {:?}",
            mphf.information
                .iter()
                .map(|b| usize::from(*b).to_string())
                .collect::<String>()
        );

        let hashes = (0..size).map(|v| mphf.hash(&v)).collect::<HashSet<_>>();
        assert_eq!(hashes.len(), size);
    }

    #[test]
    fn test_create_huge_mphf() {
        let size = 1 << 20;
        let overhead = 0.001; // attention: this is _smaller_ than for the other tests!
        let data = (0..size).collect::<Vec<_>>();

        let start = time::Instant::now();
        let mphf = SrsMphf::new(&data, overhead);
        let took = start.elapsed();
        println!(
            "done building in {:?}! uses {} bits, {} per key",
            took,
            mphf.bit_size(),
            mphf.bit_per_key()
        );

        let hashes = (0..size).map(|v| mphf.hash(&v)).collect::<HashSet<_>>();
        assert_eq!(hashes.len(), size);
    }

    #[test]
    fn test_create_huge_easy_mphf() {
        let size = 1 << 20;
        let overhead = 0.5; // attention: this is _larger_ than for the other tests!
        let data = (0..size).collect::<Vec<_>>();

        let mphf = SrsMphf::new(&data, overhead);
        println!(
            "done building! uses {} bits, {} per key",
            mphf.bit_size(),
            mphf.bit_per_key()
        );

        let hashes = (0..size).map(|v| mphf.hash(&v)).collect::<HashSet<_>>();
        assert_eq!(hashes.len(), size);
    }

    #[test]
    fn pareto() {
        const SIZE: usize = 1 << 20;
        println!("pareto size {SIZE}");

        let gen_input = || loop {
            let data = (0..SIZE)
                .map(|_| Alphanumeric.sample_string(&mut rand::thread_rng(), 16))
                .collect::<Vec<_>>();
            if data.iter().collect::<HashSet<_>>().len() == SIZE {
                return data;
            }
        };
        let input = black_box(gen_input());

        for overhead in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] {
            let start = time::Instant::now();
            let mphf = black_box(SrsMphf::new(&input, overhead));
            let took = start.elapsed();
            println!(
                "param {overhead}, bpk {}, took {took:?}",
                mphf.bit_per_key()
            );
        }
    }

    #[test]
    #[ignore]
    fn create_mphf_flame() {
        let size = 1 << 16;
        let overhead = 0.01;
        let data = (0..size).collect::<Vec<_>>();

        SrsMphf::new(&data, overhead);
    }
}
