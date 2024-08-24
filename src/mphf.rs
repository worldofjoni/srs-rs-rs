use std::{
    f64::consts::PI,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    num::NonZeroU32,
};

use bitvec::{field::BitField, order::Msb0, vec::BitVec};

use crate::RecHasher;

type Word = usize;
pub type Float = f64;
type DefaultHash = wyhash2::WyHash;

pub struct SrsMphf<T: Hash, H: BuildHasher + Clone = DefaultHash> {
    _phantom: PhantomData<T>,
    hasher: RecHasher<H>,
    /// includes root seed
    information: BitVec<Word, Msb0>,
    size: usize,
    overhead: Float,
    #[cfg(feature = "debug_output")]
    #[allow(unused)]
    stats: (),
}

impl<'a, T: Hash + 'a, H: BuildHasher + Clone> SrsMphf<T, H> {
    pub fn with_state(data: impl IntoIterator<Item = &'a T>, overhead: Float, state: H) -> Self {
        MphfBuilder::new(&mut data.into_iter().collect::<Vec<_>>(), overhead, state).build()
    }
}

impl<'a, T: Hash + 'a> SrsMphf<T> {
    pub fn new(data: impl IntoIterator<Item = &'a T>, overhead: Float) -> Self {
        Self::with_state(data, overhead, DefaultHash::default())
    }

    pub fn hash(&self, value: &T) -> usize {
        let mut result = 0;
        let mut cumulative_bits = 0.;

        let log_size = self.size.next_power_of_two().ilog2();

        for layer in (1..=log_size).rev() {
            let chunk = result;

            let ordinary_task_size = 1 << layer;
            let ordinary_bits_layer = targeted_bits_for_size(ordinary_task_size, self.overhead);

            let extra_task_size = self.size % ordinary_task_size;
            let has_special_task = extra_task_size > (1 << (layer - 1));

            let extra_task_bits = if has_special_task {
                targeted_bits_for_size(extra_task_size, self.overhead)
            } else {
                0.
            };
            let whole_layer_bits =
                (self.size >> layer) as Float * ordinary_bits_layer + extra_task_bits;

            let hash = if chunk < self.size >> layer {
                // is ordinary task
                let so_far = cumulative_bits + (chunk + 1) as Float * ordinary_bits_layer;
                let seed = self.get_seed(so_far);

                self.hasher.hash_binary(seed, value)
            } else if has_special_task {
                // is special task
                let seed = self.get_seed(cumulative_bits + whole_layer_bits);

                self.hasher.hash_ratio(seed, value, extra_task_size)
            } else {
                // no task in this layer
                0
            };

            // add entire layer worth for next iteration
            cumulative_bits += whole_layer_bits;

            result <<= 1;
            result |= hash;
        }

        result
    }

    fn get_seed(&self, bit_so_far: Float) -> usize {
        let start = bit_so_far.ceil() as Word;
        // + Word::BITS for root - Word::BITS for start

        self.information[start..][..Word::BITS as usize].load_be()
    }

    pub fn bit_size(&self) -> usize {
        self.information.len() - self.information.leading_zeros()
    }

    pub fn bit_per_key(&self) -> f64 {
        self.bit_size() as f64 / self.size as f64
    }
}

/// determine how many bits would be used
/// Includes starting zeros that can be avoided, actual size may be smaller!
pub fn determine_mvp_space_usage(num_elements: usize, overhead: Float) -> usize {
    total_bits_required(overhead, num_elements)
}
/// determine how many bits per key would be used
/// Includes starting zeros that can be avoided, actual size may be smaller!
pub fn determine_mvp_bits_per_key(num_elements: usize, overhead: Float) -> Float {
    total_bits_required(overhead, num_elements) as Float / num_elements as Float
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
        let max_bit_task = targeted_bits_for_size(size, overhead);
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
            progress_bar: indicatif::ProgressBar::new(size.next_power_of_two().ilog2() as u64),
            #[cfg(feature = "progress")]
            saved_progress: 0,
        }
    }

    pub fn build(mut self) -> SrsMphf<T, H> {
        let total_bits = total_bits_required(self.overhead, self.data.len());
        self.information.resize(total_bits, false);

        let mut stack = Vec::<TaskState>::with_capacity(self.data.len() - 1);
        for root_seed in 0.. {
            if self.srs_search_iterative(root_seed, &mut stack) {
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
        let size = self.data.len();
        let num_tasks = size - 1;
        stack.clear();

        stack.push(TaskState {
            parent_seed: root_seed,
            fractional_accounted_bits: 0.,
            started: None,
        });

        'main: while let Some(frame) = stack.last().copied() {
            let task_idx_1 = stack.len();

            let layer = self
                .data
                .len()
                .div_ceil(task_idx_1)
                .next_power_of_two()
                .ilog2();
            let chunk = task_idx_1 - (size >> layer) + (size.trailing_zeros() >= layer) as usize - 1;

            let ordinary_layer_size = 1 << layer;
            let is_ordinary = chunk < size >> layer;
            let task_size = if is_ordinary {
                ordinary_layer_size
            } else {
                size % ordinary_layer_size
            };
            let required_bits = targeted_bits_for_size(task_size, self.overhead);

            let required_bits = required_bits - frame.fractional_accounted_bits; // todo do not keep integer part separate go get same rounding errors as when hashing? or separate there too?
            let task_bit_count = required_bits.ceil() as usize; // log2 k
            let new_fractional_accounted_bits = required_bits.ceil() - required_bits;

            #[cfg(feature = "progress")]
            {
                if self.saved_progress != layer as u64 {
                    self.progress_bar.set_position(layer as u64);
                    self.saved_progress = layer as u64;
                }
            }

            for seed in ((frame.parent_seed << task_bit_count)..)
                .take(1 << task_bit_count)
                // skip already tested indices
                .skip(
                    frame
                        .started
                        .map(|s| s.current_index as usize + 1)
                        .unwrap_or(0),
                )
            {
                let data_slice = &mut self.data[chunk * ordinary_layer_size..][..task_size];

                if self.hasher.is_po2_split(seed, data_slice) {
                    data_slice.select_nth_unstable_by_key(ordinary_layer_size / 2, |v| {
                        self.hasher.hash_po2(seed, v, task_size)
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
#[allow(unused)]
fn calc_log_p(n: usize) -> Float {
    assert!(n.is_power_of_two(), "n={n}");
    // todo stirling good enough?
    (1..=n / 2)
        .map(|i| (n as Float / (8. * i as Float)) + 0.25)
        .map(Float::log2)
        .sum()
}

const L_P: [Float; 31] = [
    0.0,
    -1.0,
    -1.415037499278844,
    -1.8707169830550336,
    -2.348275566891936,
    -2.8370172874049393,
    -3.331383362996563,
    -3.828565799826217,
    -4.327156943029114,
    -4.826452505226261,
    -5.326100285149282,
    -5.825924174963724,
    -6.325836119852171,
    -6.825792092292873,
    -7.325770078523176,
    -7.825759071640692,
    -8.325753568168604,
    -8.825750816775335,
    -9.3257494420903,
    -9.82574874929842,
    -10.32574841436987,
    -10.825748257990249,
    -11.32574818174186,
    -11.825747955545237,
    -12.325747725324078,
    -12.825748940275274,
    -13.325752320771377,
    -13.825735618868379,
    -14.325762854934181,
    -14.825743217079987,
    -15.325718561245553,
];

#[inline(always)]
fn get_log_p_power(power: u32) -> Float {
    L_P[power as usize]
}

// fn sigma(j: usize, overhead: Float, size: usize) -> Float {
//     if j == 0 {
//         return 0.;
//     }

//     let layer = j.ilog2();
//     let chunk = j - (1 << layer); // starts with 0

//     (0..layer)
//         .map(|i| (1 << i) as Float * targeted_bits_on_layer(i, overhead, size.ilog2()))
//         .sum::<Float>()
//         + (chunk + 1) as Float * targeted_bits_on_layer(layer, overhead, size.ilog2())
// }

#[inline(always)]
fn targeted_bits_for_size(size: usize, overhead: Float) -> Float {
    if size.is_power_of_two() {
        overhead * (size as Float).sqrt() - get_log_p_power(size.ilog2())
    } else {
        overhead * (size as Float).sqrt() // todo correct overhead scaling?
         - get_log_p_uneven(size)
    }
}

// todo cache somehow? fast inverse square root??
// todo more exact stirling?
fn get_log_p_uneven(size: usize) -> Float {
    let r = (1 << size.ilog2()) as Float;
    let q = r / size as Float;

    -(2. * PI * r * (1. - q)).log2() / 2.
}

fn total_bits_required(overhead: Float, size: usize) -> usize {
    (1..=size.next_power_of_two().ilog2())
        .rev()
        .map(|i| {
            (size >> i) as Float * targeted_bits_for_size(1 << i, overhead)
                + if size % (1 << i) > (1 << (i - 1)) {
                    targeted_bits_for_size(size % (1 << i), overhead)
                } else {
                    0.
                }
        })
        .sum::<Float>()
        .ceil() as usize
        + Word::BITS as usize
}

#[cfg(test)]
mod test {

    use std::{collections::HashSet, hint::black_box, time};

    use float_cmp::{assert_approx_eq, F64Margin};
    use rand::distributions::{Alphanumeric, DistString};

    use crate::mphf::{
        calc_log_p, determine_mvp_bits_per_key, determine_mvp_space_usage, get_log_p_power, Float,
    };

    use super::SrsMphf;

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
            get_log_p_power(i);
        }
        let size = 40;
        let vals = (0..=size).map(|i| calc_log_p(1 << i)).collect::<Vec<_>>();
        let vals2 = (0..=size).map(get_log_p_power).collect::<Vec<_>>();
        assert_eq!(vals, vals2);
    }

    // #[test]
    // fn test_sigma() {
    //     let size = 1 << 4;
    //     let overhead = 0.01;
    //     assert_approx_eq!(Float, 0., sigma(0, overhead, size));
    //     assert_approx_eq!(Float, 2.358275566891936, sigma(1, overhead, size));
    //     assert_approx_eq!(Float, 4.238992549946969, sigma(2, overhead, size));
    //     assert_approx_eq!(Float, 16., sigma(11, overhead, size).ceil());
    // }

    // #[test]
    // fn test_sigma_large() {
    //     let size = 1 << 7;
    //     let overhead = 0.01;

    //     let mut extra_fractional = 0.;
    //     let mut bits_so_far = 0;
    //     for i in 1usize..size {
    //         println!("i={i}");
    //         let targeted_bits = overhead - calc_log_p(size >> i.ilog2());
    //         let targeted_bits = targeted_bits - extra_fractional;
    //         let task_bits = (targeted_bits).ceil() as usize;
    //         println!("  {task_bits} bit");
    //         extra_fractional = targeted_bits.ceil() - targeted_bits;
    //         bits_so_far += task_bits;

    //         assert_approx_eq!(
    //             Float,
    //             sigma(i, overhead, size),
    //             bits_so_far as Float - extra_fractional,
    //             F64Margin::zero().epsilon(1e-10)
    //         );
    //     }
    // }

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
        let data: Vec<usize> = (0..size).collect::<Vec<_>>();

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
        let overhead = 0.001;
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
    fn test_create_npo2_mphf() {
        let size = 10000;
        let overhead = 0.001;
        let data: Vec<usize> = (0..size).collect::<Vec<_>>();

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

    #[test]
    #[ignore]
    fn create_mphf_hash_flame() {
        let size = 1 << 16;
        let overhead = 0.01;
        let data = (0..size).collect::<Vec<_>>();

        let mphf = SrsMphf::new(&data, overhead);

        for _ in 0..100 {
            for i in 0..size {
                mphf.hash(&i);
            }
        }
    }
}
