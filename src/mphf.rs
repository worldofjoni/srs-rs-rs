use std::{
    hash::{BuildHasher, Hash},
    marker::PhantomData,
};

use bitvec::{bitvec, field::BitField, order::Msb0, vec::BitVec};

use crate::RecHasher;

type Word = usize;
type Float = f32;

pub struct SrsMphf<T: Hash, H: BuildHasher + Clone> {
    _phantom: PhantomData<T>,
    hasher: RecHasher<H>,
    /// includes root seed
    full_information: BitVec<Word, Msb0>,
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

impl<T: Hash> SrsMphf<T, ahash::RandomState> {
    pub fn new_random(data: &[T], overhead: Float) -> Self {
        Self::with_state(data, overhead, ahash::RandomState::new())
    }

    pub fn hash(&self, value: &T) -> usize {
        let mut result = 0;
        for i in 0..self.size.ilog2() {
            let j = (1 << i) as usize + result;
            let start = sigma(j, self.overhead, self.size).ceil() as Word; // + Word::BITS for root seed - Word::BITS for start
            let seed = self.full_information[start..][..Word::BITS as usize].load_be();
            result <<= 1;
            result |= self.hasher.hash_binary(seed, value);
        }

        result
    }

    pub fn bit_size(&self) -> usize {
        self.full_information.len()
    }
}

pub struct MphfBuilder<'a, T: Hash, H: BuildHasher + Clone> {
    data: &'a mut [&'a T],
    /// extra bits per task: overhead=log(1+eps)
    overhead: Float,
    hasher: RecHasher<H>,
    full_information: BitVec<Word, Msb0>,
    #[cfg(feature = "debug_output")]
    stats: (),
}

impl<'a, T: Hash, H: BuildHasher + Clone> MphfBuilder<'a, T, H> {
    fn new(data: &'a mut [&'a T], overhead: Float, random_state: H) -> Self {
        assert!(data.len().is_power_of_two());
        Self {
            overhead,
            data,
            hasher: RecHasher(random_state),
            full_information: bitvec!(Word, Msb0; 0 ; 0),
            #[cfg(feature = "debug_output")]
            stats: (),
        }
    }

    pub fn build(mut self) -> SrsMphf<T, H> {
        let total_bits = Word::BITS as usize
            + sigma(self.data.len() - 1, self.overhead, self.data.len()).ceil() as usize;
        println!("bits reserved {total_bits}");
        self.full_information.resize(total_bits, false);

        for root_seed in 0.. {
            if self.find_seed_task(1, root_seed, Word::BITS as usize, 0.) {
                // #[cfg(feature = "debug_output")]
                // {
                //     println!(
                //         "bijections found in total {}/{num_buckets}, for each bucket: {:?}",
                //         self.stats.num_bijections_found.iter().sum::<usize>(),
                //         self.stats.num_bijections_found
                //     );
                //     println!("bij tested: {}", self.stats.bij_tests);
                // }

                self.full_information[..Word::BITS as usize].store_be(root_seed);

                return SrsMphf {
                    size: self.data.len(),
                    _phantom: PhantomData,
                    hasher: self.hasher,
                    full_information: self.full_information,
                    overhead: self.overhead,
                    #[cfg(feature = "debug_output")]
                    stats: self.stats,
                };
            }
        }
        panic!("No MPHF found!")
    }

    pub fn find_seed_task(
        &mut self,
        task_idx_1: usize, // one-based
        parent_seed: Word,
        bit_index: usize,
        current_overhead: Float,
    ) -> bool {
        println!("task {task_idx_1}, bit index {bit_index}, current overhead {current_overhead}");
        if task_idx_1 == self.data.len() {
            // there are only n-1 nodes in a binary splitting tree
            return true;
        }

        let layer = task_idx_1.ilog2();
        let required_bits = self.overhead - calc_log_p(self.data.len() >> layer);
        println!("  required bits {required_bits}");

        let new_required_bits = required_bits - current_overhead;
        let real_task_bits = new_required_bits.ceil() as usize;
        let new_overhead = new_required_bits.ceil() - new_required_bits;

        // implicit phi
        for seed in ((parent_seed << real_task_bits)..).take(1 << real_task_bits) {
            let data_slice = &mut self.data[1 << layer..][..1 << layer];
            // #[cfg(feature = "debug_output")]
            // {
            //     self.stats.bij_tests += 1;
            // }
            #[allow(clippy::collapsible_if)]
            if self
                .hasher
                .is_binary_split(seed, &data_slice.iter().by_ref().collect::<Vec<_>>())
            // todo does this collect create overhead/allocation?
            {
                data_slice.select_nth_unstable_by_key(data_slice.len() / 2, |v| {
                    self.hasher.hash_binary(seed, v)
                });
                // todo sort only after complete layer?

                // #[cfg(feature = "debug_output")]
                // {
                //     self.stats.num_bijections_found[bucket_idx_1 - 1] += 1;
                // }

                if self.find_seed_task(
                    task_idx_1 + 1,
                    seed,
                    bit_index + real_task_bits,
                    new_overhead,
                ) {
                    let index = seed & ((1 << real_task_bits) - 1);
                    self.full_information[bit_index..][..real_task_bits].store_be(index);
                    return true;
                }
            }
        }
        // #[cfg(feature = "debug_output")]
        // {
        //     self.stats.unsuccessful_ret += 1;
        // }

        false
    }

    // fn build_layer(layer: usize)
}

/// n: size where to search binary splitting
// todo cache
fn calc_log_p(n: usize) -> Float {
    assert!(n.is_power_of_two(), "n={n}");
    // todo stirling good enough?
    (1..=n / 2)
        .map(|i| (n as Float / (8. * i as Float)) + 0.25)
        .map(Float::log2)
        .sum()
}

// todo optimize
fn sigma(j: usize, overhead: Float, size: usize) -> Float {
    j as Float * overhead - (1..=j)
        .map(|j| calc_log_p(size >> j.ilog2()))
        .sum::<Float>()
}

#[cfg(test)]
mod test {

    use std::collections::HashSet;

    use float_cmp::assert_approx_eq;

    use crate::mphf::{calc_log_p, sigma, Float};

    use super::SrsMphf;

    #[test]
    fn test_calc_log_p() {
        assert_approx_eq!(Float, 0., calc_log_p(1));
        assert_approx_eq!(Float, -1., calc_log_p(2));
        assert_approx_eq!(Float, -1.4150375, calc_log_p(4));
        assert_approx_eq!(Float, -1.8707169, calc_log_p(8));
        assert_approx_eq!(Float, -2.3482757, calc_log_p(16));
    }

    #[test]
    fn test_sigma() {
        let size = 1 << 4;
        let overhead = 0.01;
        assert_approx_eq!(Float, 0., sigma(0, overhead, size));
        assert_approx_eq!(Float, 2.3582757, sigma(1, overhead, size));
        assert_approx_eq!(Float, 4.2389927, sigma(2, overhead, size));
    }

    #[test]
    fn test_create_mphf() {
        let size = 1 << 4;
        let overhead = 0.01;
        let data = (0..size).collect::<Vec<_>>();

        let mphf = SrsMphf::new_random(&data, overhead);
        println!("done building! uses {} bits", mphf.bit_size());

        let hashes = (0..size).map(|v| mphf.hash(&v)).collect::<HashSet<_>>();
        assert_eq!(hashes.len(), size);
    }
}
