use std::{
    borrow::BorrowMut,
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
        todo!()
    }
}

pub struct MphfBuilder<'a, T: Hash, H: BuildHasher + Clone> {
    data: &'a mut [&'a T],
    /// extra bits per task: overhead=log(1+eps)
    overhead: f32,
    hasher: RecHasher<H>,
    full_information: BitVec<Word, Msb0>,
    #[cfg(feature = "debug_output")]
    stats: (),
}

impl<'a, T: Hash, H: BuildHasher + Clone> MphfBuilder<'a, T, H> {
    fn new(data: &'a mut [&'a T], overhead: Float, random_state: H) -> Self {
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
        let total_bits = todo!();

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
        bucket_bits_overhead: Float,
    ) -> bool {
        if task_idx_1 == self.data.len() {
            // there are only n-1 tasks
            return true;
        }

        let layer = task_idx_1.ilog2();
        let required_bucket_bits = self.overhead - calc_log_p(self.data.len() / (1 << layer));

        let bucket_bits_fract = required_bucket_bits + bucket_bits_overhead;
        let current_bucket_bits = bucket_bits_fract.ceil() as usize;
        let new_overhead = bucket_bits_fract.ceil() - bucket_bits_fract;

        // implicit phi
        for seed in ((parent_seed << current_bucket_bits)..).take(1 << current_bucket_bits) {
            let data_slice = &mut self.data[1 << layer..][..1 << layer];
            // #[cfg(feature = "debug_output")]
            // {
            //     self.stats.bij_tests += 1;
            // }
            #[allow(clippy::collapsible_if)]
            if self
                .hasher
                .is_binary_split(seed, &data_slice.iter().by_ref().collect::<Vec<_>>())
            // todo has this overhead?
            {
                data_slice.sort_unstable_by_key(|v| self.hasher.hash_binary(seed, v));
                // todo cached sort?
                // todo sort only after complete layer?

                // #[cfg(feature = "debug_output")]
                // {
                //     self.stats.num_bijections_found[bucket_idx_1 - 1] += 1;
                // }

                if self.find_seed_task(
                    task_idx_1 + 1,
                    seed,
                    bit_index + current_bucket_bits,
                    new_overhead,
                ) {
                    let index = seed & ((1 << current_bucket_bits) - 1);
                    self.full_information[bit_index..][..current_bucket_bits].store_be(index);
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
fn calc_log_p(n: usize) -> Float {
    assert!(n.is_power_of_two());
    (1..=n / 2)
        .map(|i| (n as Float / 8. * i as Float) + 1. / 4.)
        .map(Float::log2)
        .sum()
}

fn binary_split_probability(size: usize) -> Float {
    // using stirlings approximation
    // todo good enough?
    (size as Float)
}

#[cfg(test)]
mod test {
    use crate::rec_mvp::RecMvp;

    use super::SrsMphf;

    fn test_create_mphf() {
        let size = 100;
        let overhead = 0.01;
        let data = (0..size).collect::<Vec<_>>();

        let mphf = SrsMphf::new_random(&data, overhead);
    }
}
