use std::hash::{BuildHasher, Hasher};

#[derive(Debug)]
pub struct MphfHasher<H: BuildHasher> {
    pub hasher: H,
    #[cfg(feature = "debug_output")]
    pub num_hash_evals: Cell<usize>,
}

impl<H: BuildHasher> MphfHasher<H> {
    pub fn new(hasher: H) -> Self {
        Self {
            hasher,
            #[cfg(feature = "debug_output")]
            num_hash_evals: Cell::new(0),
        }
    }
}

type HashVal = u64;

impl<H: BuildHasher> MphfHasher<H> {
    // ============== MPHF with binary splits ===============================================================

    pub fn is_binary_split(&self, seed: usize, values: &[HashVal]) -> bool {
        let size = values.len();
        debug_assert!(size & 1 == 0);
        let mut child_sizes = [0_usize; 2];

        for val in values {
            let child_idx = self.hash_binary(seed, *val);
            child_sizes[child_idx] += 1;
        }

        #[cfg(feature = "debug_output")]
        self.num_hash_evals
            .set(self.num_hash_evals.get() + values.len());

        child_sizes[0] == size >> 1
    }

    pub fn hash_binary(&self, seed: usize, value: HashVal) -> usize {
        let hash = self.hash(seed, value) as usize;

        hash & 1
    }

    // --- non power of twos

    pub fn is_generic_split(&self, seed: usize, values: &[HashVal]) -> bool {
        if values.len().is_power_of_two() {
            self.is_binary_split(seed, values)
        } else {
            self.is_ratio_split(seed, values)
        }
    }

    pub fn hash_generic(&self, seed: usize, value: HashVal, size: usize) -> usize {
        if size.is_power_of_two() {
            self.hash_binary(seed, value)
        } else {
            self.hash_ratio(seed, value, size)
        }
    }

    /// splits into power-of-two- and rest-sized parts
    pub fn is_ratio_split(&self, seed: usize, values: &[HashVal]) -> bool {
        let mut child_sizes = [0_usize; 2];
        let size = values.len();

        for val in values {
            let child_idx = self.hash_ratio(seed, *val, size);
            child_sizes[child_idx] += 1;
        }

        #[cfg(feature = "debug_output")]
        self.num_hash_evals
            .set(self.num_hash_evals.get() + values.len());

        child_sizes[0] == 1 << size.ilog2()
    }

    pub fn hash_ratio(&self, seed: usize, value: HashVal, size: usize) -> usize {
        let hash = self.hash(seed, value);

        let distributed = ((hash as u128 * size as u128) >> u64::BITS) as usize;

        (distributed >= 1 << size.ilog2()) as usize
    }

    fn hash(&self, seed: usize, value: HashVal) -> u64 {
        let mut hasher = self.hasher.build_hasher();

        hasher.write_u64(seed as u64 ^ value);

        hasher.finish()
    }
}

#[cfg(test)]
mod test {

    use super::MphfHasher;

    #[test]
    fn test_hash_binary() {
        let hasher = MphfHasher::new(ahash::RandomState::new());
        let mut nums = [0; 2];
        for i in 0..1000 {
            let hash = hasher.hash_binary(101010, i);
            nums[hash] += 1;
        }
        println!("{nums:?}");
    }

    #[test]
    fn test_is_binary_split() {
        let hasher = MphfHasher::new(ahash::RandomState::new());
        let size = 100;
        let values = (0u64..size).collect::<Vec<_>>();
        // let values = values.iter().collect::<Vec<_>>();

        let mut seed = 0;
        for seed2 in 0.. {
            if hasher.is_binary_split(seed2, &values) {
                seed = seed2;
                println!("found seed {seed}");
                break;
            }
        }

        let mut nums = [0; 2];
        for i in values {
            let hash = hasher.hash_binary(seed, i);
            nums[hash] += 1;
        }
        println!("{nums:?}");
        assert_eq!(nums[0], nums[1]);
    }
}
