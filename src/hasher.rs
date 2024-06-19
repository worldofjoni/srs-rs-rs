use std::hash::{BuildHasher, Hash, Hasher};

use lazy_static::lazy_static;

use crate::{recsplit::MAX_LEAF_SIZE, splitting_tree::SplittingTree};

#[derive(Debug, Clone)]
pub struct RecHasher<H: BuildHasher + Clone>(pub H);

impl<H: BuildHasher + Clone> RecHasher<H> {
    pub fn hash_with_tree(&self, tree: &SplittingTree, value: &impl Hash) -> usize {
        match tree {
            SplittingTree::Inner {
                seed,
                children,
                size,
            } => {
                let bucket = self.hash_to_child(
                    *seed,
                    *size,
                    children.first().expect("has child").get_size(),
                    value,
                );
                let offset: usize = children
                    .iter()
                    .take(bucket)
                    .map(|t: &SplittingTree| t.get_size())
                    .sum();

                offset + self.hash_with_tree(&children[bucket], value)
            }
            SplittingTree::Leaf { seed, size } => self.hash_to_child(*seed, *size, 1, value),
        }
    }

    /// `splitting_strategy`: list of _ordinary_ sizes of buckets in each layer.
    pub fn construct_splitting_tree(
        &self,
        leaf_size: usize,
        values: &mut [&impl Hash],
        mut splitting_strategy: impl Iterator<Item = usize> + Clone,
    ) -> SplittingTree {
        let size = values.len();
        if size <= leaf_size {
            // debug!("constructing leaf node of size {}", size);
            let seed = self.find_split_seed(1, values); // todo use specialized version
                                                        // debug!("\tsplit with seed {seed}");
            return SplittingTree::Leaf { seed, size };
        }

        let split_degree = splitting_strategy.next().expect("splitting unit");
        let max_child_size = size.div_ceil(split_degree);
        // debug!("constructing inner node for {size} values and splitting degree {split_degree}");

        let seed = self.find_split_seed(max_child_size, values);
        // debug!("\tsplit with seed {seed}");

        values.sort_unstable_by_key(|v| self.hash_to_child(seed, size, max_child_size, v));

        let children: Vec<_> = values
            .chunks_mut(max_child_size)
            .map(|chunk| {
                self.construct_splitting_tree(leaf_size, &mut *chunk, splitting_strategy.clone())
            })
            .collect();

        SplittingTree::Inner {
            seed,
            children,
            size,
        }
    }

    /// require: sum of splits == values.len()
    fn find_split_seed(&self, max_child_size: usize, values: &[impl Hash]) -> usize {
        for i in 0..usize::MAX {
            if i % 10000 == 0 && i != 0 {
                crate::debug!("finding seed: iteration {i}");
            }

            if self.is_split(i, max_child_size, values) {
                return i;
            }
        }
        panic!("no split found");
    }

    /// `split` is list of length of splitting sections, must sum up to `values.len()`
    pub fn is_split(&self, seed: usize, max_child_size: usize, values: &[impl Hash]) -> bool {
        let size = values.len();
        let num_children = size.div_ceil(max_child_size);

        const MAX_CHILDREN: usize = 10; // given by leaf_size <= 24 and splitting strategy
        const MAX_SPLIT_DEGREE: usize = const_max(MAX_CHILDREN, MAX_LEAF_SIZE); // either split according to strategy or directly into leafs

        let mut child_sizes = [0_usize; MAX_SPLIT_DEGREE];

        for val in values {
            let child_idx = self.hash_to_child(seed, size, max_child_size, val);
            child_sizes[child_idx] += 1;
        }

        // todo: more efficient compare possible using wider registers?
        child_sizes
            .iter()
            .take(num_children - 1)
            .all(|v| *v == max_child_size)
    }

    /// hashes into one of the `0..size.div_ceil(max_child_size)` children
    pub fn hash_to_child(
        &self,
        seed: usize,
        size: usize,
        max_child_size: usize,
        value: &impl Hash,
    ) -> usize {
        let mut hasher = self.0.build_hasher();

        hasher.write_usize(seed);
        value.hash(&mut hasher);

        let hash = hasher.finish() as usize;

        // getting the child index: hash / max_child_size
        fast_div(distribute(hash, size), max_child_size)
    }

    /// `split` is list of length of splitting sections, must sum up to `values.len()`
    pub fn is_bijection(&self, seed: usize, values: &[impl Hash]) -> bool {
        let size = values.len();

        debug_assert!(size <= u32::BITS as usize);

        let mut child_sizes: u32 = 0;

        lazy_static! {
            // first value si for size 1!
            // inspired by Recsplit <https://github.com/vigna/sux/blob/5bdaa93b0d3f74841e9a5a2a04e72f1b4c8de6f9/sux/function/RecSplit.hpp#L275C1-L282C2>
            static ref MIDPOINTS: [u16; MAX_LEAF_SIZE] = (1..=MAX_LEAF_SIZE).map(|i| ((4. * i as f32).sqrt().ceil() as u16).min(i as u16)).collect::<Vec<_>>().try_into().unwrap();
        }

        let midpoint = size / 2; // todo not really worth it, maybe try with different PERFECT sizes: MIDPOINTS[size - 1] as usize; // todo find optimal midpoint
        let (v1, v2) = values.split_at(midpoint);

        for val in v1 {
            let child_idx = self.hash_bijection(seed, size, val);
            child_sizes |= 1 << child_idx;
        }

        if child_sizes.count_ones() != midpoint as u32 {
            return false;
        }

        for val in v2 {
            let child_idx = self.hash_bijection(seed, size, val);
            child_sizes |= 1 << child_idx;
        }

        child_sizes == (1 << size) - 1
    }

    /// hashes into one of the `0..size.div_ceil(max_child_size)` children
    pub fn hash_bijection(&self, seed: usize, size: usize, value: &impl Hash) -> usize {
        let mut hasher = self.0.build_hasher();

        hasher.write_usize(seed);
        value.hash(&mut hasher);

        let hash = hasher.finish() as usize;

        // getting the child index: hash / max_child_size
        distribute(hash, size)
    }

    /// Hashing for initial bucket assignment. Tries to be independent from [`hash_to_child`] to avoid correlation, thus hashes twice.
    pub fn hash_to_bucket(&self, num_buckets: usize, value: &impl Hash) -> usize {
        let mut hasher = self.0.build_hasher();
        value.hash(&mut hasher);
        value.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        distribute(hash, num_buckets)
    }
}

#[inline(always)]
fn fast_div(a: usize, b: usize) -> usize {
    if b.is_power_of_two() {
        a >> b.ilog2()
    } else {
        a / b
    }
}

/// distributes the hash evenly in `0..max`.
#[inline(always)]
fn distribute(hash: usize, max: usize) -> usize {
    ((max as u128 * hash as u128) >> usize::BITS) as usize
}

// #[inline(always)]
// fn fast_mod(a: usize, b: usize) -> usize {
//     if b.is_power_of_two() {
//         a & (b - 1)
//     } else {
//         a % b
//     }
// }

const fn const_max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod test {
    use rand::random;

    use crate::hasher::{const_max, fast_div};

    use super::RecHasher;

    #[test]
    fn test_fast() {
        let num = 1234;
        assert_eq!(fast_div(num, 8), num / 8);
        // assert_eq!(fast_mod(num, 8), num % 8);
    }

    #[test]
    fn test_max() {
        for i in 0..100 {
            for j in 0..100 {
                assert_eq!(i.max(j), const_max(i, j));
            }
        }
    }

    #[test]
    fn test_find_split() {
        let hasher = RecHasher(ahash::RandomState::new());
        for split in 2..=4 {
            println!("split in {split} parts");
            let size = 36;
            assert!(size % split == 0);
            let max_child_size = size / split;
            let values = (0..size).map(|_| random::<usize>()).collect::<Vec<_>>();

            let seed = hasher.find_split_seed(size / split, &values);

            let mut result = vec![0; split];
            values
                .iter()
                .for_each(|v| result[hasher.hash_to_child(seed, size, max_child_size, v)] += 1);
            assert_eq!(
                result,
                vec![max_child_size; split],
                "failed for split in {split} with values {values:?}"
            );
        }
    }

    #[test]
    fn test_hash_child() {
        let hasher = RecHasher(ahash::RandomState::new());
        for size in 1..100 {
            for max_child_size in 1..size {
                for value in 0..100 {
                    for seed in 0..1000 {
                        let hash = hasher.hash_to_child(seed, size, max_child_size, &value);
                        assert!(hash < size.div_ceil(max_child_size))
                    }
                }
            }
        }
    }
}
