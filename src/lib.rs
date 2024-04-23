use std::{
    hash::{BuildHasher, Hash, Hasher},
    iter::{once, repeat},
    marker::PhantomData,
};

#[cfg(not(test))]
use log::debug;
#[cfg(test)]
use std::println as debug;

#[derive(Debug)]
pub struct RecSplit<T: Hash, H: BuildHasher> {
    _phantom: PhantomData<T>,
    hasher: RecHasher<H>,
    trees: Vec<SplittingTree>,
}

pub const MAX_LEAF_SIZE: usize = 24;
impl<T: Hash, H: BuildHasher> RecSplit<T, H> {
    /// `leaf_size` must be in `1..=24`.\
    /// `values` must be **distinct**, that is they _can_ produce unique hashes.
    pub fn with_state(
        leaf_size: usize,
        avg_bucket_size: usize,
        values: &[T],
        random_state: H,
    ) -> Self {
        debug_assert!((1..=MAX_LEAF_SIZE).contains(&leaf_size));

        let size = values.len();
        let num_buckets = size.div_ceil(avg_bucket_size);
        let mut values = values.iter().collect::<Vec<_>>();
        let hasher = RecHasher(random_state);

        values.sort_unstable_by_key(|v| hasher.hash_to_bucket(num_buckets, v));
        let trees = values
            .chunk_by_mut(|a, b| {
                hasher.hash_to_bucket(num_buckets, a) == hasher.hash_to_bucket(num_buckets, b)
            })
            .map(|bucket| {
                let bucket_size = bucket.len();
                hasher.construct_splitting_tree(
                    leaf_size,
                    bucket,
                    construct_splitting_strategy(leaf_size, bucket_size),
                )
            })
            .collect::<Vec<_>>();

        Self {
            _phantom: PhantomData,
            trees,
            hasher,
        }
    }

    pub fn size(&self) -> usize {
        self.trees.iter().map(SplittingTree::get_size).sum()
    }

    /// ensures result in `0..self.get_size()`
    pub fn hash(&self, value: &T) -> usize {
        let bucket = self.hasher.hash_to_bucket(self.trees.len(), value);
        let offset: usize = self
            .trees
            .iter()
            .take(bucket)
            .map(SplittingTree::get_size)
            .sum();

        offset + self.hasher.hash_with_tree(&self.trees[bucket], value)
    }

    pub fn serialize(&self) -> Box<[u8]> {
        todo!()
    }

    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        todo!("{:?}", bytes)
    }
}

impl<T: Hash> RecSplit<T, ahash::RandomState> {
    pub fn new_random(values: &[T]) -> Self {
        Self::with_state(12, 8, values, ahash::RandomState::new())
    }
}

#[derive(Debug)]
enum SplittingTree {
    // number of previous hashes, ?, seed // todo make struct variants with names
    Inner(usize, Vec<SplittingTree>, usize),
    Leaf(usize, usize),
}

impl SplittingTree {
    fn get_size(&self) -> usize {
        *match self {
            SplittingTree::Inner(_, _, size) => size,
            SplittingTree::Leaf(_, size) => size,
        }
    }
}

/// Returns an iterator over the number of children in each layer starting at the root layer.
fn construct_splitting_strategy(
    leaf_size: usize,
    size: usize,
) -> impl Iterator<Item = usize> + Clone {
    let last_split_degree = 2.max((0.35 * leaf_size as f32 + 0.5).ceil() as usize);
    let second_last_degree = if leaf_size >= 7 {
        (0.21 * leaf_size as f32 + 0.9).ceil() as usize
    } else {
        2
    };

    let subtrees_to_cover = size.div_ceil(last_split_degree * second_last_degree * leaf_size);
    let num_2_layers = subtrees_to_cover
        .next_power_of_two()
        .checked_ilog2()
        .unwrap_or_default() as usize;

    repeat(2)
        .take(num_2_layers)
        .chain(once(second_last_degree))
        .chain(once(last_split_degree))
}

#[derive(Debug, Clone)]
struct RecHasher<H: BuildHasher>(H);

impl<H: BuildHasher> RecHasher<H> {
    fn hash_with_tree(&self, tree: &SplittingTree, value: &impl Hash) -> usize {
        match tree {
            SplittingTree::Inner(seed, subtrees, size) => {
                let bucket = self.hash_to_child(
                    *seed,
                    *size,
                    subtrees.first().expect("has child").get_size(),
                    value,
                );
                let offset: usize = subtrees
                    .iter()
                    .take(bucket)
                    .map(|t: &SplittingTree| t.get_size())
                    .sum();

                offset + self.hash_with_tree(&subtrees[bucket], value)
            }
            SplittingTree::Leaf(seed, size) => self.hash_to_child(*seed, *size, 1, value),
        }
    }

    /// `splitting_strategy`: list of _ordinary_ sizes of buckets in each layer.
    fn construct_splitting_tree(
        &self,
        leaf_size: usize,
        values: &mut [&impl Hash],
        mut splitting_strategy: impl Iterator<Item = usize> + Clone,
    ) -> SplittingTree {
        let size = values.len();
        if size <= leaf_size {
            debug!("constructing leaf node of size {}", size);
            let seed = self.find_split_seed(1, values);
            debug!("\tsplit with seed {seed}");
            return SplittingTree::Leaf(seed, size);
        }

        let split_degree = splitting_strategy.next().expect("splitting unit");
        let max_child_size = size.div_ceil(split_degree);
        debug!("constructing inner node for {size} values and splitting degree {split_degree}");

        let seed = self.find_split_seed(max_child_size, values);
        debug!("\tsplit with seed {seed}");

        values.sort_unstable_by_key(|v| self.hash_to_child(seed, size, max_child_size, v));

        let children: Vec<_> = values
            .chunks_mut(max_child_size)
            .map(|chunk| {
                self.construct_splitting_tree(leaf_size, &mut *chunk, splitting_strategy.clone())
            })
            .collect();

        SplittingTree::Inner(seed, children, size)
    }

    /// require: sum of splits == values.len()
    fn find_split_seed(&self, max_child_size: usize, values: &[impl Hash]) -> usize {
        for i in 0..usize::MAX {
            if i % 10000 == 0 && i != 0 {
                debug!("finding seed: iteration {i}");
            }

            if self.is_split(i, max_child_size, values) {
                return i;
            }
        }
        panic!("no split found");
    }

    /// `split` is list of length of splitting sections, must sum up to `values.len()`
    fn is_split(&self, seed: usize, max_child_size: usize, values: &[impl Hash]) -> bool {
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
    fn hash_to_child(
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

    /// Hashing for initial bucket assignment. Tries to be independent from [`hash_to_child`] to avoid correlation, thus hashes twice.
    fn hash_to_bucket(&self, num_buckets: usize, value: &impl Hash) -> usize {
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
    const BITS_PER_USIZE: usize = 8 * std::mem::size_of::<usize>();

    ((max as u128 * hash as u128) >> BITS_PER_USIZE) as usize
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
mod tests {

    use std::collections::HashSet;

    use rand::random;

    use crate::{
        const_max, construct_splitting_strategy, fast_div, RecHasher, RecSplit, MAX_LEAF_SIZE,
    };

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
    fn test_construct_strategy() {
        let max_leaf_size = 24;

        for size in (0..1000).step_by(100).skip(1) {
            for leaf_size in 1..=max_leaf_size.min(size) {
                assert!(
                    construct_splitting_strategy(leaf_size, size).product::<usize>() * leaf_size
                        >= size
                );
            }
        }
        // assert_eq!(
        //     construct_splitting_strategy(leaf_size, size).collect::<Vec<_>>(),
        //     &[]
        // );
    }

    #[test]
    fn test_strategy_limit() {
        for i in 0..50 {
            assert!(construct_splitting_strategy(12, i).all(|v| v <= MAX_LEAF_SIZE))
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
    fn test_single_tree() {
        let values = (0..10000).collect::<Vec<_>>();
        let tree = RecSplit::new_random(&values);
        // println!("{tree:?}");

        let set = (0..tree.size())
            .map(|i| tree.hash(&i))
            .collect::<HashSet<_>>();
        // let mut histogram = HashMap::new();

        // (0..tree.size()).for_each(|i| *histogram.entry(tree.hash(&i)).or_insert(0) += 1);
        // println!("histogram: {histogram:?}");

        assert_eq!(set.len(), values.len());
        assert_eq!(set, (0..tree.size()).collect());
    }

    #[test]
    fn test_many_tree() {
        for i in (0..10000).skip(1).step_by(1000) {
            let size = i;
            println!("size {size}");
            let values = (0..size)
                .map(|_| rand::random::<usize>())
                .collect::<Vec<_>>();
            let mphf = RecSplit::new_random(&values);

            let mut results = vec![0; size];
            values.iter().for_each(|v| results[mphf.hash(v)] += 1);
            assert!(
                results.iter().all(|v| *v == 1),
                "failed in {i} for {values:?} with {results:?}"
            );
        }
    }
}
