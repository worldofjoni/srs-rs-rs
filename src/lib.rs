use std::{
    hash::{DefaultHasher, Hash, Hasher},
    iter::{once, repeat},
    marker::PhantomData,
};

use ahash::AHasher;
use fnv::FnvHasher;
#[cfg(not(test))]
use log::debug;
#[cfg(test)]
use std::println as debug;

use rustc_hash::FxHasher;

#[derive(Debug)]
pub struct RecSplit<T: Hash> {
    _phantom: PhantomData<T>,
    tree: SplittingTree,
}

impl<T: Hash> RecSplit<T> {
    /// `leaf_size` must be in `1..=24`.\
    /// `values` must be **distinct**, that is they _can_ produce unique hashes.
    pub fn new(leaf_size: usize, values: &[T]) -> Self {
        debug_assert!((1..=24).contains(&leaf_size));

        let size = values.len();
        let mut values = values.iter().collect::<Vec<_>>();

        Self {
            _phantom: PhantomData,
            tree: construct_splitting_tree(
                leaf_size,
                &mut values,
                construct_splitting_strategy(leaf_size, size),
            ),
        }
    }

    pub fn size(&self) -> usize {
        self.tree.get_size()
    }

    /// ensures result in `0..self.get_size()`
    pub fn hash(&self, value: &T) -> usize {
        hash_with_tree(&self.tree, value)
    }

    pub fn serialize(&self) -> Box<[u8]> {
        todo!()
    }

    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        todo!()
    }
}

fn hash_with_tree<T: Hash>(tree: &SplittingTree, value: &T) -> usize {
    match tree {
        SplittingTree::Inner(seed, subtrees, _) => {
            let splits = subtrees.iter().map(|s| s.get_size()).collect::<Vec<_>>();
            let bucket = get_hash_bucket(value, *seed, &splits);
            let offset: usize = subtrees.iter().take(bucket).map(|t| t.get_size()).sum();

            offset + hash_with_tree(&subtrees[bucket], value)
        }
        SplittingTree::Leaf(seed, size) => hash_with_seed(*seed, *size, value),
    }
}

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

/// `splitting_strategy`: list of _ordinary_ sizes of buckets in each layer.
fn construct_splitting_tree<T: Hash>(
    leaf_size: usize,
    values: &mut [&T],
    mut splitting_strategy: impl Iterator<Item = usize> + Clone,
) -> SplittingTree {
    let size = values.len();
    if size <= leaf_size {
        debug!("constructing leaf node of size {}", size);
        let split = vec![1; size];
        let seed = find_split_seed(&split, values);
        debug!("\tsplit with seed {seed}");
        return SplittingTree::Leaf(seed, size);
    }

    let split_degree = splitting_strategy.next().expect("splitting unit");
    debug!("constructing inner node for {size} values and splitting degree {split_degree}");

    let expected_child_size = size.div_ceil(split_degree);
    let mut split = vec![expected_child_size; split_degree];
    *split.last_mut().expect("no empty tree") -= expected_child_size * split_degree - size;

    let seed = find_split_seed(&split, values);
    debug!("\tsplit with seed {seed}");
    values.sort_unstable_by_key(|v| get_hash_bucket(v, seed, &split));

    let children: Vec<_> = values
        .chunks_mut(expected_child_size)
        .map(|chunk| construct_splitting_tree(leaf_size, &mut *chunk, splitting_strategy.clone()))
        .collect();

    SplittingTree::Inner(seed, children, size)
}

#[derive(Debug)]
enum SplittingTree {
    // number of previous hashes, ?, seed
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

/// Hashes `value` with hash function Phi_`seed`^`max`, that is each hash is in range [0,`max`).
fn hash_with_seed<T: Hash>(seed: usize, max: usize, value: &T) -> usize {
    let mut hasher = AHasher::default();
    hasher.write_usize(seed);
    value.hash(&mut hasher);
    // debug!("raw hash {}", hasher.finish());
    hasher.finish() as usize % max
}

/// require: sum of splits == values.len()
fn find_split_seed<T: Hash>(split: &[usize], values: &[T]) -> usize {
    for i in 0..usize::MAX {
        if i % 10000 == 0 && i != 0 {
            debug!("finding seed: iteration {i}");
        }

        if is_split(i, split, values) {
            return i;
        }
    }
    panic!("no split found");
}

/// `split` is list of length of splitting sections, must sum up to `values.len()`
fn is_split<T: Hash>(seed: usize, splits: &[usize], values: &[T]) -> bool {
    let size = values.len();
    debug_assert_eq!(
        size,
        splits.iter().sum::<usize>(),
        "splits did not sum up to number of values"
    );
    let mut num_in_splits = vec![0; splits.len()];

    for val in values {
        let bucket_idx = get_hash_bucket(val, seed, splits);
        num_in_splits[bucket_idx] += 1;
    }
    // debug!("splits {splits:?}, buckets {num_in_splits:?}");

    num_in_splits == splits
}

/// returns the bucket of a value according to a splitting and seed
fn get_hash_bucket<T: Hash>(value: &T, seed: usize, splits: &[usize]) -> usize {
    let hash = hash_with_seed(seed, splits.iter().sum(), value);
    // debug!("hash {hash}");
    debug_assert!(hash < splits.iter().sum());

    // TODO: improve with precalculated boundaries?
    for (bucked_idx, bucket_max) in splits
        .iter()
        .scan(0usize, |a, b| {
            *a += *b;
            Some(*a)
        })
        .enumerate()
    {
        if hash < bucket_max {
            return bucked_idx;
        }
    }
    unreachable!("hash shall be smaller than number of values (sum of splits)");
}

#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};

    use rand::random;

    use crate::{
        construct_splitting_strategy, find_split_seed, get_hash_bucket, hash_with_seed, RecSplit,
    };

    // #[test]
    // fn test_bucket_index() {
    //     assert_eq!(0, get_hash_bucket(0, &[1, 1, 2]));
    //     assert_eq!(1, get_hash_bucket(1, &[1, 1, 2]));
    //     assert_eq!(2, get_hash_bucket(3, &[1, 1, 2]));
    //     assert_eq!(0, get_hash_bucket(0, &[5, 5]));
    //     assert_eq!(0, get_hash_bucket(4, &[5, 5]));
    //     assert_eq!(1, get_hash_bucket(5, &[5, 5]));
    //     assert_eq!(1, get_hash_bucket(9, &[5, 5]));
    // }

    // #[test]
    // fn test_split() {
    //     const SIZE: usize = 1000;
    //     let values = (0..SIZE).collect::<Vec<_>>();
    //     let split = [SIZE / 2; 2];
    //     assert_eq!(43, find_split_seed(&split, &values))
    // }

    // #[test]
    // fn test_split_mphf() {
    //     const SIZE: usize = 10;
    //     let values = (0..SIZE).collect::<Vec<_>>();
    //     let split = [1; SIZE];
    //     assert_eq!(43, find_split_seed(&split, &values))
    // }

    #[test]
    fn get_bucket_vs_hash_with_seed() {
        let size = 100;
        let splits = vec![1; size];

        for seed in 0..100 {
            for value in 0..1000 {
                assert_eq!(
                    hash_with_seed(seed, size, &value),
                    get_hash_bucket(&value, seed, &splits)
                );
            }
        }
    }

    #[test]
    fn test_construct_strategy() {
        let max_leaf_size = 24;

        for size in (0..1000).step_by(100).skip(1) {
            for leaf_size in 1..max_leaf_size.min(size) {
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
    fn test_find_split() {
        for split in 2..=4 {
            println!("split in {split} parts");
            let size = 36;
            assert!(size % split == 0);
            let splits = vec![size / split; split];
            let values = (0..size).map(|_| random::<usize>()).collect::<Vec<_>>();

            let seed = find_split_seed(&splits, &values);

            let mut result = vec![0; split];
            values
                .iter()
                .for_each(|v| result[get_hash_bucket(&v, seed, &splits)] += 1);
            assert_eq!(
                result, splits,
                "failed for split in {split} with values {values:?}"
            );
        }
    }

    #[test]
    fn test_single_tree() {
        env_logger::init();

        let values = (0..100).collect::<Vec<_>>();
        let tree = RecSplit::new(10, &values);
        println!("{tree:?}");

        let set = (0..tree.size())
            .map(|i| tree.hash(&i))
            .collect::<HashSet<_>>();
        let mut histogram = HashMap::new();

        (0..tree.size()).for_each(|i| *histogram.entry(tree.hash(&i)).or_insert(0) += 1);
        println!("histogram: {histogram:?}");

        assert_eq!(set.len(), values.len());
        assert_eq!(set, (0..tree.size()).collect());
    }

    #[test]
    fn test_many_tree() {
        env_logger::init();

        for i in (0..500).step_by(100) {
            let size = i;
            println!("size {size}");
            let leaf_size = 15;
            let values = (0..size)
                .map(|_| rand::random::<usize>())
                .collect::<Vec<_>>();
            let mphf = RecSplit::new(leaf_size, &values);

            let mut results = vec![0; size];
            values.iter().for_each(|v| results[mphf.hash(v)] += 1);
            assert!(
                results.iter().all(|v| *v == 1),
                "failed in {i} for {values:?} with {results:?}"
            );
        }
    }
}
