use std::{
    hash::{DefaultHasher, Hash, Hasher},
    iter::repeat,
    marker::PhantomData,
};

pub struct RecSplit<T: Hash> {
    _phantom: PhantomData<T>,
    tree: SplittingTree,
}

pub type HashInt = u64;

impl<T: Hash> RecSplit<T> {
    pub fn new(leaf_size: usize, tree_arity: usize, values: &[T]) -> Self {
        debug_assert!(leaf_size >= tree_arity);
        Self {
            _phantom: PhantomData,
            tree: construct_splitting_tree(leaf_size, values, repeat(0)), // todo
        }
    }

    pub fn hash(&self, value: &T) -> HashInt {
        todo!()
    }

    pub fn serialize(&self) -> Box<[u8]> {
        todo!()
    }

    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        todo!()
    }
}

/// `splitting units`: list of _ordinary_ sizes of buckets in each layer.
fn construct_splitting_tree<T: Hash>(
    leaf_size: usize,
    values: &[T],
    mut splitting_units: impl Iterator<Item = usize>,
) -> SplittingTree {
    if values.len() <= leaf_size {
        let split = vec![1; values.len()];
        let hash = find_split_seed(&split, values);
        return SplittingTree::Leaf(hash);
    }

    let size = values.len();
    let split_unit = splitting_units.next().expect("splitting unit");
    let mut split = vec![split_unit; size.div_ceil(split_unit)];
    *split.last_mut().expect("no empty tree") -= split_unit * split.len() - size;

    let seed = find_split_seed(&split, values);
    todo!();
}

enum SplittingTree {
    Inner(HashInt, Vec<SplittingTree>),
    Leaf(HashInt),
}

/// Hashes `value` with hash function Phi_`seed`^`max`, that is each hash is in range [0,`max`).
fn hash_with_seed<T: Hash>(seed: HashInt, max: HashInt, value: T) -> HashInt {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(seed);
    value.hash(&mut hasher);
    hasher.finish() % max
}

/// require: sum of splits == values.len()
fn find_split_seed<T: Hash>(split: &[usize], values: &[T]) -> HashInt {
    for i in 0..HashInt::MAX {
        if i % 10000 == 0 {
            println!("finding seed: iteration {i}");
        }

        if is_split(i, split, values) {
            return i;
        }
    }
    panic!("no split found");
}

/// `split` is list of length of splitting sections, must sum up to `values.len()`
fn is_split<T: Hash>(seed: HashInt, splits: &[usize], values: &[T]) -> bool {
    let max_seed = values.len();
    debug_assert_eq!(
        max_seed,
        splits.iter().sum(),
        "splits did not sum up to number of values"
    );
    let mut num_in_splits = vec![0; splits.len()];

    for val in values {
        let hash = hash_with_seed(seed, max_seed as HashInt, val); // todo: avoid `as`
        let bucket_idx = get_hash_bucket(hash, splits);
        num_in_splits[bucket_idx] += 1;
    }
    // println!("buckets {num_in_splits:?}");

    num_in_splits
        .iter()
        .zip(splits.iter())
        .all(|(num, split)| *num == *split)
}

fn get_hash_bucket(hash: HashInt, splits: &[usize]) -> usize {
    debug_assert!((hash as usize) < splits.iter().sum());
    for (bucked_idx, bucket_max) in splits
        .iter()
        .scan(0usize, |a, b| {
            *a += *b;
            Some(*a)
        })
        .enumerate()
    {
        if (hash as usize) < bucket_max {
            return bucked_idx;
        }
    }
    unreachable!("hash shall be smaller than number of values (sum of splits)");
}

#[cfg(test)]
mod tests {
    use crate::{find_split_seed, get_hash_bucket};

    #[test]
    fn test_bucket_index() {
        assert_eq!(0, get_hash_bucket(0, &[1, 1, 2]));
        assert_eq!(1, get_hash_bucket(1, &[1, 1, 2]));
        assert_eq!(2, get_hash_bucket(3, &[1, 1, 2]));
        assert_eq!(0, get_hash_bucket(0, &[5, 5]));
        assert_eq!(0, get_hash_bucket(4, &[5, 5]));
        assert_eq!(1, get_hash_bucket(5, &[5, 5]));
        assert_eq!(1, get_hash_bucket(9, &[5, 5]));
    }

    #[test]
    fn test_split() {
        const SIZE: usize = 1000;
        let values = (0..SIZE).collect::<Vec<_>>();
        let split = [SIZE / 2; 2];
        assert_eq!(43, find_split_seed(&split, &values))
    }

    #[test]
    fn test_split_mphf() {
        const SIZE: usize = 10;
        let values = (0..SIZE).collect::<Vec<_>>();
        let split = [1; SIZE];
        assert_eq!(43, find_split_seed(&split, &values))
    }
}
