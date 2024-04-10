use std::{
    hash::{DefaultHasher, Hash, Hasher},
    iter::{once, repeat},
    marker::PhantomData,
};

#[derive(Debug)]
pub struct RecSplit<T: Hash> {
    _phantom: PhantomData<T>,
    tree: SplittingTree,
    size: usize,
}

pub type HashInt = u64;

impl<T: Hash> RecSplit<T> {
    /// `leaf_size` must be in `1..=24`.
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
            size,
        }
    }

    pub fn get_size(&self) -> usize {
        self.size
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
    if values.len() <= leaf_size {
        println!("constructing leaf node of size {}", values.len());
        let split = vec![1; values.len()];
        let seed = find_split_seed(&split, values);
        println!("\tsplit with seed {seed}");
        return SplittingTree::Leaf(seed);
    }

    let size = values.len();
    let split_degree = splitting_strategy.next().expect("splitting unit");
    println!("constructing inner node for {size} values and splitting degree {split_degree}");

    let expected_child_size = size.div_ceil(split_degree);
    let mut split = vec![expected_child_size; split_degree];
    *split.last_mut().expect("no empty tree") -= expected_child_size * split_degree - size;

    let seed = find_split_seed(&split, values);
    println!("\tsplit with seed {seed}");
    values.sort_unstable_by_key(|v| hash_with_seed(seed, size as u64, v));

    let children: Vec<_> = values
        .chunks_mut(expected_child_size)
        .map(|chunk| construct_splitting_tree(leaf_size, &mut *chunk, splitting_strategy.clone()))
        .collect();

    SplittingTree::Inner(seed, children)
}

#[derive(Debug)]
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
        if i % 10000 == 0 && i != 0 {
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
    use crate::{construct_splitting_strategy, find_split_seed, get_hash_bucket, RecSplit};

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
    fn test_tree() {
        let values = (0..100).collect::<Vec<_>>();
        let tree = RecSplit::new(10, &values);
        println!("{tree:?}");
    }
}
