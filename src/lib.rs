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
    tree: SplittingTree,
}

impl<T: Hash, H: BuildHasher> RecSplit<T, H> {
    /// `leaf_size` must be in `1..=24`.\
    /// `values` must be **distinct**, that is they _can_ produce unique hashes.
    pub fn with_state(leaf_size: usize, values: &[T], random_state: H) -> Self {
        debug_assert!((1..=24).contains(&leaf_size));

        let size = values.len();
        let mut values = values.iter().collect::<Vec<_>>();
        let hasher = RecHasher(random_state);
        Self {
            _phantom: PhantomData,
            tree: hasher.construct_splitting_tree(
                leaf_size,
                &mut values,
                construct_splitting_strategy(leaf_size, size),
            ),
            hasher,
        }
    }

    pub fn size(&self) -> usize {
        self.tree.get_size()
    }

    /// ensures result in `0..self.get_size()`
    pub fn hash(&self, value: &T) -> usize {
        self.hasher.hash_with_tree(&self.tree, value)
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
        Self::with_state(16, values, ahash::RandomState::new())
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
        let mut num_in_splits = vec![0; size.div_ceil(max_child_size)];

        for val in values {
            let bucket_idx = self.hash_to_child(seed, size, max_child_size, val);
            num_in_splits[bucket_idx] += 1;
        }
        // debug!("splits {splits:?}, buckets {num_in_splits:?}");

        num_in_splits
            .iter()
            .rev()
            .skip(1)
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

        if max_child_size.is_power_of_two() {
            (hash % size) >> max_child_size.ilog2()
        } else {
            hash % size / max_child_size
        }
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use rand::random;

    use crate::{construct_splitting_strategy, RecHasher, RecSplit};

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
        let values = (0..1000).collect::<Vec<_>>();
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
        for i in (0..500).step_by(100) {
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
