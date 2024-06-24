#[cfg(not(test))]
pub use log::debug;
#[cfg(test)]
pub use std::println as debug;

mod golomb_rice;
mod hasher;
pub mod rec_tphf;
pub mod mvp;
mod recsplit;
mod splitting_tree;

pub use recsplit::LooseRecSplit;

pub use hasher::RecHasher;
