#[cfg(not(test))]
pub use log::debug;
#[cfg(test)]
pub use std::println as debug;

mod golomb_rice;
mod hasher;
pub mod mvp;
pub mod rec_mvp;
pub mod rec_tphf;
mod recsplit;
mod splitting_tree;

pub use recsplit::LooseRecSplit;

pub use hasher::RecHasher;
