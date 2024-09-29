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

pub mod util;

pub use recsplit::LooseRecSplit;
pub mod mvp_assembled;

pub mod mphf;

#[cfg(feature = "cpp_binds")]
pub mod cpp_bindings;


pub use hasher::RecHasher;
