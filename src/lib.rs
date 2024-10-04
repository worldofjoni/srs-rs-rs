
//! This crate provides a space efficient [minimal perfect hash function](https://en.wikipedia.org/wiki/Perfect_hash_function#Minimal_perfect_hash_function) (MPHF) called [`SrsRecSplit`].
//! For more details, see the [Github](https://github.com/worldofjoni/srs-rs-rs) page.


pub mod hasher;
pub mod mphf;

#[cfg(feature = "cpp_binds")]
pub mod cpp_bindings;

pub use mphf::SrsRecSplit;
