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

#[cfg(feature = "debug_output")]
pub use hasher::BIJECTIONS_CHECKED;
#[cfg(feature = "debug_output")]
pub use hasher::SPLITS_CHECKED;

pub use hasher::RecHasher;

pub fn get_graphics_path(filename: &str) -> String {
    use std::fs::DirBuilder;
    let folder = "target/graphics";
    DirBuilder::new().recursive(true).create(folder).unwrap();
    format!("{folder}/{filename}")
}
