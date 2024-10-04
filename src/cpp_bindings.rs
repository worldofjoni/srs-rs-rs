#![allow(unused, non_snake_case)]
use cxx::CxxString;

use cxx::CxxVector;

#[cxx::bridge(namespace = "srs_rs_rs")]
mod ffi {
    extern "Rust" {
        type SrsRecSplit;
        fn constructSrsRecSplit(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsRecSplit>;
        fn query(&self, value: &CxxString) -> usize;
        fn getBits(&self) -> usize;
        fn getBitsPerKey(&self) -> f64;
    }
}

struct SrsRecSplit(Box<crate::mphf::SrsRecSplit<CxxString>>);

fn constructSrsRecSplit(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsRecSplit> {
    Box::new(SrsRecSplit(Box::new(crate::mphf::SrsRecSplit::new(
        data, overhead,
    ))))
}

impl SrsRecSplit {
    fn query(&self, value: &CxxString) -> usize {
        self.0.query(value)
    }

    fn getBits(&self) -> usize {
        self.0.bit_size()
    }

    fn getBitsPerKey(&self) -> f64 {
        self.0.bit_per_key()
    }
}
