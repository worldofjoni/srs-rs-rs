#![allow(unused, non_snake_case)]
use cxx::CxxString;

use cxx::CxxVector;

#[cxx::bridge(namespace = "srs")]
mod ffi {
    extern "Rust" {
        type SrsMphf;
        fn constructSrsMphf(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsMphf>;
        fn hash(&self, value: &CxxString) -> usize;
        fn getBits(&self) -> usize;
        fn getBitsPerKey(&self) -> f64;
    }
}


struct SrsMphf(Box<crate::mphf::SrsMphf<CxxString>>);

fn constructSrsMphf(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsMphf> {
    Box::new(SrsMphf(Box::new(crate::mphf::SrsMphf::new(data, overhead))))
}

impl SrsMphf {
    fn hash(&self, value: &CxxString) -> usize {
        self.0.hash(value)
    }

    fn getBits(&self) -> usize {
        self.0.bit_size()
    }
    
    fn getBitsPerKey(&self) -> f64 {
        self.0.bit_per_key()
    }
}