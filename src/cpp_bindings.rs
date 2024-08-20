use cxx::CxxString;

use cxx::CxxVector;

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type SrsMphf;
        fn create(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsMphf>;
        fn hash(&self, value: &CxxString) -> usize;
    }
}


struct SrsMphf(Box<crate::mphf::SrsMphf<CxxString>>);

fn create(data: &CxxVector<CxxString>, overhead: f64) -> Box<SrsMphf> {
    Box::new(SrsMphf(Box::new(crate::mphf::SrsMphf::new(data, overhead))))
}

impl SrsMphf {
    fn hash(&self, value: &CxxString) -> usize {
        self.0.hash(value)
    }
}