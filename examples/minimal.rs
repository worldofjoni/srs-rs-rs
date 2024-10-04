extern crate srs_rs_rs;
use srs_rs_rs::SrsRecSplit;

fn main() {
    let mphf = SrsRecSplit::new([&"hello", &"world", &"42", &"mphf"], 0.01);
    println!("{}, {}", mphf.query(&"hello"), mphf.query(&"world"));
}
