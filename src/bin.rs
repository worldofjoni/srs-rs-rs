use recsplit::{mphf::SrsMphf, mvp::MVP, LooseRecSplit};

// necessary to generate assembly
fn main() {
    // LooseRecSplit::new_random(&(0..100).collect::<Vec<_>>());

    // MVP::new_random(
    //     &(0..100).collect::<Vec<_>>().chunks(10).collect::<Vec<_>>(),
    //     0.01,
    // );

    SrsMphf::new_random(&(0..1 << 10).collect::<Vec<_>>(), 0.001);
}
