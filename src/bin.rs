use recsplit::{mvp::MVP, LooseRecSplit};

// necessary to generate assembly
fn main() {
    LooseRecSplit::new_random(&(0..100).collect::<Vec<_>>());

    MVP::new_random(
        &(0..100).collect::<Vec<_>>().chunks(10).collect::<Vec<_>>(),
        0.01,
    );
}
