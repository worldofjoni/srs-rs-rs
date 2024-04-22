use recsplit::RecSplit;

fn main() {
    RecSplit::new_random(&(0..100).collect::<Vec<_>>());
}
