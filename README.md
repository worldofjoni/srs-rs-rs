# RecSplit in Rust
Implementation of [RecSplit](https://arxiv.org/abs/1910.06416)---a [minimal perfect hash function](https://en.wikipedia.org/wiki/Perfect_hash_function#Minimal_perfect_hash_function)---in Rust.



| ⚠️   | WIP! |
| --- | ---- |


## Note
This repo is in no way affiliated nor endorsed by the authors of RecSplit.


## Profiling
### Flamegraph
To use [cargo flamegraph](https://github.com/flamegraph-rs/flamegraph) on windows, use [dtrace_blondie](https://github.com/nico-abram/blondie/) instead of windows dtrace.
Therefore install blondie with `cargo install blondie` and set the `DTRACE` env var to the path to blondie:
`$env:DTRACE = "dtrace_blondie"`.
Then you can run flamegraph as usual, eg.g with `cargo flamegraph --unit-test -- <test_name>` in a admin terminal or with gsudo.

### asm view
To view the generated assembly, make sure a binary gets compiles (e.g. main) that includes the wanted code.
Then you can use `cargo-asm` to run `cargo asm --rust <function name>` to show the assembly of that function