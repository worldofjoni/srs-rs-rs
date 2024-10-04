
# Profiling
## Flamegraph
To use [cargo flamegraph](https://github.com/flamegraph-rs/flamegraph) on windows, use [dtrace_blondie](https://github.com/nico-abram/blondie/) instead of windows dtrace.
Therefore install blondie with `cargo install blondie` and set the `DTRACE` env var to the path to blondie:
`$env:DTRACE = "dtrace_blondie"`.
Then you can run flamegraph as usual, eg.g with `cargo flamegraph --unit-test --profile profile -- <test_name>` in a admin terminal or with sudo.

### Recursive Functions
The standart flamegraph for recursive functions is not very helpful because for every call, the stack gets larger and larger and thus these calls do not get unified in the flamegraph.
The following postprocesssing script removes recursive stack entries to get a more meaningful flamegraph:
`cargo flamegraph --post-process "wsl -e sed -re 's/(.*);(.*my_recrusive_function.*)/\2/g'" --unit-test -- <test_name>`

Replace `my_recursive_funtion` with your recursive function and `<test_name>` with your test to run on.

## asm view
To view the generated assembly, make sure a binary gets compiles (e.g. main) that includes the wanted code.
Then you can use `cargo-asm` to run `cargo asm --rust <function name>` to show the assembly of that function

Rebuilding the desired binary and/or cleaning the release directory may help if a function is not found. It also may be inlined aggressively. Search for the binary entry function.

## Graph generation
To generate a graph, run the according test using
`cargo test --release --all-features -- --ignored --nocapture <test_name>`


## Perf
- `cargo build --profile profile --tests`
- `perf record -g  target/profile/deps/srs-xxxx -- <test to test>`
- `perf report -i perf.data`
- convert for firefox profiler: `perf script -F +pid > ff.perf`


### Perf in WSL2
install `linux-tools` and use `/usr/lib/linux-tools/xxx-generic/perf`

TODO is this working correctly?