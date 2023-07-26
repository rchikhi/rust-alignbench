# rust-alignbench

A pairwise alignment benchmark of a few libraries:
* Rust-bio
* Parasailors: https://github.com/anp/parasailors
* Block_aligner: https://github.com/Daniel-Liu-c0deb0t/block-aligner
* Libwfa: https://github.com/chfi/rs-wfa/
* Wfa2: https://github.com/tanghaibao/rust-wfa2
* ksw2: https://github.com/lh3/ksw2
* Scrooge: https://github.com/CMU-SAFARI/Scrooge
 
## Running

Speed test: `cargo bench`

Alignment test: `cargo test -- --show-output`

## Speed results

![](analysis/out.png)
