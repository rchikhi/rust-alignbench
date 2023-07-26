# Runtime analysis

Install `criterion`: `cargo install cargo-criterion`

Then run:
    
    `cargo criterion --message-format=json | tee bench_results.json` 

to export benchmarking runtime results. They are subsequently plotted in `notebook.ipynb`. 
