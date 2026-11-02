//! Peak additional live Rust bytes, excluding already constructed source storage.

use std::hint::black_box;
#[path = "common/allocations.rs"]
mod allocations;
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/normalization.rs"]
mod normalization;
#[path = "common/pipeline.rs"]
mod pipeline;

fn main() {
    println!("rows,cols,density,orientation,lazy_peak_bytes,eager_peak_bytes");
    for fixture in pipeline::fixtures() {
        for (orientation, source) in [
            ("csc", fixture.sparse.clone()),
            ("csr", fixture.sparse.to_csr()),
        ] {
            let lazy = allocations::measure(|| {
                let op = pipeline::columns().fit_apply(black_box(&source)).unwrap();
                let products = pipeline::pair(&op, &fixture);
                (op, products)
            });
            let eager = allocations::measure(|| {
                let op = pipeline::eager_columns(black_box(&source));
                let products = pipeline::pair(&op, &fixture);
                (op, products)
            });
            println!(
                "{},{},{},{orientation},{lazy},{eager}",
                fixture.rows, fixture.cols, fixture.density
            );
        }
        let source = pipeline::dense(&fixture.sparse);
        let lazy = allocations::measure(|| {
            let op = pipeline::mixed().fit_apply(black_box(&source)).unwrap();
            let products = pipeline::pair(&op, &fixture);
            (op, products)
        });
        let eager = allocations::measure(|| {
            let op = pipeline::eager_mixed(black_box(&source));
            let products = pipeline::pair(&op, &fixture);
            (op, products)
        });
        println!(
            "{},{},{},dense_mixed,{lazy},{eager}",
            fixture.rows, fixture.cols, fixture.density
        );
    }
}
