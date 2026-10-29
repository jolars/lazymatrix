//! Normalize a filesystem Zarr array by scanning one chunk at a time.
//!
//! Run with `cargo run --release --example zarrs_chunked --features zarrs -- [rows] [cols]`.
//! The matrix is generated chunk by chunk; working vectors remain in RAM.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use lazymatrix::{
    Centering, ColumnNormalization, LazyMatrix, MatTransposeVecInto, MatVecInto, Scaling,
    ZarrMatrix,
};
use zarrs::array::{Array, ArrayBuilder};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ListableStorageTraits;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let nrows: usize = args
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(10_000);
    let ncols: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(32);
    if args.next().is_some() {
        return Err("usage: zarrs_chunked [rows] [cols]".into());
    }
    let directory = tempfile::tempdir()?;
    let store = Arc::new(FilesystemStore::new(directory.path())?);
    let chunks = [1024, 8];
    let array = ArrayBuilder::new(
        vec![nrows as u64, ncols as u64],
        chunks.to_vec(),
        backend_aliases::zarrs_compat::float64(),
        0.0f64,
    )
    .build(store.clone(), "/matrix")?;
    array.store_metadata()?;
    for i in 0..array.chunk_grid_shape()[0] {
        for j in 0..array.chunk_grid_shape()[1] {
            let mut values = vec![0.0; (chunks[0] * chunks[1]) as usize];
            for row in 0..chunks[0] {
                for col in 0..chunks[1] {
                    let global_row = i * chunks[0] + row;
                    let global_col = j * chunks[1] + col;
                    values[(row * chunks[1] + col) as usize] =
                        ((global_row % 101 + 17 * (global_col % 101)) % 101) as f64 / 50.0;
                }
            }
            backend_aliases::zarrs_compat::store_chunk(&array, &[i, j], &values)?;
        }
    }
    println!(
        "Matrix: {nrows} x {ncols}, chunks: {} x {}, logical data: {} bytes",
        chunks[0],
        chunks[1],
        nrows as u128 * ncols as u128 * 8
    );
    println!("Stored data and metadata: {} bytes", store.size()?);
    drop(array);
    let array = Array::open(store, "/matrix")?;
    let matrix = ZarrMatrix::<_, f64>::try_new(array)?;
    let start = Instant::now();
    let lazy = LazyMatrix::new(
        matrix,
        ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
    )?;
    println!("ColumnNormalization: {:?}", start.elapsed());
    let v = vec![1.0; ncols];
    let mut y = vec![0.0; nrows];
    let mut z = vec![0.0; ncols];
    let start = Instant::now();
    lazy.matvec_into(&v, &mut y)?;
    println!(
        "Forward: {:?}, sum: {}",
        start.elapsed(),
        y.iter().sum::<f64>()
    );
    let start = Instant::now();
    lazy.mat_transpose_vec_into(&y, &mut z)?;
    println!(
        "Transpose: {:?}, sum: {}",
        start.elapsed(),
        z.iter().sum::<f64>()
    );
    Ok(())
}
