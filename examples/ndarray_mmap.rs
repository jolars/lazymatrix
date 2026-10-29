//! Normalize a memory-mapped ndarray without allocating an owned matrix.
//!
//! Run with `cargo run --release --example ndarray_mmap --features ndarray -- [rows] [cols]`.
//! The temporary file and all mappings remain private to this process. Working
//! vectors must fit in RAM; the OS manages residency of the matrix's pages.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use std::error::Error;
use std::io::Seek;
use std::time::Instant;

use lazymatrix::{Centering, LazyMatrix, MatTransposeVecInto, MatVecInto, Normalization, Scaling};
use memmap2::MmapMut;
use ndarray::{Array1, ArrayView2, ArrayViewMut2};
use ndarray_npy::npy::header::{Header, Layout};
use ndarray_npy::{ViewMutNpyExt, ViewNpyExt, WritableElement};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let nrows: usize = args
        .next()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(10_000);
    let ncols: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(32);
    if args.next().is_some() {
        return Err("usage: ndarray_mmap [rows] [cols]".into());
    }
    let mut file = tempfile::tempfile()?;
    // A Fortran-order file lets column statistics read contiguous pages.
    Header {
        type_descriptor: f64::type_descriptor(),
        layout: Layout::Fortran,
        shape: vec![nrows, ncols],
    }
    .write(&mut file)?;
    let bytes = nrows
        .checked_mul(ncols)
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or("matrix is too large to map")?;
    let file_len = file
        .stream_position()?
        .checked_add(u64::try_from(bytes)?)
        .ok_or("backing file length overflow")?;
    file.set_len(file_len)?;
    // The file is private, and no other handle accesses its contents while mapped.
    let mut mapping = unsafe { MmapMut::map_mut(&file)? };
    {
        let mut view = ArrayViewMut2::<f64>::view_mut_npy(&mut mapping)?;
        for j in 0..ncols {
            for i in 0..nrows {
                view[(i, j)] = ((i % 101 + 17 * (j % 101)) % 101) as f64 / 50.0;
            }
        }
    }
    mapping.flush()?;
    let mapping = mapping.make_read_only()?;
    let view = ArrayView2::<f64>::view_npy(&mapping)?;
    println!(
        "Matrix: {nrows} x {ncols}, backing file: {} bytes",
        file.metadata()?.len()
    );
    let start = Instant::now();
    let lazy = LazyMatrix::new(
        view,
        Normalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
    )?;
    println!("Normalization: {:?}", start.elapsed());
    let v = Array1::ones(ncols);
    let mut y = Array1::zeros(nrows);
    let mut z = Array1::zeros(ncols);
    let start = Instant::now();
    lazy.matvec_into(&v, &mut y)?;
    println!("Forward: {:?}, sum: {}", start.elapsed(), y.sum());
    let start = Instant::now();
    lazy.mat_transpose_vec_into(&y, &mut z)?;
    println!("Transpose: {:?}, sum: {}", start.elapsed(), z.sum());
    Ok(())
}
