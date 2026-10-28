use crate::nalgebra_0_33 as nalgebra;
use crate::nalgebra_sparse_0_10 as nalgebra_sparse;
use nalgebra::{ClosedAddAssign, ClosedMulAssign};

#[path = "csr.rs"]
mod csr;
#[path = "dense.rs"]
mod dense;
#[path = "gram.rs"]
mod gram;
#[path = "matmat.rs"]
mod matmat;
#[path = "sparse.rs"]
mod sparse;
#[path = "vector.rs"]
mod vector;
