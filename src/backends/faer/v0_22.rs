use crate::faer_0_22 as faer;
use crate::faer_traits_0_22 as faer_traits;

#[path = "transpose_legacy.rs"]
mod transpose;

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
