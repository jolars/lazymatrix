use super::{MatrixErrorType, MatrixOwned, MatrixShape, MatrixWrite, Scalar};

/// Read-only indexed access to dense, in-memory matrix storage.
///
/// Reading an entry takes O(1) time and performs no decoding or I/O. Storage
/// may be strided; this capability neither requires nor implies contiguity.
pub trait MatrixRead<F: Scalar>: MatrixShape {
    /// Read the entry at `(row, column)`.
    fn get(&self, row: usize, column: usize) -> F;
}

impl<M: MatrixRead<F> + ?Sized, F: Scalar> MatrixRead<F> for &M {
    fn get(&self, row: usize, column: usize) -> F {
        (**self).get(row, column)
    }
}
impl<M: MatrixRead<F> + ?Sized, F: Scalar> MatrixRead<F> for &mut M {
    fn get(&self, row: usize, column: usize) -> F {
        (**self).get(row, column)
    }
}
impl<F: Scalar> MatrixRead<F> for super::DenseBlock<'_, F> {
    fn get(&self, row: usize, column: usize) -> F {
        self.get(row, column)
    }
}

/// Batched product `out = A B`, overwriting reusable dense output.
///
/// For an `n × p` operator and a `p × k` input, output has shape `n × k`.
/// Inputs are borrowed without requiring cloning or contiguity. Every output
/// entry is overwritten, ignoring previous values. Implementations batch the
/// responses rather than calling the public vector product for each column.
/// Internal scratch is permitted; output reuse does not imply zero allocation.
/// No normalized design matrix may be materialized.
///
/// # Errors
/// Returns the operator's error. Output may be partial and must be discarded.
///
/// # Panics
/// Dimension mismatches panic before output writes or storage reads, even for
/// empty batches. A valid zero-column batch performs no storage reads.
pub trait MatMatInto<F: Scalar>: MatrixShape + MatrixErrorType {
    fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized;

    /// Apply optional column normalization without materializing the design.
    ///
    /// The default uses an indexed view of scaled coefficients and corrects
    /// the raw product only after success. Dense backends override this hook
    /// to center entries before accumulation, preserving small variations.
    /// Parameter lengths and exact zero scales are checked before writes or
    /// reads. Negative scales and nonfinite values are preserved.
    fn matmat_normalized_into<R, O>(
        &self,
        rhs: &R,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        crate::matmat::normalized(self, rhs, centers, scales, out)
    }
}

/// Batched transpose product `out = Aᵀ B`, overwriting dense output.
///
/// For an `n × p` operator and an `n × k` input, output has shape `p × k`.
/// Borrowing, workspace, errors, validation, and empty-batch behavior follow
/// [`MatMatInto`]. Transposition does not change storage orientation or grant
/// opposite-orientation borrowing capabilities.
pub trait MatTransposeMatInto<F: Scalar>: MatrixShape + MatrixErrorType {
    fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized;

    /// Apply the transpose of an optionally normalized operator.
    ///
    /// Validation and parameter semantics follow
    /// [`MatMatInto::matmat_normalized_into`]. Corrections run only after the
    /// raw product succeeds. Dense backends can center before accumulation.
    fn mat_transpose_mat_normalized_into<R, O>(
        &self,
        rhs: &R,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        crate::matmat::transpose_normalized(self, rhs, centers, scales, out)
    }
}

/// Allocating batched forward products with caller-selected dense storage.
///
/// Automatically available for every [`MatMatInto`] implementation. Select the
/// output backend through the return type, as with explicit eager conversion.
pub trait MatMat<F: Scalar>: MatMatInto<F> {
    fn matmat<R, O>(&self, rhs: &R) -> Result<O, Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixOwned<F> + MatrixRead<F>,
    {
        crate::matmat::validate_rhs(self, rhs, false);
        let mut out = O::zeros(self.nrows(), rhs.ncols());
        if rhs.ncols() != 0 {
            self.matmat_into(rhs, &mut out)?;
        }
        Ok(out)
    }
}
impl<M: MatMatInto<F> + ?Sized, F: Scalar> MatMat<F> for M {}

/// Allocating batched transpose products with caller-selected dense storage.
///
/// Automatically available for every [`MatTransposeMatInto`] implementation.
pub trait MatTransposeMat<F: Scalar>: MatTransposeMatInto<F> {
    fn mat_transpose_mat<R, O>(&self, rhs: &R) -> Result<O, Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixOwned<F> + MatrixRead<F>,
    {
        crate::matmat::validate_rhs(self, rhs, true);
        let mut out = O::zeros(self.ncols(), rhs.ncols());
        if rhs.ncols() != 0 {
            self.mat_transpose_mat_into(rhs, &mut out)?;
        }
        Ok(out)
    }
}
impl<M: MatTransposeMatInto<F> + ?Sized, F: Scalar> MatTransposeMat<F> for M {}

macro_rules! forward {
    ($reference:ty, $trait:ident, $product:ident, $normalized:ident) => {
        impl<M: $trait<F> + ?Sized, F: Scalar> $trait<F> for $reference {
            fn $product<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                (**self).$product(rhs, out)
            }
            fn $normalized<R, O>(
                &self,
                rhs: &R,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                (**self).$normalized(rhs, centers, scales, out)
            }
        }
    };
}
forward!(&M, MatMatInto, matmat_into, matmat_normalized_into);
forward!(&mut M, MatMatInto, matmat_into, matmat_normalized_into);
forward!(
    &M,
    MatTransposeMatInto,
    mat_transpose_mat_into,
    mat_transpose_mat_normalized_into
);
forward!(
    &mut M,
    MatTransposeMatInto,
    mat_transpose_mat_into,
    mat_transpose_mat_normalized_into
);
