//! Shared row-normalization kernels and backend adapters.

use crate::{MatrixRead, MatrixShape, Scalar, VectorView, VectorViewMut};

pub(crate) fn validate<F: Scalar, X: VectorView<F> + ?Sized, Y: VectorView<F> + ?Sized>(
    matrix: &(impl MatrixShape + ?Sized),
    transpose: bool,
    x: &X,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &Y,
) {
    let (input, output) = if transpose {
        (matrix.nrows(), matrix.ncols())
    } else {
        (matrix.ncols(), matrix.nrows())
    };
    assert_eq!(x.len(), input, "row-normalized input dimension mismatch");
    assert_eq!(
        out.len(),
        output,
        "row-normalized output dimension mismatch"
    );
    validate_parameters(matrix.nrows(), centers, scales);
}

pub(crate) fn validate_parameters<F: Scalar>(
    nrows: usize,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) {
    if let Some(centers) = centers {
        assert_eq!(centers.len(), nrows, "centers length must equal nrows");
    }
    if let Some(scales) = scales {
        assert_eq!(scales.len(), nrows, "scales length must equal nrows");
        for (row, &scale) in scales.iter().enumerate() {
            assert!(scale != F::zero(), "scale at row {row} must be nonzero");
        }
    }
}

pub(crate) fn dense_product<M, F, X, Y>(
    matrix: &M,
    transpose: bool,
    x: &X,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut Y,
) where
    M: MatrixRead<F> + ?Sized,
    F: Scalar,
    X: VectorView<F> + ?Sized,
    Y: VectorViewMut<F> + ?Sized,
{
    validate(matrix, transpose, x, centers, scales, out);
    for o in 0..out.len() {
        let value = (0..x.len())
            .map(|k| {
                let (i, j) = if transpose { (k, o) } else { (o, k) };
                crate::materialize::normalized(matrix.get(i, j), i, centers, scales) * x.get(k)
            })
            .sum();
        out.set(o, value);
    }
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
macro_rules! dense_stats {
    () => {
        fn row_normalization_stats(
            &self,
            spec: crate::RowNormalization,
        ) -> Result<crate::RowNormalizationStats<F>, Self::Error> {
            Ok(self.dense_row_normalization_stats(spec))
        }
    };
}
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) use dense_stats;

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
macro_rules! dense_products {
    () => {
        fn row_normalized_matvec_into<
            X: crate::VectorView<F> + ?Sized,
            Y: crate::VectorViewMut<F> + ?Sized,
        >(
            &self,
            x: &X,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
            out: &mut Y,
        ) -> Result<(), Self::Error> {
            self.dense_row_normalized_product_into(false, x, centers, scales, out)
        }

        fn row_normalized_transpose_into<
            X: crate::VectorView<F> + ?Sized,
            Y: crate::VectorViewMut<F> + ?Sized,
        >(
            &self,
            x: &X,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
            out: &mut Y,
        ) -> Result<(), Self::Error> {
            self.dense_row_normalized_product_into(true, x, centers, scales, out)
        }
    };
}
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) use dense_products;

#[cfg(feature = "faer_all")]
macro_rules! dense {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::RowStats<F> for $matrix {
            crate::row_normalization::dense_stats!();
        }
        impl<F: crate::Scalar> crate::RowNormalizationKernel<F> for $matrix {
            crate::row_normalization::dense_products!();
        }
    };
}
#[cfg(feature = "faer_all")]
pub(crate) use dense;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
macro_rules! sparse_stats {
    () => {
        fn row_normalization_stats(
            &self,
            spec: crate::RowNormalization,
        ) -> Result<crate::RowNormalizationStats<F>, Self::Error> {
            crate::normalization::row::traversal_stats(self, spec)
        }
    };
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use sparse_stats;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
macro_rules! sparse_products {
    () => {
        fn row_normalized_matvec_into<
            X: crate::VectorView<F> + ?Sized,
            Y: crate::VectorViewMut<F> + ?Sized,
        >(
            &self,
            x: &X,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
            out: &mut Y,
        ) -> Result<(), Self::Error> {
            crate::normalization::row::traversal_product(self, false, x, centers, scales, out)
        }

        fn row_normalized_transpose_into<
            X: crate::VectorView<F> + ?Sized,
            Y: crate::VectorViewMut<F> + ?Sized,
        >(
            &self,
            x: &X,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
            out: &mut Y,
        ) -> Result<(), Self::Error> {
            crate::normalization::row::traversal_product(self, true, x, centers, scales, out)
        }
    };
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use sparse_products;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
macro_rules! sparse {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::RowStats<F> for $matrix {
            crate::row_normalization::sparse_stats!();
        }
        impl<F: crate::Scalar> crate::RowNormalizationKernel<F> for $matrix {
            crate::row_normalization::sparse_products!();
        }
    };
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) use sparse;
