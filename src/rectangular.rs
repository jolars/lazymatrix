//! Rectangular execution preserves each logical parent's normalization.

use std::ops::Range;

use crate::normalization::pipeline::{Columns, Rows, Stage};
use crate::{
    ColumnNormalizationParams, EagerMatrix, LazyMatrix, MatrixErrorType, MatrixRead, MatrixShape,
    MatrixTraversal, NormalizationChainParams, NormalizationParams, RectangularProducts,
    RowNormalizationParams, Scalar, Submatrix, SubmatrixKernel, Transpose, VectorView,
    VectorViewMut,
};
use num_traits::Zero;

fn validate<F: Scalar, X: VectorView<F> + ?Sized, Y: VectorView<F> + ?Sized>(
    parent: &(impl MatrixShape + ?Sized),
    rows: &Range<usize>,
    columns: &Range<usize>,
    transpose: bool,
    x: &X,
    out: &Y,
) {
    crate::submatrix::validate_ranges(parent, rows, columns);
    let (input, output) = if transpose {
        (rows.len(), columns.len())
    } else {
        (columns.len(), rows.len())
    };
    assert_eq!(x.len(), input, "submatrix input dimension mismatch");
    assert_eq!(out.len(), output, "submatrix output dimension mismatch");
}

macro_rules! column_products {
    ($forward:ident, $transpose:expr, $kernel:ident) => {
        fn $forward<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: F,
            x: &X,
            beta: F,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<F> + ?Sized,
            Y: VectorViewMut<F> + ?Sized,
        {
            self.data().$kernel(
                rows,
                columns.clone(),
                self.normalization().centers().map(|c| &c[columns.clone()]),
                self.normalization().scales().map(|s| &s[columns]),
                alpha,
                x,
                beta,
                out,
            )
        }
    };
}

impl<M: SubmatrixKernel<F>, F: Scalar, S: AsRef<[F]>> RectangularProducts
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
{
    type Scalar = F;

    column_products!(rectangle_matvec_into, false, submatrix_matvec_into);
    column_products!(rectangle_transpose_into, true, submatrix_transpose_into);
}

macro_rules! row_products {
    ($method:ident, $transpose:expr) => {
        fn $method<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: F,
            x: &X,
            beta: F,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<F> + ?Sized,
            Y: VectorViewMut<F> + ?Sized,
        {
            crate::normalization::row::rectangle_product(
                self.data(),
                rows,
                columns,
                $transpose,
                alpha,
                x,
                beta,
                self.normalization().centers(),
                self.normalization().scales(),
                out,
            )
        }
    };
}

impl<M: MatrixTraversal<F>, F: Scalar, S: AsRef<[F]>> RectangularProducts
    for LazyMatrix<M, RowNormalizationParams<F, S>>
{
    type Scalar = F;

    row_products!(rectangle_matvec_into, false);
    row_products!(rectangle_transpose_into, true);
}

pub(crate) trait ChainRectangle<M: MatrixErrorType, P: Stage> {
    #[allow(clippy::too_many_arguments)]
    fn product<X: VectorView<P::Scalar> + ?Sized, Y: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        rows: Range<usize>,
        columns: Range<usize>,
        transpose: bool,
        alpha: P::Scalar,
        x: &X,
        beta: P::Scalar,
        out: &mut Y,
    ) -> Result<(), M::Error>;
}

impl<M, P> ChainRectangle<M, P> for Columns
where
    P: Stage<Kind = Columns>,
    M: MatrixTraversal<P::Scalar>,
{
    fn product<X: VectorView<P::Scalar> + ?Sized, Y: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        rows: Range<usize>,
        columns: Range<usize>,
        transpose: bool,
        alpha: P::Scalar,
        x: &X,
        beta: P::Scalar,
        out: &mut Y,
    ) -> Result<(), M::Error> {
        crate::normalization::pipeline::product_rectangle(
            data, params, rows, columns, transpose, alpha, x, beta, out,
        )
    }
}

impl<M, P> ChainRectangle<M, P> for Rows
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn product<X: VectorView<P::Scalar> + ?Sized, Y: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        rows: Range<usize>,
        columns: Range<usize>,
        transpose: bool,
        alpha: P::Scalar,
        x: &X,
        beta: P::Scalar,
        out: &mut Y,
    ) -> Result<(), M::Error> {
        validate(data, &rows, &columns, transpose, x, out);
        if alpha == P::Scalar::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        for o in 0..out.len() {
            let product = (0..x.len())
                .map(|k| {
                    let (i, j) = if transpose { (k, o) } else { (o, k) };
                    let row = rows.start + i;
                    let column = columns.start + j;
                    params.entry(data.get(row, column), row, column) * x.get(k)
                })
                .sum::<P::Scalar>();
            let value = alpha * product;
            out.set(
                o,
                if beta == P::Scalar::zero() {
                    value
                } else {
                    value + beta * out.get(o)
                },
            );
        }
        Ok(())
    }
}

macro_rules! chain_products {
    ($method:ident, $transpose:expr) => {
        fn $method<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: Self::Scalar,
            x: &X,
            beta: Self::Scalar,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<Self::Scalar> + ?Sized,
            Y: VectorViewMut<Self::Scalar> + ?Sized,
        {
            <NormalizationChainParams<A, B> as Stage>::Kind::product(
                self.data(),
                self.normalization(),
                rows,
                columns,
                $transpose,
                alpha,
                x,
                beta,
                out,
            )
        }
    };
}

impl<M, A, B> RectangularProducts for LazyMatrix<M, NormalizationChainParams<A, B>>
where
    M: MatrixShape + MatrixErrorType,
    NormalizationChainParams<A, B>: Stage,
    <NormalizationChainParams<A, B> as Stage>::Kind:
        ChainRectangle<M, NormalizationChainParams<A, B>>,
{
    type Scalar = <NormalizationChainParams<A, B> as NormalizationParams>::Scalar;

    chain_products!(rectangle_matvec_into, false);
    chain_products!(rectangle_transpose_into, true);
}

macro_rules! eager_products {
    ($method:ident, $kernel:ident) => {
        fn $method<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: Self::Scalar,
            x: &X,
            beta: Self::Scalar,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<Self::Scalar> + ?Sized,
            Y: VectorViewMut<Self::Scalar> + ?Sized,
        {
            self.data()
                .$kernel(rows, columns, None, None, alpha, x, beta, out)
        }
    };
}

impl<M, P> RectangularProducts for EagerMatrix<M, P>
where
    P: NormalizationParams,
    M: SubmatrixKernel<P::Scalar>,
{
    type Scalar = P::Scalar;

    eager_products!(rectangle_matvec_into, submatrix_matvec_into);
    eager_products!(rectangle_transpose_into, submatrix_transpose_into);
}

macro_rules! transpose_products {
    ($method:ident, $opposite:ident) => {
        fn $method<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: Self::Scalar,
            x: &X,
            beta: Self::Scalar,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<Self::Scalar> + ?Sized,
            Y: VectorViewMut<Self::Scalar> + ?Sized,
        {
            self.as_inner()
                .$opposite(columns, rows, alpha, x, beta, out)
        }
    };
}

impl<O: RectangularProducts + ?Sized> RectangularProducts for Transpose<'_, O> {
    type Scalar = O::Scalar;

    transpose_products!(rectangle_matvec_into, rectangle_transpose_into);
    transpose_products!(rectangle_transpose_into, rectangle_matvec_into);
}

macro_rules! restricted_products {
    ($method:ident) => {
        fn $method<X, Y>(
            &self,
            rows: Range<usize>,
            columns: Range<usize>,
            alpha: Self::Scalar,
            x: &X,
            beta: Self::Scalar,
            out: &mut Y,
        ) -> Result<(), Self::Error>
        where
            X: VectorView<Self::Scalar> + ?Sized,
            Y: VectorViewMut<Self::Scalar> + ?Sized,
        {
            crate::submatrix::validate_ranges(self, &rows, &columns);
            let parent_rows = self.row_range();
            let parent_columns = self.column_range();
            self.as_inner().$method(
                parent_rows.start + rows.start..parent_rows.start + rows.end,
                parent_columns.start + columns.start..parent_columns.start + columns.end,
                alpha,
                x,
                beta,
                out,
            )
        }
    };
}

impl<O: RectangularProducts + ?Sized> RectangularProducts for Submatrix<'_, O> {
    type Scalar = O::Scalar;

    restricted_products!(rectangle_matvec_into);
    restricted_products!(rectangle_transpose_into);
}

macro_rules! forward {
    ($reference:ty) => {
        impl<O: RectangularProducts + ?Sized> RectangularProducts for $reference {
            type Scalar = O::Scalar;
            fn rectangle_matvec_into<X, Y>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                alpha: Self::Scalar,
                x: &X,
                beta: Self::Scalar,
                out: &mut Y,
            ) -> Result<(), Self::Error>
            where
                X: VectorView<Self::Scalar> + ?Sized,
                Y: VectorViewMut<Self::Scalar> + ?Sized,
            {
                (**self).rectangle_matvec_into(rows, columns, alpha, x, beta, out)
            }
            fn rectangle_transpose_into<X, Y>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                alpha: Self::Scalar,
                x: &X,
                beta: Self::Scalar,
                out: &mut Y,
            ) -> Result<(), Self::Error>
            where
                X: VectorView<Self::Scalar> + ?Sized,
                Y: VectorViewMut<Self::Scalar> + ?Sized,
            {
                (**self).rectangle_transpose_into(rows, columns, alpha, x, beta, out)
            }
        }
    };
}
forward!(&O);
forward!(&mut O);
