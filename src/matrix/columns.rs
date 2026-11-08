use crate::{
    Columns, FittedRowNormalization, LazyMatrix, MatrixShape, NormalizationChainParams,
    NormalizationParams, NormalizedColumn, RawColumns, RowNormalizationParams, Scalar,
    SparseColumnRef, SparseColumns,
};

macro_rules! columns {
    ([$($generic:tt)*] $params:ty) => {
        impl<M: MatrixShape, $($generic)*> LazyMatrix<M, $params>
        where
            $params: NormalizationParams,
        {
            /// Borrow a logical column without copying storage or fitted parameters.
            ///
            /// Borrowing takes O(1) time and allocates nothing. The view applies
            /// every saved stage in order. Only natural raw column borrowing is
            /// required, including for explicit mixed chains on sparse storage.
            /// See [`NormalizedColumn`] for operation costs and numerical behavior.
            ///
            /// # Panics
            /// Panics if `j >= self.ncols()`.
            pub fn column(&self, j: usize) -> NormalizedColumn<'_, M::Column<'_>, $params>
            where
                M: RawColumns<<$params as NormalizationParams>::Scalar>,
            {
                assert!(j < self.ncols(), "column index out of bounds");
                NormalizedColumn::new(self.data().raw_column(j), self.normalization(), j)
            }

            /// Borrow a logical CSC column and its raw sparse representation.
            ///
            /// Borrowing takes O(1) time and allocates nothing. Stored indices
            /// and values preserve storage order, duplicates, and explicit zeros.
            /// The background can vary by row; inspect it with
            /// [`NormalizedColumn::implicit_value`]. Explicit mixed chains need
            /// no fitting or product capabilities.
            ///
            /// # Panics
            /// Panics if `j >= self.ncols()`.
            pub fn sparse_column(&self, j: usize) -> NormalizedColumn<
                '_, SparseColumnRef<'_, <$params as NormalizationParams>::Scalar>, $params,
            >
            where
                M: SparseColumns<<$params as NormalizationParams>::Scalar>,
            {
                assert!(j < self.ncols(), "column index out of bounds");
                let (indices, values) = self.data().sparse_column(j);
                NormalizedColumn::new(
                    SparseColumnRef::new(indices, values, self.nrows()),
                    self.normalization(),
                    j,
                )
            }
        }

        impl<M, $($generic)*> Columns<<$params as NormalizationParams>::Scalar>
            for LazyMatrix<M, $params>
        where
            $params: NormalizationParams,
            M: RawColumns<<$params as NormalizationParams>::Scalar>,
        {
            type Column<'a> = NormalizedColumn<'a, M::Column<'a>, $params>
            where
                Self: 'a;

            fn column(&self, j: usize) -> Self::Column<'_> {
                LazyMatrix::<M, $params>::column(self, j)
            }
        }
    };
}

columns!([F: Scalar, S: AsRef<[F]>] RowNormalizationParams<F, S>);
columns!([F: Scalar, S: AsRef<[F]>] FittedRowNormalization<RowNormalizationParams<F, S>>);
columns!([A, B] NormalizationChainParams<A, B>);
