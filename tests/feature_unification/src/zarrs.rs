#[path = "oracle.rs"]
mod oracle;

pub fn check(
    data_type: zarrs::array::DataType,
    store: impl FnOnce(&zarrs::array::Array<zarrs::storage::store::MemoryStore>, &[f64]),
) {
    use lazymatrix::{ZarrMatrix, ZarrMatrixError};
    use std::sync::Arc;
    use zarrs::array::{Array, ArrayBuilder};
    use zarrs::storage::store::MemoryStore;
    let array = ArrayBuilder::new(vec![3, 2], vec![2, 2], data_type, 5.0_f64)
        .build(Arc::new(MemoryStore::new()), "/matrix")
        .unwrap();
    store(&array, &[1.0, 0.0, 2.0, 3.0, 0.0, 6.0]);
    let matrix: ZarrMatrix<Array<MemoryStore>> = ZarrMatrix::try_new(array).unwrap();
    let array: Array<MemoryStore> = matrix.into_inner();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    use lazymatrix::{MatTransposeVec, MatVec, RowNormalization};
    assert_eq!(
        matrix.matvec(&vec![2.0, -1.0]).unwrap(),
        vec![2.0, 1.0, -6.0]
    );
    let rows = RowNormalization::min_max().fit_apply(&matrix).unwrap();
    assert_eq!(rows.normalization().centers().unwrap(), [0., 2., 0.]);
    assert_eq!(rows.normalization().scales().unwrap(), [1., 1., 6.]);
    assert_eq!(rows.matvec(&vec![2., -1.]).unwrap(), [2., -1., -1.]);
    assert_eq!(
        rows.mat_transpose_vec(&vec![1., 2., -1.]).unwrap(),
        [1., 1.]
    );
    oracle::check(
        matrix,
        vec![2.0, -1.0],
        vec![1.0, 2.0, -1.0],
        vec![0.0; 3],
        vec![0.0; 2],
    );
    use std::error::Error;
    let error = ZarrMatrixError::from(zarrs::array::ArrayError::InvalidChunkGridIndicesError(
        vec![9, 9],
    ));
    assert!(error.source().unwrap().is::<zarrs::array::ArrayError>());
}
