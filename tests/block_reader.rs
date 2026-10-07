use lazymatrix::{DenseBlock, MatrixShape};

#[test]
fn packed_blocks_expose_dimensions_and_values() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let block = DenseBlock::new(&values, 2, 3);
    assert_eq!((block.nrows(), block.ncols()), (2, 3));
    assert_eq!(block.values(), &values);
    assert_eq!(block.get(1, 2), 6.0);
    assert!(DenseBlock::<f64>::new(&[], 0, 3).values().is_empty());
}

#[test]
#[should_panic(expected = "block dimensions")]
fn block_dimensions_must_match_values() {
    DenseBlock::new(&[1.0], 2, 3);
}

#[test]
#[should_panic(expected = "block dimensions overflow")]
fn block_dimensions_cannot_overflow() {
    DenseBlock::<f64>::new(&[], usize::MAX, 2);
}

#[test]
#[should_panic(expected = "block index")]
fn block_index_checks_each_dimension() {
    DenseBlock::new(&[1.0, 2.0, 3.0, 4.0], 2, 2).get(0, 2);
}
