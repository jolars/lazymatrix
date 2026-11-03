//! Native backend traversal preserves selected entries and storage order.

#[cfg(feature = "sprs_all")]
use lazymatrix::MatrixTraversal;

#[cfg(feature = "sprs_all")]
#[test]
fn native_sparse_traversal_preserves_selected_explicit_zeros() {
    for csc in [true, false] {
        let indptr = vec![0_usize, 2, 4, 5];
        let indices = vec![0_usize, 2, 1, 2, 1];
        let values = vec![99., 99., 0., 7., 99.];
        let matrix = if csc {
            sprs::CsMat::new_csc((3, 3), indptr, indices, values)
        } else {
            sprs::CsMat::new((3, 3), indptr, indices, values)
        };
        let mut visited = Vec::new();
        let (rows, columns) = if csc { (1..3, 1..2) } else { (1..2, 1..3) };
        matrix
            .for_each_entry(rows, columns, |i, j, v| visited.push((i, j, v)))
            .unwrap();
        assert_eq!(
            visited,
            if csc {
                vec![(1, 1, 0.), (2, 1, 7.)]
            } else {
                vec![(1, 1, 0.), (1, 2, 7.)]
            }
        );
    }
}
