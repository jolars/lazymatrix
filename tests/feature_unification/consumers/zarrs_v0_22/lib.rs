#[path = "../../src/zarrs.rs"]
mod shared;

/// Exercise this consumer's array after Cargo unifies adapter features.
pub fn check() {
    shared::check(zarrs::array::DataType::Float64, |array, values| {
        array
            .store_array_subset_elements(&array.subset_all(), values)
            .unwrap()
    });
}
