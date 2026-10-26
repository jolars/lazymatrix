//! Each consumer enables exactly one adapter through a normal dependency.

#[cfg(all(test, feature = "faer_v0_22"))]
#[test]
fn faer_v0_22_consumer() {
    consumer_faer_v0_22::check();
}

#[cfg(all(test, feature = "faer_v0_23"))]
#[test]
fn faer_v0_23_consumer() {
    consumer_faer_v0_23::check();
}

#[cfg(all(test, feature = "faer_v0_24"))]
#[test]
fn faer_v0_24_consumer() {
    consumer_faer_v0_24::check();
}

#[cfg(all(test, feature = "nalgebra_v0_32"))]
#[test]
fn nalgebra_v0_32_consumer() {
    consumer_nalgebra_v0_32::check();
}

#[cfg(all(test, feature = "nalgebra_v0_33"))]
#[test]
fn nalgebra_v0_33_consumer() {
    consumer_nalgebra_v0_33::check();
}

#[cfg(all(test, feature = "nalgebra_v0_34"))]
#[test]
fn nalgebra_v0_34_consumer() {
    consumer_nalgebra_v0_34::check();
}

#[cfg(all(test, feature = "nalgebra_v0_35"))]
#[test]
fn nalgebra_v0_35_consumer() {
    consumer_nalgebra_v0_35::check();
}

#[cfg(all(test, feature = "ndarray_v0_15"))]
#[test]
fn ndarray_v0_15_consumer() {
    consumer_ndarray_v0_15::check();
}

#[cfg(all(test, feature = "ndarray_v0_16"))]
#[test]
fn ndarray_v0_16_consumer() {
    consumer_ndarray_v0_16::check();
}

#[cfg(all(test, feature = "ndarray_v0_17"))]
#[test]
fn ndarray_v0_17_consumer() {
    consumer_ndarray_v0_17::check();
}

#[cfg(all(test, feature = "zarrs_v0_22"))]
#[test]
fn zarrs_v0_22_consumer() {
    consumer_zarrs_v0_22::check();
}

#[cfg(all(test, feature = "zarrs_v0_23"))]
#[test]
fn zarrs_v0_23_consumer() {
    consumer_zarrs_v0_23::check();
}
