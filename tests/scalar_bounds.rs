use lazymatrix::Scalar;

fn require_scalar<T: Scalar>(value: T) -> T {
    value
}

// This must compile without adding Default or a static lifetime to T.
fn numeric_bounds_are_sufficient<T>(value: T) -> T
where
    T: num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + std::fmt::Debug,
{
    require_scalar(value)
}

#[test]
fn scalar_requires_only_numeric_bounds() {
    assert_eq!(numeric_bounds_are_sufficient(2.0_f64), 2.0);
    assert_eq!(numeric_bounds_are_sufficient(3.0_f32), 3.0);
}
