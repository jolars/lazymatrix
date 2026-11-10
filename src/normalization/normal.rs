//! Standard-normal calibration without an additional core dependency.

pub(super) fn quantile_range(lower: f64, upper: f64) -> f64 {
    let width = upper - lower;
    if width <= 1e-7 * lower.min(1.0 - upper) {
        // The quantile derivative avoids cancellation for nearly equal bounds.
        // The midpoint error is quadratic in the relative probability width.
        let z = quantile(lower + width / 2.0);
        return (width.ln() + z * z / 2.0 + 0.9189385332046727).exp();
    }
    quantile(upper) - quantile(lower)
}

// Wichura's AS 241 rational approximations (Applied Statistics 37, 1988,
// pp. 477-484, https://doi.org/10.2307/2347330). Evaluate the smaller tail
// directly so subnormal lower probabilities remain usable.
fn quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        return q * ratio(
            0.180625 - q * q,
            &[
                3.3871328727963665,
                133.14166789178438,
                1971.5909503065513,
                13_731.693_765_509_46,
                45921.95393154987,
                67265.7709270087,
                33430.57558358813,
                2509.0809287301227,
            ],
            &[
                1.0,
                42.31333070160091,
                687.1870074920579,
                5394.196021424751,
                21213.794301586596,
                39307.89580009271,
                28729.085735721943,
                5226.495278852854,
            ],
        );
    }
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        ratio(
            r - 1.6,
            &[
                1.4234371107496835,
                4.630337846156546,
                5.769497221460691,
                3.6478483247632045,
                1.2704582524523684,
                0.2417807251774506,
                0.022723844989269184,
                0.0007745450142783414,
            ],
            &[
                1.0,
                2.053191626637759,
                1.6763848301838038,
                0.6897673349851,
                0.14810397642748008,
                0.015198666563616457,
                0.0005475938084995345,
                1.0507500716444169e-9,
            ],
        )
    } else {
        ratio(
            r - 5.0,
            &[
                6.657904643501103,
                5.463784911164114,
                1.7848265399172913,
                0.29656057182850487,
                0.026532189526576123,
                0.0012426609473880784,
                0.000027115555687434876,
                2.0103343992922881e-7,
            ],
            &[
                1.0,
                0.5998322065558879,
                0.1369298809227358,
                0.014875361290850615,
                0.0007868691311456133,
                0.000018463183175100547,
                1.421_511_758_316_446e-7,
                2.0442631033899397e-15,
            ],
        )
    };
    if q < 0.0 { -value } else { value }
}

fn ratio(x: f64, numerator: &[f64], denominator: &[f64]) -> f64 {
    let polynomial = |coefficients: &[f64]| {
        coefficients
            .iter()
            .rev()
            .fold(0.0, |sum, &coefficient| sum * x + coefficient)
    };
    polynomial(numerator) / polynomial(denominator)
}

#[cfg(test)]
mod tests {
    use super::{quantile, quantile_range};
    use approx::assert_relative_eq;

    #[test]
    fn normal_quantiles_match_reference_values() {
        for (probability, expected) in [
            (0.5, 0.0),
            (0.25, -0.6744897501960817),
            (0.1, -1.2815515655446004),
            (0.9, 1.2815515655446004),
            (0.975, 1.959963984540054),
            (1e-20, -9.262340089798409),
            (1e-100, -21.273453560965322),
            (1e-300, -37.0470962993612),
            (f64::from_bits(1), -38.467405617144344),
        ] {
            assert_relative_eq!(
                quantile(probability),
                expected,
                epsilon = 1e-14,
                max_relative = 1e-14
            );
        }
        assert_relative_eq!(
            quantile_range(0.25, 0.75),
            1.3489795003921634,
            epsilon = 1e-14
        );
    }

    #[test]
    fn adjacent_bounds_have_a_positive_calibration() {
        for lower in [0.1, 0.25, 0.5, 0.9, 1e-100, 1e-300, f64::from_bits(1)] {
            let upper = f64::from_bits(lower.to_bits() + 1);
            let range = quantile_range(lower, upper);
            assert!(
                range.is_finite() && range > 0.0,
                "range at {lower}: {range}"
            );
        }
        let width = f64::from_bits(0.5_f64.to_bits() + 1) - 0.5;
        assert_relative_eq!(
            quantile_range(0.5, 0.5 + width),
            width * (2.0 * std::f64::consts::PI).sqrt(),
            max_relative = 1e-14
        );
    }
}
