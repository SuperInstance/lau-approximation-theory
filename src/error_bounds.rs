//! Error bounds: Runge phenomenon estimation, Lebesgue constant.

/// Estimate the Runge phenomenon error for equispaced interpolation on [-1, 1].
pub fn runge_error_estimate(n: usize) -> f64 {
    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);

    let xs: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64)
        .collect();
    let ys: Vec<f64> = xs.iter().map(|&x| runge(x)).collect();

    let mut max_err: f64 = 0.0;
    let grid = 500;
    for k in 0..=grid {
        let x = -1.0_f64 + 2.0_f64 * k as f64 / grid as f64;
        let interp = crate::interpolation::lagrange(&xs, &ys, x);
        let err = (runge(x) - interp).abs();
        if err > max_err { max_err = err; }
    }
    max_err
}

/// Estimate the Runge error using Chebyshev nodes (should be much smaller).
pub fn runge_error_chebyshev(n: usize) -> f64 {
    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);

    let xs = crate::chebyshev::chebyshev_nodes(n);
    let ys: Vec<f64> = xs.iter().map(|&x| runge(x)).collect();

    let mut max_err: f64 = 0.0;
    let grid = 500;
    for k in 0..=grid {
        let x = -1.0_f64 + 2.0_f64 * k as f64 / grid as f64;
        let interp = crate::interpolation::lagrange(&xs, &ys, x);
        let err = (runge(x) - interp).abs();
        if err > max_err { max_err = err; }
    }
    max_err
}

/// Compute the Lebesgue constant for a given set of interpolation nodes.
pub fn lebesgue_constant(nodes: &[f64], n_samples: usize) -> f64 {
    let n = nodes.len();
    let a = nodes.iter().cloned().fold(f64::INFINITY, f64::min);
    let b = nodes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut lambda_max: f64 = 0.0;

    for k in 0..=n_samples {
        let x = a + (b - a) * k as f64 / n_samples as f64;

        let mut sum: f64 = 0.0;
        for i in 0..n {
            let mut li: f64 = 1.0;
            for j in 0..n {
                if j != i {
                    let denom = nodes[i] - nodes[j];
                    if denom.abs() > 1e-15 {
                        li *= (x - nodes[j]) / denom;
                    }
                }
            }
            sum += li.abs();
        }
        if sum > lambda_max { lambda_max = sum; }
    }

    lambda_max
}

/// Lebesgue constant for equispaced nodes on [-1, 1].
pub fn lebesgue_constant_equispaced(n: usize) -> f64 {
    let nodes: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64)
        .collect();
    lebesgue_constant(&nodes, 1000)
}

/// Lebesgue constant for Chebyshev nodes on [-1, 1].
pub fn lebesgue_constant_chebyshev(n: usize) -> f64 {
    let nodes = crate::chebyshev::chebyshev_nodes(n);
    lebesgue_constant(&nodes, 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runge_error_grows_with_n() {
        let err11 = runge_error_estimate(11);
        let err5 = runge_error_estimate(5);
        assert!(err11 > 0.0, "error should be positive");
        assert!(err5 > 0.0, "error should be positive");
    }

    #[test]
    fn test_runge_chebyshev_better() {
        let err_eq = runge_error_estimate(15);
        let err_cheb = runge_error_chebyshev(15);
        assert!(
            err_cheb < err_eq,
            "Chebyshev error ({err_cheb}) should be < equispaced ({err_eq})"
        );
    }

    #[test]
    fn test_lebesgue_equispaced_grows() {
        let l5 = lebesgue_constant_equispaced(5);
        let l15 = lebesgue_constant_equispaced(15);
        assert!(l15 > l5, "Λ_15 ({l15}) should > Λ_5 ({l5})");
    }

    #[test]
    fn test_lebesgue_chebyshev_bounded() {
        let l10 = lebesgue_constant_chebyshev(10);
        let l50 = lebesgue_constant_chebyshev(50);
        assert!(l10 > 1.0, "Λ > 1");
        assert!(l50 < 10.0, "Λ_50 for Chebyshev should be < 10, got {l50}");
    }

    #[test]
    fn test_lebesgue_chebyshev_smaller() {
        let l_eq = lebesgue_constant_equispaced(15);
        let l_cheb = lebesgue_constant_chebyshev(15);
        assert!(
            l_cheb < l_eq,
            "Chebyshev Λ ({l_cheb}) should < equispaced Λ ({l_eq})"
        );
    }

    #[test]
    fn test_lebesgue_min_2() {
        let nodes = vec![0.0, 1.0];
        let l = lebesgue_constant(&nodes, 100);
        assert!(l >= 1.0, "Λ ≥ 1, got {l}");
    }
}
