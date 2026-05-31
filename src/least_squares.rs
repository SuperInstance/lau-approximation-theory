//! Least squares approximation: linear and polynomial.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Linear least squares: fit y = a + b*x to data.
/// Returns (intercept, slope).
pub fn linear_least_squares(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        // All x are the same
        return (sum_y / n, 0.0);
    }

    let b = (n * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n;
    (a, b)
}

/// Polynomial least squares: fit y = c0 + c1*x + c2*x^2 + ... + c_degree*x^degree.
/// Returns coefficients [c0, c1, ..., c_degree].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialFit {
    /// Coefficients [c0, c1, ..., c_degree].
    pub coeffs: Vec<f64>,
    /// Degree of the polynomial.
    pub degree: usize,
}

impl PolynomialFit {
    /// Fit a polynomial of given degree to data using least squares.
    pub fn fit(xs: &[f64], ys: &[f64], degree: usize) -> Self {
        assert_eq!(xs.len(), ys.len());
        let n = xs.len();
        let m = degree + 1;

        // Build Vandermonde-like matrix A where A[i][j] = xs[i]^j
        let mut a_data = vec![0.0; n * m];
        for i in 0..n {
            let mut val = 1.0;
            for j in 0..m {
                a_data[i * m + j] = val;
                val *= xs[i];
            }
        }

        let a = DMatrix::from_row_slice(n, m, &a_data);
        let b = DVector::from_row_slice(ys);

        // Solve normal equations: A^T A x = A^T b
        let ata = &a.transpose() * &a;
        let atb = &a.transpose() * &b;

        let coeffs_vec = ata
            .lu()
            .solve(&atb)
            .expect("Least squares system is singular");

        PolynomialFit {
            coeffs: coeffs_vec.iter().copied().collect(),
            degree,
        }
    }

    /// Evaluate the fitted polynomial at x.
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut xp = 1.0;
        for &c in &self.coeffs {
            result += c * xp;
            xp *= x;
        }
        result
    }

    /// R² (coefficient of determination) on given data.
    pub fn r_squared(&self, xs: &[f64], ys: &[f64]) -> f64 {
        let mean_y: f64 = ys.iter().sum::<f64>() / ys.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for i in 0..xs.len() {
            let pred = self.eval(xs[i]);
            ss_res += (ys[i] - pred).powi(2);
            ss_tot += (ys[i] - mean_y).powi(2);
        }
        if ss_tot < 1e-15 {
            return 1.0;
        }
        1.0 - ss_res / ss_tot
    }
}

/// Convenience: polynomial least squares returning coefficients.
pub fn polynomial_least_squares(xs: &[f64], ys: &[f64], degree: usize) -> Vec<f64> {
    PolynomialFit::fit(xs, ys, degree).coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_linear_perfect() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let (a, b) = linear_least_squares(&xs, &ys);
        assert!(approx_eq(a, 1.0, 1e-10), "intercept: {a}");
        assert!(approx_eq(b, 2.0, 1e-10), "slope: {b}");
    }

    #[test]
    fn test_linear_noisy() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.1, 2.9, 5.1, 6.9, 9.1]; // ~y = 2x + 1
        let (a, b) = linear_least_squares(&xs, &ys);
        assert!(approx_eq(a, 1.1, 0.2));
        assert!(approx_eq(b, 2.0, 0.1));
    }

    #[test]
    fn test_polynomial_linear() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x + 2.0).collect();
        let fit = PolynomialFit::fit(&xs, &ys, 1);
        assert!(approx_eq(fit.coeffs[0], 2.0, 1e-10));
        assert!(approx_eq(fit.coeffs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_polynomial_quadratic() {
        let xs: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x - 2.0 * x + 1.0).collect();
        let fit = PolynomialFit::fit(&xs, &ys, 2);
        assert!(approx_eq(fit.coeffs[0], 1.0, 1e-8), "c0={}", fit.coeffs[0]);
        assert!(approx_eq(fit.coeffs[1], -2.0, 1e-8), "c1={}", fit.coeffs[1]);
        assert!(approx_eq(fit.coeffs[2], 1.0, 1e-8), "c2={}", fit.coeffs[2]);
    }

    #[test]
    fn test_polynomial_r_squared() {
        let xs: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let fit = PolynomialFit::fit(&xs, &ys, 2);
        let r2 = fit.r_squared(&xs, &ys);
        assert!(approx_eq(r2, 1.0, 1e-10), "R² = {r2}");
    }

    #[test]
    fn test_polynomial_overfit() {
        // Degree = n-1 should interpolate exactly
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 0.5, 2.0, 0.0];
        let fit = PolynomialFit::fit(&xs, &ys, 3);
        for i in 0..xs.len() {
            assert!(
                approx_eq(fit.eval(xs[i]), ys[i], 1e-8),
                "point {i}: expected {}, got {}",
                ys[i],
                fit.eval(xs[i])
            );
        }
    }

    #[test]
    fn test_constant_fit() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![5.0, 5.0, 5.0];
        let fit = PolynomialFit::fit(&xs, &ys, 0);
        assert!(approx_eq(fit.coeffs[0], 5.0, 1e-10));
    }
}
