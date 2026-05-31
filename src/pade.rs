//! Padé approximants: rational function approximation [m/n] of a function.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A Padé approximant R(x) = P(x)/Q(x) where P is degree m and Q is degree n.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadeApproximant {
    /// Numerator coefficients [p0, p1, ..., p_m].
    pub p: Vec<f64>,
    /// Denominator coefficients [q0=1, q1, ..., q_n].
    pub q: Vec<f64>,
}

impl PadeApproximant {
    /// Compute the [m/n] Padé approximant given Taylor coefficients of f at x=0.
    ///
    /// `taylor` should contain at least m+n+1 Taylor coefficients: [f(0), f'(0), f''(0)/2!, ...].
    pub fn from_taylor(taylor: &[f64], m: usize, n: usize) -> Self {
        let total = m + n + 1;
        assert!(
            taylor.len() >= total,
            "need at least {} Taylor coefficients, got {}",
            total,
            taylor.len()
        );

        // Build the system to find q1, ..., q_n
        // From the Padé equations:
        // For k = m+1, ..., m+n:
        //   c_k + sum_{j=1}^{n} q_j * c_{k-j} = 0
        // where c_i are the Taylor coefficients and we define q_0 = 1.
        //
        // Then: p_i = c_i + sum_{j=1}^{min(i,n)} q_j * c_{i-j} for i = 0, ..., m

        let mut q = vec![0.0; n + 1];
        q[0] = 1.0;

        if n > 0 {
            // Solve for q1..qn
            let mut mat = vec![0.0; n * n];
            let mut rhs = vec![0.0; n];

            for i in 0..n {
                let k = m + 1 + i;
                rhs[i] = -taylor[k];
                for j in 0..n {
                    let idx = k - (j + 1);
                    if idx >= 0 {
                        mat[i * n + j] = taylor[idx];
                    }
                }
            }

            // Solve n x n system
            let a = DMatrix::from_row_slice(n, n, &mat);
            let b = nalgebra::DVector::from_row_slice(&rhs);
            if let Some(sol) = a.lu().solve(&b) {
                for j in 0..n {
                    q[j + 1] = sol[j];
                }
            }
        }

        // Compute p
        let mut p = vec![0.0; m + 1];
        for i in 0..=m {
            p[i] = taylor[i];
            for j in 1..=n.min(i) {
                p[i] += q[j] * taylor[i - j];
            }
        }

        PadeApproximant { p, q }
    }

    /// Evaluate the Padé approximant at x.
    pub fn eval(&self, x: f64) -> f64 {
        let num = Self::eval_poly(&self.p, x);
        let den = Self::eval_poly(&self.q, x);
        if den.abs() < 1e-15 {
            f64::INFINITY * num.signum()
        } else {
            num / den
        }
    }

    fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
        let mut result = 0.0;
        let mut xp = 1.0;
        for &c in coeffs {
            result += c * xp;
            xp *= x;
        }
        result
    }
}

/// Compute [m/n] Padé approximant from Taylor coefficients.
pub fn pade_approximant(taylor: &[f64], m: usize, n: usize) -> PadeApproximant {
    PadeApproximant::from_taylor(taylor, m, n)
}

/// Compute Taylor coefficients of a function at x=0 using numerical differentiation.
pub fn numerical_taylor<F>(f: F, order: usize, h: f64) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let mut coeffs = Vec::with_capacity(order + 1);
    for k in 0..=order {
        // Use finite differences to compute f^(k)(0) / k!
        if k == 0 {
            coeffs.push(f(0.0));
        } else if k == 1 {
            coeffs.push((f(h) - f(-h)) / (2.0 * h));
        } else {
            // Use Cauchy's integral formula approximation or higher-order finite diff
            // Simple approach: f^(k)(0) ≈ k! * sum via central differences
            let coeff = taylor_coeff_numerical(&f, k, h);
            coeffs.push(coeff);
        }
    }
    coeffs
}

fn taylor_coeff_numerical<F>(f: &F, k: usize, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    // Use the formula: f^(k)(0)/k! ≈ sum_{j=0}^{k} C(k,j) * (-1)^(k-j) * f(j*h - k*h/2) / (k! * h^k)
    // Better: use the simple recursion with step h
    let n = k;
    let mut sum = 0.0;
    for j in 0..=n {
        let binom = comb(n, j) as f64;
        let sign = if (n - j) % 2 == 0 { 1.0 } else { -1.0 };
        let x = (j as f64 - n as f64 / 2.0) * h;
        sum += sign * binom * f(x);
    }
    sum / (h.powi(k as i32) * factorial(k) as f64)
}

fn comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k.min(n - k) {
        result = result * (n - i) / (i + 1);
    }
    result
}

fn factorial(n: usize) -> usize {
    (1..=n).product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_pade_exp() {
        // Taylor coefficients of e^x: [1, 1, 1/2, 1/6, 1/24, ...]
        let taylor: Vec<f64> = (0..7)
            .map(|k| 1.0 / factorial(k) as f64)
            .collect();
        // [3/3] Padé of e^x
        let pade = pade_approximant(&taylor, 3, 3);
        for x in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let expected = x.exp();
            let got = pade.eval(x);
            assert!(
                approx_eq(got, expected, 1e-2),
                "e^x at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_pade_sin() {
        // Taylor: sin(x) = x - x^3/6 + x^5/120 - ...
        let taylor: Vec<f64> = (0..7)
            .map(|k| {
                if k % 2 == 0 {
                    0.0
                } else {
                    (-1.0_f64).powi((k - 1) as i32 / 2) / factorial(k) as f64
                }
            })
            .collect();
        // [3/2] Padé of sin(x)
        let pade = pade_approximant(&taylor, 3, 2);
        for x in [-1.0_f64, 0.0, 0.5, 1.0] {
            let expected = x.sin();
            let got = pade.eval(x);
            assert!(
                approx_eq(got, expected, 1e-2),
                "sin(x) at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_pade_rational_exact() {
        // f(x) = 1/(1-x) has Taylor [1, 1, 1, 1, ...], Padé [0/1] should give exact
        let taylor = vec![1.0, 1.0, 1.0];
        let pade = pade_approximant(&taylor, 0, 1);
        assert!(approx_eq(pade.p[0], 1.0, 1e-10));
        assert!(approx_eq(pade.q[0], 1.0, 1e-10));
        assert!(approx_eq(pade.q[1], -1.0, 1e-10));
        // R(x) = 1/(1-x)
        assert!(approx_eq(pade.eval(0.5), 2.0, 1e-10));
    }

    #[test]
    fn test_pade_cos() {
        // Taylor: cos(x) = 1 - x^2/2 + x^4/24 - ...
        let taylor: Vec<f64> = (0..7)
            .map(|k| {
                if k % 2 == 1 {
                    0.0
                } else {
                    (-1.0_f64).powi(k as i32 / 2) / factorial(k) as f64
                }
            })
            .collect();
        // [2/2] Padé of cos(x)
        let pade = pade_approximant(&taylor, 2, 2);
        for x in [-1.0_f64, 0.0, 0.5, 1.0] {
            let expected = x.cos();
            let got = pade.eval(x);
            assert!(
                approx_eq(got, expected, 1e-2),
                "cos(x) at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_pade_at_zero() {
        let taylor = vec![1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
        let pade = pade_approximant(&taylor, 2, 2);
        // At x=0: P(0)/Q(0) = p[0]/q[0] = 1.0
        assert!(approx_eq(pade.eval(0.0), 1.0, 1e-10));
    }
}
