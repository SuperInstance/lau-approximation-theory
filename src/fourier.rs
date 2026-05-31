//! Trigonometric approximation: Fourier series truncation.

use serde::{Deserialize, Serialize};

/// A truncated Fourier series approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierApprox {
    /// Number of terms (2*N + 1 including a0).
    pub n_terms: usize,
    /// Coefficients: [a0, a1, b1, a2, b2, ..., a_N, b_N].
    pub coeffs: Vec<f64>,
    /// Period (assumes function has period T).
    pub period: f64,
}

impl FourierApprox {
    /// Compute truncated Fourier series of `f` on [0, T] with `n_terms` cosine+ sine pairs.
    /// Total coefficients: 2*n_terms + 1.
    pub fn fit<F>(f: F, period: f64, n_terms: usize, n_samples: usize) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let mut coeffs = Vec::with_capacity(2 * n_terms + 1);
        let omega = 2.0 * std::f64::consts::PI / period;

        // a0
        let mut a0 = 0.0;
        for k in 0..n_samples {
            let t = period * k as f64 / n_samples as f64;
            a0 += f(t);
        }
        a0 /= n_samples as f64;
        coeffs.push(a0);

        for n in 1..=n_terms {
            let mut an = 0.0;
            let mut bn = 0.0;
            for k in 0..n_samples {
                let t = period * k as f64 / n_samples as f64;
                let ft = f(t);
                an += ft * (omega * n as f64 * t).cos();
                bn += ft * (omega * n as f64 * t).sin();
            }
            an *= 2.0 / n_samples as f64;
            bn *= 2.0 / n_samples as f64;
            coeffs.push(an);
            coeffs.push(bn);
        }

        FourierApprox {
            n_terms,
            coeffs,
            period,
        }
    }

    /// Evaluate the truncated Fourier series at `t`.
    pub fn eval(&self, t: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI / self.period;
        let mut result = self.coeffs[0];
        for n in 1..=self.n_terms {
            let an = self.coeffs[2 * n - 1];
            let bn = self.coeffs[2 * n];
            result += an * (omega * n as f64 * t).cos() + bn * (omega * n as f64 * t).sin();
        }
        result
    }
}

/// Compute Fourier truncation: fit and return a FourierApprox.
pub fn fourier_truncate<F>(f: F, period: f64, n_terms: usize, n_samples: usize) -> FourierApprox
where
    F: Fn(f64) -> f64,
{
    FourierApprox::fit(f, period, n_terms, n_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_fourier_constant() {
        let approx = fourier_truncate(|_| 5.0, 2.0 * std::f64::consts::PI, 3, 100);
        for t in [0.0, 1.0, 2.0, 5.0] {
            assert!(
                approx_eq(approx.eval(t), 5.0, 1e-8),
                "at t={t}: got {}",
                approx.eval(t)
            );
        }
    }

    #[test]
    fn test_fourier_cosine() {
        // f(t) = cos(t), period = 2π
        let approx = fourier_truncate(
            |t| t.cos(),
            2.0 * std::f64::consts::PI,
            3,
            200,
        );
        for t in [0.0_f64, 0.5, 1.0, 2.0, 4.0] {
            let expected = t.cos();
            let got = approx.eval(t);
            assert!(
                approx_eq(got, expected, 1e-4),
                "at t={t}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_fourier_sine() {
        let approx = fourier_truncate(
            |t| t.sin(),
            2.0 * std::f64::consts::PI,
            3,
            200,
        );
        for t in [0.0_f64, 0.5, 1.0, 2.0, 4.0] {
            let expected = t.sin();
            let got = approx.eval(t);
            assert!(
                approx_eq(got, expected, 1e-4),
                "at t={t}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_fourier_square_wave() {
        // Square wave: f(t) = 1 for t in [0, π), -1 for t in [π, 2π)
        let sq = |t: f64| if (t % (2.0 * std::f64::consts::PI)) < std::f64::consts::PI { 1.0 } else { -1.0 };
        let approx = fourier_truncate(sq, 2.0 * std::f64::consts::PI, 10, 1000);
        // More terms should give better approximation at midpoints (away from discontinuity)
        let t = std::f64::consts::PI / 2.0;
        let got = approx.eval(t);
        assert!(
            approx_eq(got, 1.0, 0.1),
            "square wave at π/2: got {got}"
        );
    }

    #[test]
    fn test_fourier_more_terms_better() {
        let f = |t: f64| (3.0 * t).sin() + (t * 0.5).cos();
        let approx3 = fourier_truncate(f, 2.0 * std::f64::consts::PI, 3, 200);
        let approx10 = fourier_truncate(f, 2.0 * std::f64::consts::PI, 10, 200);

        let t = 1.0;
        let err3 = (f(t) - approx3.eval(t)).abs();
        let err10 = (f(t) - approx10.eval(t)).abs();
        assert!(
            err10 <= err3 + 1e-6,
            "more terms should not be worse: err3={err3}, err10={err10}"
        );
    }

    #[test]
    fn test_fourier_periodicity() {
        let approx = fourier_truncate(
            |t| t.sin() + (2.0 * t).cos(),
            2.0 * std::f64::consts::PI,
            5,
            200,
        );
        let t = 1.0;
        let v1 = approx.eval(t);
        let v2 = approx.eval(t + 2.0 * std::f64::consts::PI);
        assert!(approx_eq(v1, v2, 1e-8), "periodicity: {v1} vs {v2}");
    }
}
