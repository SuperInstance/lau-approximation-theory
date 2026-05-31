//! Cubic spline interpolation with natural and clamped boundary conditions.

use serde::{Deserialize, Serialize};

/// Boundary condition for cubic spline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplineBoundary {
    /// Natural spline: S''(a) = S''(b) = 0.
    Natural,
    /// Clamped spline: S'(a) = fpa, S'(b) = fpb.
    Clamped { fpa: f64, fpb: f64 },
}

/// A cubic spline interpolant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubicSpline {
    /// Knots (must be sorted).
    pub xs: Vec<f64>,
    /// Coefficients: for interval [xs[i], xs[i+1]],
    /// S_i(x) = a[i] + b[i]*(x-xs[i]) + c[i]*(x-xs[i])^2 + d[i]*(x-xs[i])^3
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl CubicSpline {
    /// Build a cubic spline from data points and boundary conditions.
    pub fn new(xs: &[f64], ys: &[f64], boundary: &SplineBoundary) -> Self {
        assert_eq!(xs.len(), ys.len());
        let n = xs.len() - 1; // number of intervals
        assert!(n >= 1, "need at least 2 points for spline");

        let mut h = vec![0.0; n];
        for i in 0..n {
            h[i] = xs[i + 1] - xs[i];
        }

        // Solve tridiagonal system for c
        // Natural: c[0] = 0, c[n] = 0
        // Clamped: 2*h[0]*c[0] + h[0]*c[1] = 3*((y[1]-y[0])/h[0] - fpa), etc.
        let mut alpha = vec![0.0; n + 1];
        let mut rhs = vec![0.0; n + 1];

        for i in 1..n {
            alpha[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        match boundary {
            SplineBoundary::Natural => {
                // c[0] = 0, c[n] = 0
                // Solve for c[1..n]
                if n >= 2 {
                    let mut l = vec![0.0; n + 1];
                    let mut mu = vec![0.0; n + 1];
                    let mut z = vec![0.0; n + 1];
                    let mut c = vec![0.0; n + 1];

                    l[1] = 2.0 * h[0] + 2.0 * h[1]; // Actually use standard formulation
                    // Standard natural spline tridiagonal:
                    // For i=1..n-1: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = alpha[i]
                    // With c[0]=0, c[n]=0
                    // Simplified: just solve the inner system

                    let m = n - 1; // size of inner system
                    if m > 0 {
                        let mut diag = vec![0.0; m];
                        let mut upper = vec![0.0; m];
                        let mut lower = vec![0.0; m];
                        let mut b_vec = vec![0.0; m];

                        for j in 0..m {
                            let i = j + 1; // actual index
                            diag[j] = 2.0 * (h[i - 1] + h[i]);
                            b_vec[j] = alpha[i];
                            if j > 0 {
                                lower[j] = h[i - 1];
                            }
                            if j < m - 1 {
                                upper[j] = h[i];
                            }
                        }

                        // Thomas algorithm
                        for j in 1..m {
                            let w = lower[j] / diag[j - 1];
                            diag[j] -= w * upper[j - 1];
                            b_vec[j] -= w * b_vec[j - 1];
                        }
                        let mut sol = vec![0.0; m];
                        sol[m - 1] = b_vec[m - 1] / diag[m - 1];
                        for j in (0..m - 1).rev() {
                            sol[j] = (b_vec[j] - upper[j] * sol[j + 1]) / diag[j];
                        }

                        let mut c_full = vec![0.0; n + 1];
                        for j in 0..m {
                            c_full[j + 1] = sol[j];
                        }

                        return Self::build_from_c(xs, ys, &h, &c_full, n);
                    }
                }
                // Fallback for n=1 (linear)
                let c = vec![0.0; n + 1];
                return Self::build_from_c(xs, ys, &h, &c, n);
            }
            SplineBoundary::Clamped { fpa, fpb } => {
                alpha[0] = 3.0 * ((ys[1] - ys[0]) / h[0] - fpa);
                alpha[n] = 3.0 * (fpb - (ys[n] - ys[n - 1]) / h[n - 1]);

                // Full tridiagonal system for c[0..=n]
                let m = n + 1;
                let mut diag = vec![0.0; m];
                let mut upper = vec![0.0; m];
                let mut lower = vec![0.0; m];
                let mut b_vec = vec![0.0; m];

                diag[0] = 2.0 * h[0];
                b_vec[0] = alpha[0];
                upper[0] = h[0];

                for i in 1..n {
                    diag[i] = 2.0 * (h[i - 1] + h[i]);
                    b_vec[i] = alpha[i];
                    lower[i] = h[i - 1];
                    upper[i] = h[i];
                }

                diag[n] = 2.0 * h[n - 1];
                b_vec[n] = alpha[n];
                lower[n] = h[n - 1];

                // Thomas algorithm
                for i in 1..m {
                    let w = lower[i] / diag[i - 1];
                    diag[i] -= w * upper[i - 1];
                    b_vec[i] -= w * b_vec[i - 1];
                }

                let mut c = vec![0.0; m];
                c[n] = b_vec[n] / diag[n];
                for i in (0..n).rev() {
                    c[i] = (b_vec[i] - upper[i] * c[i + 1]) / diag[i];
                }

                return Self::build_from_c(xs, ys, &h, &c, n);
            }
        }
    }

    fn build_from_c(
        xs: &[f64],
        ys: &[f64],
        h: &[f64],
        c: &[f64],
        n: usize,
    ) -> Self {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut a_y = vec![0.0; n];

        for i in 0..n {
            a_y[i] = ys[i];
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
            b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        }

        CubicSpline {
            xs: xs.to_vec(),
            a: a_y,
            b,
            c: c[..n].to_vec(),
            d,
        }
    }

    /// Evaluate the spline at point `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len() - 1;
        if n == 0 {
            return self.a[0];
        }
        // Find interval
        let i = if x <= self.xs[0] {
            0
        } else if x >= self.xs[n] {
            n - 1
        } else {
            // Binary search
            let mut lo = 0;
            let mut hi = n;
            while lo < hi - 1 {
                let mid = (lo + hi) / 2;
                if self.xs[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let dx = x - self.xs[i];
        self.a[i] + self.b[i] * dx + self.c[i] * dx * dx + self.d[i] * dx * dx * dx
    }

    /// Evaluate at multiple points.
    pub fn eval_batch(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }

    /// Evaluate first derivative at point `x`.
    pub fn eval_deriv(&self, x: f64) -> f64 {
        let n = self.xs.len() - 1;
        let i = if x <= self.xs[0] {
            0
        } else if x >= self.xs[n] {
            n - 1
        } else {
            let mut lo = 0;
            let mut hi = n;
            while lo < hi - 1 {
                let mid = (lo + hi) / 2;
                if self.xs[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let dx = x - self.xs[i];
        self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_natural_spline_linear() {
        // Linear data: spline should be exact
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        assert!(approx_eq(s.eval(0.5), 0.5, 1e-10));
        assert!(approx_eq(s.eval(1.5), 1.5, 1e-10));
    }

    #[test]
    fn test_natural_spline_quadratic() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        // Natural spline won't perfectly reproduce x^2 (S''=0 at boundaries vs S''=2 for x^2)
        // but should be close at interior points
        assert!(approx_eq(s.eval(1.0), 1.0, 1e-10));
        assert!(approx_eq(s.eval(2.0), 4.0, 1e-10));
        // Interior point
        let v15 = s.eval(1.5);
        assert!(approx_eq(v15, 2.25, 0.5), "at 1.5: {v15}");
        // Clamped should be exact
        let sc = CubicSpline::new(&xs, &ys, &SplineBoundary::Clamped { fpa: 0.0, fpb: 6.0 });
        for x in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let expected = x * x;
            let got = sc.eval(x);
            assert!(
                approx_eq(got, expected, 1e-8),
                "clamped at x={x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_clamped_spline_sin() {
        let xs: Vec<f64> = (0..=10).map(|i| i as f64 * std::f64::consts::PI / 10.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let s = CubicSpline::new(
            &xs,
            &ys,
            &SplineBoundary::Clamped {
                fpa: 1.0,
                fpb: (xs[10]).cos(),
            },
        );
        // Check midpoints are close to sin
        for i in 0..10 {
            let mid = (xs[i] + xs[i + 1]) / 2.0;
            let expected = mid.sin();
            let got = s.eval(mid);
            assert!(
                approx_eq(got, expected, 1e-4),
                "at x={mid}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_spline_passes_through_knots() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        for i in 0..xs.len() {
            assert!(approx_eq(s.eval(xs[i]), ys[i], 1e-10), "knot {i}");
        }
    }

    #[test]
    fn test_spline_smoothness_c2() {
        // Natural spline on sin should be C2 — check derivative continuity
        let xs: Vec<f64> = (0..=8).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        // Check derivative is continuous at interior knots
        for i in 1..xs.len() - 1 {
            let eps = 1e-8;
            let left = s.eval_deriv(xs[i] - eps);
            let right = s.eval_deriv(xs[i] + eps);
            assert!(
                approx_eq(left, right, 1e-4),
                "deriv discontinuity at x={}: left={}, right={}",
                xs[i], left, right
            );
        }
    }

    #[test]
    fn test_spline_batch_eval() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        let vals = s.eval_batch(&[0.0, 0.5, 1.0, 1.5, 2.0]);
        assert!(approx_eq(vals[0], 0.0, 1e-10));
        assert!(approx_eq(vals[2], 1.0, 1e-10));
        assert!(approx_eq(vals[4], 4.0, 1e-10));
    }

    #[test]
    fn test_spline_extrapolation() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 2.0, 4.0];
        let s = CubicSpline::new(&xs, &ys, &SplineBoundary::Natural);
        // For linear data, extrapolation should be linear
        assert!(approx_eq(s.eval(-1.0), -2.0, 1e-10));
        assert!(approx_eq(s.eval(3.0), 6.0, 1e-10));
    }
}
