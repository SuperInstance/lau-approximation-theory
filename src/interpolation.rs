//! Polynomial interpolation: Lagrange and Newton divided differences.

use serde::{Deserialize, Serialize};

/// Lagrange interpolation at point `x` given data points `(xs[i], ys[i])`.
///
/// Returns `P(x)` where `P` is the unique degree-≤(n-1) polynomial passing
/// through all data points.
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have same length");
    assert!(!xs.is_empty(), "need at least one data point");
    let n = xs.len();
    let mut result = 0.0;
    for i in 0..n {
        let mut basis = 1.0;
        for j in 0..n {
            if j != i {
                let denom = xs[i] - xs[j];
                if denom.abs() < 1e-15 {
                    continue;
                }
                basis *= (x - xs[j]) / denom;
            }
        }
        result += ys[i] * basis;
    }
    result
}

/// Lagrange interpolation returning values at many query points.
pub fn lagrange_batch(xs: &[f64], ys: &[f64], query: &[f64]) -> Vec<f64> {
    query.iter().map(|&x| lagrange(xs, ys, x)).collect()
}

/// Newton divided-difference interpolation table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtonTable {
    /// Sorted x values.
    pub xs: Vec<f64>,
    /// Divided-difference coefficients `f[x0], f[x0,x1], f[x0,x1,x2], ...`.
    pub coeffs: Vec<f64>,
}

impl NewtonTable {
    /// Build the Newton divided-difference table from data points.
    pub fn build(xs: &[f64], ys: &[f64]) -> Self {
        assert_eq!(xs.len(), ys.len());
        let n = xs.len();
        let mut dd = vec![vec![0.0; n]; n];
        for i in 0..n {
            dd[i][0] = ys[i];
        }
        for j in 1..n {
            for i in 0..(n - j) {
                let denom = xs[i + j] - xs[i];
                dd[i][j] = if denom.abs() < 1e-15 {
                    0.0
                } else {
                    (dd[i + 1][j - 1] - dd[i][j - 1]) / denom
                };
            }
        }
        let coeffs: Vec<f64> = (0..n).map(|j| dd[0][j]).collect();
        NewtonTable {
            xs: xs.to_vec(),
            coeffs,
        }
    }

    /// Evaluate the Newton interpolating polynomial at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 {
            return 0.0;
        }
        let mut result = self.coeffs[0];
        let mut product = 1.0;
        for i in 1..n {
            product *= x - self.xs[i - 1];
            result += self.coeffs[i] * product;
        }
        result
    }

    /// Add a new data point (incremental Newton).
    pub fn add_point(&mut self, x: f64, y: f64) {
        let n = self.xs.len();
        // Recompute: add the point and rebuild
        let mut xs = self.xs.clone();
        let mut ys: Vec<f64> = (0..n).map(|i| {
            // Recover ys from the table: y_i = lagrange eval at x_i
            // Simpler: just store ys alongside. For now, rebuild.
            self.eval(self.xs[i])
        }).collect();
        xs.push(x);
        ys.push(y);
        *self = Self::build(&xs, &ys);
    }
}

/// Convenience: Newton interpolation at a single point.
pub fn newton(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    NewtonTable::build(xs, ys).eval(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_lagrange_linear() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 2.0];
        // f(x) = 2x
        assert!(approx_eq(lagrange(&xs, &ys, 0.5), 1.0, 1e-10));
        assert!(approx_eq(lagrange(&xs, &ys, 2.0), 4.0, 1e-10));
    }

    #[test]
    fn test_lagrange_quadratic() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0]; // f(x) = x^2
        assert!(approx_eq(lagrange(&xs, &ys, 0.5), 0.25, 1e-10));
        assert!(approx_eq(lagrange(&xs, &ys, 1.5), 2.25, 1e-10));
        assert!(approx_eq(lagrange(&xs, &ys, 3.0), 9.0, 1e-10));
    }

    #[test]
    fn test_lagrange_single_point() {
        let xs = vec![5.0];
        let ys = vec![3.0];
        assert!(approx_eq(lagrange(&xs, &ys, 100.0), 3.0, 1e-10));
    }

    #[test]
    fn test_lagrange_passes_through_data() {
        let xs = vec![0.0, 1.0, -1.0, 2.0];
        let ys = vec![1.0, 2.0, 0.0, 5.0];
        for i in 0..xs.len() {
            assert!(approx_eq(lagrange(&xs, &ys, xs[i]), ys[i], 1e-10));
        }
    }

    #[test]
    fn test_newton_linear() {
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, 3.0]; // f(x) = 1 + 2x
        assert!(approx_eq(newton(&xs, &ys, 0.5), 2.0, 1e-10));
        assert!(approx_eq(newton(&xs, &ys, 2.0), 5.0, 1e-10));
    }

    #[test]
    fn test_newton_quadratic() {
        let xs = vec![-1.0, 0.0, 1.0];
        let ys = vec![1.0, 0.0, 1.0]; // f(x) = x^2
        assert!(approx_eq(newton(&xs, &ys, 0.5), 0.25, 1e-10));
        assert!(approx_eq(newton(&xs, &ys, 2.0), 4.0, 1e-10));
    }

    #[test]
    fn test_newton_table_eval() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 5.0, 10.0]; // f(x) = x^2 + 1
        let table = NewtonTable::build(&xs, &ys);
        assert!(approx_eq(table.eval(1.5), 3.25, 1e-10));
        assert!(approx_eq(table.eval(4.0), 17.0, 1e-10));
    }

    #[test]
    fn test_newton_matches_lagrange() {
        let xs = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let ys: Vec<f64> = xs.iter().map(|x: &f64| x.sin()).collect();
        for x in [0.25, 0.75, 1.25, 1.75] {
            let l = lagrange(&xs, &ys, x);
            let n = newton(&xs, &ys, x);
            assert!(approx_eq(l, n, 1e-10), "at x={x}: lagrange={l}, newton={n}");
        }
    }

    #[test]
    fn test_lagrange_batch() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let vals = lagrange_batch(&xs, &ys, &[0.5, 1.5, 3.0]);
        assert!(approx_eq(vals[0], 0.25, 1e-10));
        assert!(approx_eq(vals[1], 2.25, 1e-10));
        assert!(approx_eq(vals[2], 9.0, 1e-10));
    }
}
