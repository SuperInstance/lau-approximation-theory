//! Chebyshev polynomials: evaluation, roots, nodes.

use serde::{Deserialize, Serialize};

/// Chebyshev polynomial basis of the first kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChebyshevBasis {
    /// Maximum degree.
    pub max_degree: usize,
}

impl ChebyshevBasis {
    /// Create a basis up to given degree.
    pub fn new(max_degree: usize) -> Self {
        ChebyshevBasis { max_degree }
    }

    /// Evaluate T_n(x) using the recurrence T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}.
    pub fn eval(&self, n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        let mut t_prev = 1.0;
        let mut t_curr = x;
        for _ in 2..=n {
            let t_next = 2.0 * x * t_curr - t_prev;
            t_prev = t_curr;
            t_curr = t_next;
        }
        t_curr
    }

    /// Evaluate all T_0..=T_n at x. Returns vector of length n+1.
    pub fn eval_all(&self, x: f64) -> Vec<f64> {
        let n = self.max_degree;
        let mut vals = Vec::with_capacity(n + 1);
        if n >= 0 {
            vals.push(1.0);
        }
        if n >= 1 {
            vals.push(x);
        }
        for k in 2..=n {
            let prev2 = vals[k - 2];
            let prev1 = vals[k - 1];
            vals.push(2.0 * x * prev1 - prev2);
        }
        vals
    }

    /// Derivative of T_n at x: T_n'(x) = n * U_{n-1}(x) where U is Chebyshev of 2nd kind.
    /// We use the identity: T_n'(x) = n * sin(n*arccos(x)) / sin(arccos(x))
    pub fn eval_deriv(&self, n: usize, x: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return 1.0;
        }
        // Use sin formulation
        let theta = x.acos();
        n as f64 * (n as f64 * theta).sin() / theta.sin().max(1e-15)
    }
}

/// Chebyshev nodes of the first kind: x_k = cos((2k+1)π / (2n)), k=0..n-1.
pub fn chebyshev_nodes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos())
        .collect()
}

/// Chebyshev roots (same as nodes of the first kind, T_n zeros).
pub fn chebyshev_roots(n: usize) -> Vec<f64> {
    chebyshev_nodes(n)
}

/// Chebyshev extrema (nodes of the second kind): cos(kπ/n), k=0..n.
pub fn chebyshev_extrema(n: usize) -> Vec<f64> {
    (0..=n)
        .map(|k| (k as f64 * std::f64::consts::PI / n as f64).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_t0() {
        let basis = ChebyshevBasis::new(5);
        assert!(approx_eq(basis.eval(0, 0.5), 1.0, 1e-10));
        assert!(approx_eq(basis.eval(0, -1.0), 1.0, 1e-10));
    }

    #[test]
    fn test_t1() {
        let basis = ChebyshevBasis::new(5);
        assert!(approx_eq(basis.eval(1, 0.5), 0.5, 1e-10));
        assert!(approx_eq(basis.eval(1, -1.0), -1.0, 1e-10));
    }

    #[test]
    fn test_t2() {
        // T_2(x) = 2x^2 - 1
        let basis = ChebyshevBasis::new(5);
        assert!(approx_eq(basis.eval(2, 0.0), -1.0, 1e-10));
        assert!(approx_eq(basis.eval(2, 1.0), 1.0, 1e-10));
        assert!(approx_eq(basis.eval(2, 0.5), -0.5, 1e-10));
    }

    #[test]
    fn test_t3() {
        // T_3(x) = 4x^3 - 3x
        let basis = ChebyshevBasis::new(5);
        assert!(approx_eq(basis.eval(3, 0.5), -1.0, 1e-10));
        assert!(approx_eq(basis.eval(3, 1.0), 1.0, 1e-10));
    }

    #[test]
    fn test_t_bounds() {
        // |T_n(x)| ≤ 1 for x ∈ [-1,1]
        let basis = ChebyshevBasis::new(10);
        for n in 0..=10 {
            for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                let val = basis.eval(n, x);
                assert!(
                    val.abs() <= 1.0 + 1e-10,
                    "|T_{n}({x})| = {val} > 1"
                );
            }
        }
    }

    #[test]
    fn test_chebyshev_nodes_in_interval() {
        let nodes = chebyshev_nodes(5);
        assert_eq!(nodes.len(), 5);
        for &x in &nodes {
            assert!(x >= -1.0 - 1e-10 && x <= 1.0 + 1e-10, "node {x} out of [-1,1]");
        }
    }

    #[test]
    fn test_chebyshev_roots_are_zeros() {
        let basis = ChebyshevBasis::new(10);
        let roots = chebyshev_roots(6);
        for &x in &roots {
            let val = basis.eval(6, x);
            assert!(
                val.abs() < 1e-10,
                "T_6({x}) = {val}, expected ~0"
            );
        }
    }

    #[test]
    fn test_chebyshev_nodes_symmetric() {
        let nodes = chebyshev_nodes(4);
        // Nodes should be symmetric about 0
        for k in 0..2 {
            assert!(
                approx_eq(nodes[k], -nodes[3 - k], 1e-10),
                "asymmetry: {} vs {}",
                nodes[k],
                -nodes[3 - k]
            );
        }
    }

    #[test]
    fn test_eval_all() {
        let basis = ChebyshevBasis::new(4);
        let vals = basis.eval_all(0.5);
        assert_eq!(vals.len(), 5);
        assert!(approx_eq(vals[0], 1.0, 1e-10));
        assert!(approx_eq(vals[1], 0.5, 1e-10));
        // T_2(0.5) = -0.5
        assert!(approx_eq(vals[2], -0.5, 1e-10));
    }

    #[test]
    fn test_chebyshev_extrema() {
        let ext = chebyshev_extrema(4);
        assert_eq!(ext.len(), 5);
        // Extrema should be ±1 and interior values
        assert!(approx_eq(ext[0], 1.0, 1e-10));
        assert!(approx_eq(ext[4], -1.0, 1e-10));
    }
}
