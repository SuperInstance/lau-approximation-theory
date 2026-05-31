//! Remez algorithm (exchange algorithm) for minimax polynomial approximation.

use crate::chebyshev::chebyshev_nodes;

/// Run a basic Remez exchange algorithm to find the minimax polynomial of given degree
/// approximating `f` on [a, b].
///
/// Returns the polynomial coefficients [c0, c1, ..., c_degree] (in increasing degree order)
/// and the estimated maximum error.
///
/// This is a simplified implementation suitable for moderate degrees and smooth functions.
pub fn remez<F>(f: F, degree: usize, a: f64, b: f64, max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(f64) -> f64,
{
    let n = degree + 1; // number of polynomial coefficients
    let n2 = n + 1;     // number of reference points

    // Initialize reference points using Chebyshev nodes mapped to [a, b]
    let mut ref_points: Vec<f64> = chebyshev_nodes(n2)
        .into_iter()
        .map(|t| (a + b) / 2.0 + (b - a) / 2.0 * t)
        .collect();
    ref_points.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut best_coeffs = vec![0.0; n];
    let mut best_error = 0.0;

    for _ in 0..max_iter {
        // Solve the linearized Remez system:
        // For each reference point x_j: sum_i c_i * x_j^i + (-1)^j * h = f(x_j)
        // Unknowns: [c_0, c_1, ..., c_n-1, h]
        let mut mat = vec![0.0; n2 * n2];
        let mut rhs = vec![0.0; n2];

        for j in 0..n2 {
            let x = ref_points[j];
            let mut xp = 1.0;
            for i in 0..n {
                mat[j * n2 + i] = xp;
                xp *= x;
            }
            // Last column: alternating sign
            mat[j * n2 + n] = if j % 2 == 0 { 1.0 } else { -1.0 };
            rhs[j] = f(x);
        }

        // Solve using simple Gaussian elimination (small system)
        let solution = gauss_solve(&mat, &rhs, n2);

        if solution.is_none() {
            break;
        }
        let sol = solution.unwrap();
        let coeffs = sol[..n].to_vec();
        let h = sol[n];

        // Find the points where |f(x) - p(x)| is maximized on a dense grid
        let grid_size = 200;
        let mut max_err = 0.0_f64;
        let mut new_refs = Vec::new();

        // Find extrema of the error function
        let mut errors: Vec<(f64, f64)> = (0..=grid_size)
            .map(|k| {
                let x = a + (b - a) * k as f64 / grid_size as f64;
                let mut poly_val = 0.0;
                let mut xp = 1.0;
                for &c in &coeffs {
                    poly_val += c * xp;
                    xp *= x;
                }
                let err = f(x) - poly_val;
                (x, err)
            })
            .collect();

        // Find local extrema
        let mut extrema: Vec<(f64, f64)> = Vec::new();
        for i in 1..errors.len() - 1 {
            let (x0, e0) = errors[i - 1];
            let (x1, e1) = errors[i];
            let (x2, e2) = errors[i + 1];
            if (e1 >= e0 && e1 >= e2) || (e1 <= e0 && e1 <= e2) {
                extrema.push((x1, e1));
            }
        }

        max_err = errors.iter().map(|(_, e)| e.abs()).fold(0.0_f64, f64::max);

        // Select n+1 alternating extrema
        if extrema.len() >= n2 {
            // Pick extrema with alternating signs, starting from the one with largest absolute error
            let max_idx = extrema
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            new_refs.push(extrema[max_idx].0);
            let mut last_sign = extrema[max_idx].1.signum();

            // Go forward from max_idx
            let mut forward: Vec<(f64, f64)> = extrema[max_idx + 1..].to_vec();
            let mut backward: Vec<(f64, f64)> = extrema[..max_idx].to_vec();
            backward.reverse();

            // Merge both directions, picking alternating signs
            let mut all_remaining: Vec<(f64, f64)> = Vec::new();
            let mut fi = 0;
            let mut bi = 0;
            loop {
                let f_dist = if fi < forward.len() {
                    Some((forward[fi].0 - new_refs.last().unwrap()).abs())
                } else {
                    None
                };
                let b_dist = if bi < backward.len() {
                    Some((backward[bi].0 - new_refs.last().unwrap()).abs())
                } else {
                    None
                };

                match (f_dist, b_dist) {
                    (Some(fd), Some(bd)) => {
                        if fd < bd {
                            all_remaining.push(forward[fi]);
                            fi += 1;
                        } else {
                            all_remaining.push(backward[bi]);
                            bi += 1;
                        }
                    }
                    (Some(_), None) => {
                        all_remaining.push(forward[fi]);
                        fi += 1;
                    }
                    (None, Some(_)) => {
                        all_remaining.push(backward[bi]);
                        bi += 1;
                    }
                    (None, None) => break,
                }
            }

            // Actually, simpler approach: just pick extrema with alternating signs
            let mut refs_sorted: Vec<(f64, f64)> = extrema.clone();
            refs_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            new_refs.clear();
            let start_sign = refs_sorted[0].1.signum();
            let mut want_sign = start_sign;
            for &(x, e) in &refs_sorted {
                if new_refs.is_empty() || e.signum() == want_sign {
                    new_refs.push(x);
                    want_sign = -want_sign;
                }
                if new_refs.len() == n2 {
                    break;
                }
            }

            if new_refs.len() < n2 {
                // Fallback: take evenly spaced extrema
                new_refs = extrema
                    .iter()
                    .step_by((extrema.len() / n2).max(1))
                    .take(n2)
                    .map(|(x, _)| *x)
                    .collect();
            }
        }

        if new_refs.len() != n2 {
            best_coeffs = coeffs;
            best_error = h.abs();
            break;
        }

        new_refs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ref_points = new_refs;
        best_coeffs = coeffs;
        best_error = max_err;
    }

    (best_coeffs, best_error)
}

fn gauss_solve(mat: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }

        // Eliminate
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
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

    #[test]
    fn test_remez_constant() {
        // Approximating a constant with degree 0
        let (coeffs, err) = remez(|_| 3.0, 0, -1.0, 1.0, 20);
        assert!(approx_eq(coeffs[0], 3.0, 1e-6), "coeff={}", coeffs[0]);
        assert!(err < 1e-6, "error={err}");
    }

    #[test]
    fn test_remez_linear() {
        // Approximating 2x+1 with degree 1
        let (coeffs, _err) = remez(|x| 2.0 * x + 1.0, 1, -1.0, 1.0, 20);
        assert!(approx_eq(coeffs[0], 1.0, 1e-4), "c0={}", coeffs[0]);
        assert!(approx_eq(coeffs[1], 2.0, 1e-4), "c1={}", coeffs[1]);
    }

    #[test]
    fn test_remez_quadratic_approx() {
        // Approximate x^2 with degree 1 on [0, 1]
        // Best linear approximation to x^2 on [0,1] is x - 1/8, max error = 1/8
        let (coeffs, err) = remez(|x| x * x, 1, 0.0, 1.0, 30);
        assert!(err < 0.15, "error too large: {err}");
        // Check it's a reasonable approximation
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let diff = (x * x - eval_poly(&coeffs, x)).abs();
            assert!(diff < 0.2, "error at {x}: {diff}");
        }
    }

    #[test]
    fn test_remez_sin() {
        // Approximate sin(x) with degree 3 on [-1, 1]
        let (coeffs, err) = remez(|x| x.sin(), 3, -1.0, 1.0, 30);
        assert!(err < 0.01, "error too large: {err}");
        // Check approximation quality
        for x in [-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let diff = (x.sin() - eval_poly(&coeffs, x)).abs();
            assert!(diff < 0.02, "error at {x}: {diff}");
        }
    }
}
