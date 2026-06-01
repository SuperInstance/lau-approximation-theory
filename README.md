# lau-approximation-theory

A comprehensive approximation theory toolkit for Rust — polynomial interpolation, cubic splines, least-squares fitting, Chebyshev bases, Padé approximants, the Remez exchange algorithm, Fourier series truncation, error-bound analysis (Runge phenomenon, Lebesgue constants), and applied agent-behavior modeling.

> **62 tests** · depends on `nalgebra` + `serde`

---

## What This Does

This crate implements the core algorithms of classical approximation theory as pure, composable Rust functions and structs. It covers the full pipeline:

- **Interpolation**: Lagrange and Newton divided-difference polynomials that pass exactly through data points.
- **Splines**: Cubic spline interpolation with natural or clamped boundary conditions, solved via the Thomas algorithm for the tridiagonal system.
- **Least Squares**: Linear regression and polynomial least-squares (Vandermonde matrix → normal equations solved with LU decomposition), including R² goodness-of-fit.
- **Chebyshev Bases**: Evaluation of Chebyshev polynomials T_n(x), their derivatives, nodes (roots), and extrema.
- **Remez Algorithm**: The exchange algorithm for minimax (best-uniform) polynomial approximation — iteratively solves the alternation system.
- **Fourier Approximation**: Truncated Fourier series via numerical integration of a0, an, bn coefficients.
- **Padé Approximants**: Rational function [m/n] approximation from Taylor coefficients, using nalgebra to solve the linear system.
- **Error Bounds**: Runge phenomenon estimation (equispaced vs. Chebyshev nodes), Lebesgue constant computation.
- **Agent Behavior Modeling**: Applied layer — wraps polynomial/spline/piecewise-linear/moving-average models with predict, MAE, RMSE, and model comparison.

---

## Key Idea

Approximation theory asks: *given a function or data, what is the best simple function that approximates it?* The answer depends on what "best" means:

- **Interpolation** → exact at data points (Lagrange, Newton, splines).
- **Least squares** → minimizes sum of squared errors (polynomial regression).
- **Minimax** → minimizes the worst-case error (Remez algorithm).
- **Rational** → captures poles and asymptotics better than polynomials (Padé).
- **Spectral** → captures periodicity (Fourier truncation).

Each module is independent and composable. You can build a Newton table, evaluate Chebyshev nodes, run Remez, compute a Padé approximant, or fit an agent behavior model — all from the same crate.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-approximation-theory = { git = "https://github.com/SuperInstance/lau-approximation-theory" }
```

Or, if published:

```toml
[dependencies]
lau-approximation-theory = "0.1"
```

Dependencies: `nalgebra` (linear algebra), `serde` (serialization).

---

## Quick Start

```rust
use lau_approximation_theory::{
    interpolation, spline, least_squares, chebyshev, fourier, pade, remez,
    error_bounds, agent_model,
};

// --- Lagrange interpolation ---
let xs = vec![0.0, 1.0, 2.0];
let ys = vec![0.0, 1.0, 4.0]; // y = x²
let val = interpolation::lagrange(&xs, &ys, 1.5);
assert!((val - 2.25).abs() < 1e-10);

// --- Newton divided differences ---
let table = interpolation::NewtonTable::build(&xs, &ys);
assert!((table.eval(3.0) - 9.0).abs() < 1e-10);

// --- Cubic spline (natural) ---
let spline = spline::CubicSpline::new(
    &[0.0, 1.0, 2.0, 3.0],
    &[0.0, 1.0, 4.0, 9.0],
    &spline::SplineBoundary::Natural,
);
let mid = spline.eval(1.5); // close to 2.25

// --- Least squares (polynomial degree 2) ---
let fit = least_squares::PolynomialFit::fit(
    &[0.0, 1.0, 2.0, 3.0],
    &[1.0, 2.0, 5.0, 10.0], // y = x² + 1
    2,
);
let r2 = fit.r_squared(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 5.0, 10.0]);

// --- Chebyshev nodes ---
let nodes = chebyshev::chebyshev_nodes(6); // 6 roots of T_6 on [-1, 1]

// --- Remez (minimax degree-1 approximation of x² on [0, 1]) ---
let (coeffs, max_err) = remez::remez(|x| x * x, 1, 0.0, 1.0, 30);

// --- Padé [2/2] of e^x ---
let taylor: Vec<f64> = (0..5).map(|k| 1.0 / (1..=k).product::<u64>() as f64).collect();
let approx = pade::pade_approximant(&taylor, 2, 2);
let e1 = approx.eval(1.0); // ≈ 2.718...

// --- Fourier truncation ---
let fapprox = fourier::fourier_truncate(|t| t.sin(), 2.0 * std::f64::consts::PI, 5, 200);

// --- Agent behavior model ---
let model = agent_model::AgentBehaviorModel::polynomial(
    "agent-42", &[0.0, 1.0, 2.0, 3.0], &[0.0, 2.0, 4.0, 6.0], 1,
);
let pred = model.predict(2.5); // ≈ 5.0
```

---

## API Reference

### `interpolation` — Lagrange & Newton

| Function / Struct | Description |
|---|---|
| `lagrange(xs, ys, x)` | Evaluate Lagrange interpolating polynomial at `x` |
| `lagrange_batch(xs, ys, query)` | Lagrange at multiple query points |
| `NewtonTable::build(xs, ys)` | Build Newton divided-difference table |
| `NewtonTable::eval(x)` | Evaluate Newton polynomial at `x` |
| `NewtonTable::add_point(x, y)` | Incrementally add a data point |
| `newton(xs, ys, x)` | One-shot Newton interpolation |

### `spline` — Cubic Splines

| Function / Struct | Description |
|---|---|
| `CubicSpline::new(xs, ys, boundary)` | Build spline from knots + boundary condition |
| `CubicSpline::eval(x)` | Evaluate spline at `x` |
| `CubicSpline::eval_batch(points)` | Evaluate at multiple points |
| `CubicSpline::eval_deriv(x)` | Evaluate first derivative S'(x) |
| `SplineBoundary::Natural` | S''(a) = S''(b) = 0 |
| `SplineBoundary::Clamped { fpa, fpb }` | S'(a) = fpa, S'(b) = fpb |

### `least_squares` — Linear & Polynomial

| Function / Struct | Description |
|---|---|
| `linear_least_squares(xs, ys)` | Fit y = a + bx → (intercept, slope) |
| `PolynomialFit::fit(xs, ys, degree)` | Polynomial least squares of given degree |
| `PolynomialFit::eval(x)` | Evaluate fitted polynomial |
| `PolynomialFit::r_squared(xs, ys)` | Coefficient of determination R² |

### `chebyshev` — Chebyshev Polynomials

| Function / Struct | Description |
|---|---|
| `ChebyshevBasis::new(max_degree)` | Create basis for degrees 0..=max_degree |
| `ChebyshevBasis::eval(n, x)` | Evaluate T_n(x) via recurrence |
| `ChebyshevBasis::eval_all(x)` | All T_0..=T_n at x |
| `ChebyshevBasis::eval_deriv(n, x)` | Derivative T_n'(x) |
| `chebyshev_nodes(n)` | Roots of T_n on [-1, 1] |
| `chebyshev_roots(n)` | Alias for `chebyshev_nodes` |
| `chebyshev_extrema(n)` | Extrema (2nd-kind nodes) |

### `remez` — Remez Exchange Algorithm

| Function | Description |
|---|---|
| `remez(f, degree, a, b, max_iter)` | Find minimax polynomial on [a, b] → (coeffs, max_error) |

### `fourier` — Truncated Fourier Series

| Function / Struct | Description |
|---|---|
| `FourierApprox::fit(f, period, n_terms, n_samples)` | Compute truncated Fourier coefficients |
| `FourierApprox::eval(t)` | Evaluate at time `t` |
| `fourier_truncate(f, period, n_terms, n_samples)` | Convenience wrapper |

### `pade` — Padé Approximants

| Function / Struct | Description |
|---|---|
| `PadeApproximant::from_taylor(taylor, m, n)` | Build [m/n] Padé from Taylor coefficients |
| `PadeApproximant::eval(x)` | Evaluate R(x) = P(x)/Q(x) |
| `pade_approximant(taylor, m, n)` | Convenience function |
| `numerical_taylor(f, order, h)` | Numerically compute Taylor coefficients |

### `error_bounds` — Runge & Lebesgue

| Function | Description |
|---|---|
| `runge_error_estimate(n)` | Runge phenomenon error for equispaced nodes on [-1, 1] |
| `runge_error_chebyshev(n)` | Same but with Chebyshev nodes |
| `lebesgue_constant(nodes, n_samples)` | Lebesgue constant Λ for arbitrary nodes |
| `lebesgue_constant_equispaced(n)` | Λ for equispaced nodes on [-1, 1] |
| `lebesgue_constant_chebyshev(n)` | Λ for Chebyshev nodes |

### `agent_model` — Agent Behavior Modeling

| Function / Struct | Description |
|---|---|
| `AgentBehaviorModel::polynomial(id, times, values, degree)` | Polynomial behavior model |
| `AgentBehaviorModel::spline(id, times, values, boundary)` | Spline behavior model |
| `AgentBehaviorModel::piecewise_linear(id, times, values)` | Piecewise-linear model |
| `AgentBehaviorModel::moving_average(id, data, window)` | Smoothed moving-average model |
| `AgentBehaviorModel::predict(t)` | Predict value at time t |
| `AgentBehaviorModel::mean_absolute_error(times, actual)` | MAE on test data |
| `AgentBehaviorModel::rmse(times, actual)` | RMSE on test data |
| `compare_models(models, times, actual)` | Rank models by MAE |

---

## How It Works

### Interpolation (Lagrange, Newton)

**Lagrange** builds the cardinal basis polynomials L_i(x) = ∏_{j≠i} (x - x_j)/(x_i - x_j) and sums y_i · L_i(x). O(n²) per evaluation.

**Newton** stores divided-difference coefficients f[x_0], f[x_0,x_1], … computed in O(n²) and evaluates in O(n) using the nested form: P(x) = c_0 + (x-x_0)(c_1 + (x-x_1)(c_2 + …)). The `NewtonTable` supports incremental point addition.

### Cubic Splines

On each interval [x_i, x_{i+1}], the spline is S_i(x) = a_i + b_i·(x-x_i) + c_i·(x-x_i)² + d_i·(x-x_i)³. Continuity of S, S', S'' at interior knots yields a tridiagonal system solved via the **Thomas algorithm** (O(n)).

- **Natural**: S''(x_0) = S''(x_n) = 0 (default, slight error at boundaries).
- **Clamped**: S'(x_0) = fpa, S'(x_n) = fpb (exact for polynomials up to degree 3 when derivative is known).

### Least Squares

Builds the Vandermonde matrix A where A_{ij} = x_i^j and solves the **normal equations** A^T A c = A^T b using `nalgebra`'s LU decomposition. The `PolynomialFit` struct stores coefficients and provides eval and R².

### Chebyshev Polynomials

Uses the three-term recurrence T_0=1, T_1=x, T_{n+1} = 2x·T_n - T_{n-1}. Derivatives use T_n'(x) = n·sin(n·arccos(x))/sin(arccos(x)). Chebyshev nodes cluster near ±1, avoiding the Runge phenomenon.

### Remez (Exchange) Algorithm

1. Initialize reference points from Chebyshev nodes mapped to [a, b].
2. Solve the alternation system: for each reference point x_j, impose P(x_j) + (-1)^j·h = f(x_j).
3. Find error extrema on a dense grid.
4. Select n+2 alternating-sign extrema as new reference points.
5. Repeat for `max_iter` iterations.

Returns the polynomial coefficients and the equioscillation error bound.

### Padé Approximants

Given Taylor coefficients c_0, c_1, …, c_{m+n} of f at x=0:
1. Solve the linear system: for k = m+1..m+n, c_k + Σ_{j=1}^n q_j·c_{k-j} = 0 for q_1..q_n.
2. Compute p_i = c_i + Σ_{j=1}^{min(i,n)} q_j·c_{i-j} for i = 0..m.
3. Result: R(x) = P(x)/Q(x) where Q(0)=1.

### Fourier Truncation

Computes a_0, a_n, b_n via Riemann sum approximation on n_samples evenly-spaced points over one period [0, T]:
- a_0 = (1/N) Σ f(t_k)
- a_n = (2/N) Σ f(t_k) cos(nωt_k)
- b_n = (2/N) Σ f(t_k) sin(nωt_k)

### Error Bounds

- **Runge phenomenon**: Measures interpolation error for 1/(1+25x²) on [-1, 1] using equispaced vs. Chebyshev nodes. Chebyshev nodes keep the error bounded.
- **Lebesgue constant**: Λ = max_x Σ |L_i(x)| — measures how much interpolation amplifies perturbations. Equispaced Λ grows exponentially; Chebyshev Λ grows logarithmically.

### Agent Behavior Models

Wraps the approximation primitives in a unified `AgentBehaviorModel` with four strategies (polynomial, spline, piecewise-linear, moving average). Provides prediction, error metrics (MAE, RMSE), and a `compare_models` function that ranks multiple models.

---

## The Math

### Lagrange Basis
$$P(x) = \sum_{i=0}^{n} y_i \prod_{\substack{j=0 \\ j \ne i}}^{n} \frac{x - x_j}{x_i - x_j}$$

### Newton Divided Differences
$$f[x_i, \ldots, x_j] = \frac{f[x_{i+1}, \ldots, x_j] - f[x_i, \ldots, x_{j-1}]}{x_j - x_i}$$

### Cubic Spline (Thomas Algorithm)
For natural splines, solve h_{i-1}·c_{i-1} + 2(h_{i-1}+h_i)·c_i + h_i·c_{i+1} = α_i with c_0 = c_n = 0.

### Least Squares Normal Equations
$$(A^T A) \mathbf{c} = A^T \mathbf{y}$$

### Chebyshev Recurrence
$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x \cdot T_n(x) - T_{n-1}(x)$$

### Padé System
$$c_k + \sum_{j=1}^{n} q_j \cdot c_{k-j} = 0, \quad k = m+1, \ldots, m+n$$

### Remez Alternation
$$P(x_j) + (-1)^j h = f(x_j), \quad j = 0, 1, \ldots, n+1$$

### Fourier Coefficients
$$a_0 = \frac{1}{T}\int_0^T f(t)\,dt, \quad a_n = \frac{2}{T}\int_0^T f(t)\cos(n\omega t)\,dt, \quad b_n = \frac{2}{T}\int_0^T f(t)\sin(n\omega t)\,dt$$

### Lebesgue Constant
$$\Lambda_n = \max_{x \in [a,b]} \sum_{i=0}^{n} |L_i(x)|$$

---

## License

MIT or Apache-2.0 (at your option).
