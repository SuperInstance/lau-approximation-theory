# lau-approximation-theory

> Approximation theory: polynomial/spline interpolation, least squares, Chebyshev polynomials, Padé approximants, Remez algorithm, and error analysis for agent behavior modeling

## What This Does

Approximation theory: polynomial/spline interpolation, least squares, Chebyshev polynomials, Padé approximants, Remez algorithm, and error analysis for agent behavior modeling. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-approximation-theory
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_approximation_theory::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct AgentBehaviorModel 
pub enum ApproximationMethod 
    pub fn polynomial(agent_id: &str, times: &[f64], values: &[f64], degree: usize) -> Self 
    pub fn spline(agent_id: &str, times: &[f64], values: &[f64], boundary: &SplineBoundary) -> Self 
    pub fn piecewise_linear(agent_id: &str, times: &[f64], values: &[f64]) -> Self 
    pub fn moving_average(agent_id: &str, data: &[f64], window: usize) -> Self 
    pub fn predict(&self, t: f64) -> f64 
    pub fn mean_absolute_error(&self, times: &[f64], actual: &[f64]) -> f64 
    pub fn rmse(&self, times: &[f64], actual: &[f64]) -> f64 
pub fn compare_models(models: &[AgentBehaviorModel], times: &[f64], actual: &[f64]) -> Vec<(String, f64)> 
pub enum SplineBoundary 
pub struct CubicSpline 
    pub fn new(xs: &[f64], ys: &[f64], boundary: &SplineBoundary) -> Self 
    pub fn eval(&self, x: f64) -> f64 
    pub fn eval_batch(&self, points: &[f64]) -> Vec<f64> 
    pub fn eval_deriv(&self, x: f64) -> f64 
pub fn runge_error_estimate(n: usize) -> f64 
pub fn runge_error_chebyshev(n: usize) -> f64 
pub fn lebesgue_constant(nodes: &[f64], n_samples: usize) -> f64 
pub fn lebesgue_constant_equispaced(n: usize) -> f64 
pub fn lebesgue_constant_chebyshev(n: usize) -> f64 
pub fn linear_least_squares(xs: &[f64], ys: &[f64]) -> (f64, f64) 
pub struct PolynomialFit 
    pub fn fit(xs: &[f64], ys: &[f64], degree: usize) -> Self 
    pub fn eval(&self, x: f64) -> f64 
    pub fn r_squared(&self, xs: &[f64], ys: &[f64]) -> f64 
pub fn polynomial_least_squares(xs: &[f64], ys: &[f64], degree: usize) -> Vec<f64> 
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 
pub fn lagrange_batch(xs: &[f64], ys: &[f64], query: &[f64]) -> Vec<f64> 
pub struct NewtonTable 
    pub fn build(xs: &[f64], ys: &[f64]) -> Self 
    pub fn eval(&self, x: f64) -> f64 
    pub fn add_point(&mut self, x: f64, y: f64) 
pub fn newton(xs: &[f64], ys: &[f64], x: f64) -> f64 
pub struct FourierApprox 
    pub fn fit<F>(f: F, period: f64, n_terms: usize, n_samples: usize) -> Self
    pub fn eval(&self, t: f64) -> f64 
pub fn fourier_truncate<F>(f: F, period: f64, n_terms: usize, n_samples: usize) -> FourierApprox
pub fn remez<F>(f: F, degree: usize, a: f64, b: f64, max_iter: usize) -> (Vec<f64>, f64)
pub struct ChebyshevBasis 
    pub fn new(max_degree: usize) -> Self 
    pub fn eval(&self, n: usize, x: f64) -> f64 
    pub fn eval_all(&self, x: f64) -> Vec<f64> 
    pub fn eval_deriv(&self, n: usize, x: f64) -> f64 
pub fn chebyshev_nodes(n: usize) -> Vec<f64> 
pub fn chebyshev_roots(n: usize) -> Vec<f64> 
pub fn chebyshev_extrema(n: usize) -> Vec<f64> 
pub struct PadeApproximant 
    pub fn from_taylor(taylor: &[f64], m: usize, n: usize) -> Self 
    pub fn eval(&self, x: f64) -> f64 
pub fn pade_approximant(taylor: &[f64], m: usize, n: usize) -> PadeApproximant 
pub fn numerical_taylor<F>(f: F, order: usize, h: f64) -> Vec<f64>
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**62 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
