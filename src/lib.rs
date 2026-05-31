//! # lau-approximation-theory
//!
//! Approximation theory toolkit: polynomial/spline interpolation, least squares,
//! Chebyshev polynomials, Padé approximants, Remez algorithm basics, Fourier
//! truncation, and error analysis — with applications to agent behavior modeling.

pub mod interpolation;
pub mod spline;
pub mod least_squares;
pub mod chebyshev;
pub mod remez;
pub mod fourier;
pub mod pade;
pub mod error_bounds;
pub mod agent_model;

pub use interpolation::{lagrange, newton};
pub use spline::{CubicSpline, SplineBoundary};
pub use least_squares::{linear_least_squares, polynomial_least_squares};
pub use chebyshev::{ChebyshevBasis, chebyshev_nodes, chebyshev_roots};
pub use remez::remez;
pub use fourier::fourier_truncate;
pub use pade::pade_approximant;
pub use error_bounds::{lebesgue_constant, runge_error_estimate};
pub use agent_model::AgentBehaviorModel;
