//! Application: approximating complex agent dynamics with tractable models.

use serde::{Deserialize, Serialize};

use crate::least_squares::PolynomialFit;
use crate::spline::{CubicSpline, SplineBoundary};

/// A model of agent behavior using approximation techniques.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBehaviorModel {
    /// Name/identifier for the agent.
    pub agent_id: String,
    /// The approximation method used.
    pub method: ApproximationMethod,
}

/// Available approximation methods for agent modeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApproximationMethod {
    /// Polynomial fit of given degree.
    Polynomial {
        fit: PolynomialFit,
    },
    /// Cubic spline interpolation.
    Spline {
        spline: CubicSpline,
    },
    /// Piecewise linear interpolation.
    PiecewiseLinear {
        xs: Vec<f64>,
        ys: Vec<f64>,
    },
    /// Moving average smoothing.
    MovingAverage {
        window: usize,
        data: Vec<f64>,
    },
}

impl AgentBehaviorModel {
    /// Create a polynomial model for agent behavior.
    pub fn polynomial(agent_id: &str, times: &[f64], values: &[f64], degree: usize) -> Self {
        let fit = PolynomialFit::fit(times, values, degree);
        AgentBehaviorModel {
            agent_id: agent_id.to_string(),
            method: ApproximationMethod::Polynomial { fit },
        }
    }

    /// Create a cubic spline model for agent behavior.
    pub fn spline(agent_id: &str, times: &[f64], values: &[f64], boundary: &SplineBoundary) -> Self {
        let spline = CubicSpline::new(times, values, boundary);
        AgentBehaviorModel {
            agent_id: agent_id.to_string(),
            method: ApproximationMethod::Spline { spline },
        }
    }

    /// Create a piecewise linear model.
    pub fn piecewise_linear(agent_id: &str, times: &[f64], values: &[f64]) -> Self {
        AgentBehaviorModel {
            agent_id: agent_id.to_string(),
            method: ApproximationMethod::PiecewiseLinear {
                xs: times.to_vec(),
                ys: values.to_vec(),
            },
        }
    }

    /// Create a moving average model.
    pub fn moving_average(agent_id: &str, data: &[f64], window: usize) -> Self {
        let smoothed = Self::compute_moving_average(data, window);
        AgentBehaviorModel {
            agent_id: agent_id.to_string(),
            method: ApproximationMethod::MovingAverage {
                window,
                data: smoothed,
            },
        }
    }

    /// Predict the agent's behavior at time t.
    pub fn predict(&self, t: f64) -> f64 {
        match &self.method {
            ApproximationMethod::Polynomial { fit } => fit.eval(t),
            ApproximationMethod::Spline { spline } => spline.eval(t),
            ApproximationMethod::PiecewiseLinear { xs, ys } => {
                if xs.is_empty() {
                    return 0.0;
                }
                if t <= xs[0] {
                    return ys[0];
                }
                if t >= *xs.last().unwrap() {
                    return *ys.last().unwrap();
                }
                // Find interval
                for i in 0..xs.len() - 1 {
                    if xs[i] <= t && t <= xs[i + 1] {
                        let frac = (t - xs[i]) / (xs[i + 1] - xs[i]);
                        return ys[i] + frac * (ys[i + 1] - ys[i]);
                    }
                }
                *ys.last().unwrap()
            }
            ApproximationMethod::MovingAverage { data, .. } => {
                // Return the smoothed value at the nearest index
                let idx = (t.round() as usize).min(data.len().saturating_sub(1));
                data.get(idx).copied().unwrap_or(0.0)
            }
        }
    }

    /// Compute prediction error (MAE) on test data.
    pub fn mean_absolute_error(&self, times: &[f64], actual: &[f64]) -> f64 {
        let n = times.len().min(actual.len());
        if n == 0 {
            return 0.0;
        }
        let total_error: f64 = (0..n)
            .map(|i| (actual[i] - self.predict(times[i])).abs())
            .sum();
        total_error / n as f64
    }

    /// Compute RMSE on test data.
    pub fn rmse(&self, times: &[f64], actual: &[f64]) -> f64 {
        let n = times.len().min(actual.len());
        if n == 0 {
            return 0.0;
        }
        let total_sq: f64 = (0..n)
            .map(|i| (actual[i] - self.predict(times[i])).powi(2))
            .sum();
        (total_sq / n as f64).sqrt()
    }

    fn compute_moving_average(data: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || data.is_empty() {
            return data.to_vec();
        }
        let n = data.len();
        let mut result = vec![0.0; n];
        let half = window / 2;
        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let count = hi - lo;
            result[i] = data[lo..hi].iter().sum::<f64>() / count as f64;
        }
        result
    }
}

/// Compare multiple agent models and return their MAEs.
pub fn compare_models(models: &[AgentBehaviorModel], times: &[f64], actual: &[f64]) -> Vec<(String, f64)> {
    models
        .iter()
        .map(|m| (m.agent_id.clone(), m.mean_absolute_error(times, actual)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_polynomial_agent() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let values: Vec<f64> = times.iter().map(|t| 2.0 * t + 1.0).collect();
        let model = AgentBehaviorModel::polynomial("agent1", &times, &values, 1);
        assert!(approx_eq(model.predict(2.5), 6.0, 1e-8));
        assert!(approx_eq(model.predict(0.0), 1.0, 1e-8));
    }

    #[test]
    fn test_spline_agent() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 1.0, 4.0, 9.0]; // x^2
        let model = AgentBehaviorModel::spline("agent2", &times, &values, &SplineBoundary::Natural);
        // Should pass through knots
        for i in 0..times.len() {
            assert!(approx_eq(model.predict(times[i]), values[i], 1e-8));
        }
    }

    #[test]
    fn test_piecewise_linear_agent() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![0.0, 2.0, 0.0];
        let model = AgentBehaviorModel::piecewise_linear("agent3", &times, &values);
        assert!(approx_eq(model.predict(0.5), 1.0, 1e-10));
        assert!(approx_eq(model.predict(1.5), 1.0, 1e-10));
    }

    #[test]
    fn test_moving_average_agent() {
        let data = vec![1.0, 3.0, 1.0, 3.0, 1.0];
        let model = AgentBehaviorModel::moving_average("agent4", &data, 3);
        // Middle values should be smoothed
        let mid = model.predict(2.0);
        assert!(mid > 0.0 && mid < 4.0, "smoothed value: {mid}");
    }

    #[test]
    fn test_mae() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![0.0, 2.0, 4.0];
        let model = AgentBehaviorModel::polynomial("test", &times, &values, 1);
        let mae = model.mean_absolute_error(&times, &values);
        assert!(mae < 0.01, "MAE should be near 0 for exact fit: {mae}");
    }

    #[test]
    fn test_rmse() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values: Vec<f64> = times.iter().map(|t| t * t).collect();
        let model = AgentBehaviorModel::polynomial("test", &times, &values, 2);
        let rmse = model.rmse(&times, &values);
        assert!(rmse < 0.01, "RMSE should be near 0: {rmse}");
    }

    #[test]
    fn test_compare_models() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let actual = vec![0.0, 1.0, 4.0, 9.0];

        let m1 = AgentBehaviorModel::polynomial("poly", &times, &actual, 1);
        let m2 = AgentBehaviorModel::polynomial("quad", &times, &actual, 2);

        let results = compare_models(&[m1, m2], &times, &actual);
        assert_eq!(results.len(), 2);
        // Quadratic should be better (lower MAE) than linear for x^2
        assert!(results[1].1 < results[0].1, "quad MAE should be < linear MAE");
    }

    #[test]
    fn test_agent_predict_out_of_range() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![1.0, 2.0, 3.0];
        let model = AgentBehaviorModel::piecewise_linear("test", &times, &values);
        assert!(approx_eq(model.predict(-1.0), 1.0, 1e-10));
        assert!(approx_eq(model.predict(5.0), 3.0, 1e-10));
    }
}
