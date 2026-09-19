//! Libor market model covariance building blocks.
//!
//! Port of `ql/legacy/libormarketmodels/` pieces exercised by
//! `libormarketmodel.cpp` `testSimpleCovarianceModels`. Process, cap/swaption
//! engines, and calibration are deferred.

mod lfmcovarproxy;
mod lmexpcorrmodel;
mod lmlinexpvolmodel;

pub use lfmcovarproxy::LfmCovarianceProxy;
pub use lmexpcorrmodel::LmExponentialCorrelationModel;
pub use lmlinexpvolmodel::LmLinearExponentialVolatilityModel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{Shared, shared};
    use crate::types::{Real, Size, Time};

    /// `libormarketmodel.cpp` `testSimpleCovarianceModels` (corr + vol + proxy).
    #[test]
    fn simple_covariance_models() {
        let size: Size = 10;
        let tol = 1.0e-14;

        let corr = shared(LmExponentialCorrelationModel::new(size, 0.1).unwrap());
        let c = corr.correlation(0.0);
        // Pin ρ_ij = exp(-β|i-j|) (not just C ≈ L Lᵀ).
        assert!((c[(0, 0)] - 1.0).abs() <= tol);
        assert!((c[(0, 1)] - (-0.1_f64).exp()).abs() <= tol);
        assert!((c[(2, 5)] - (-0.3_f64).exp()).abs() <= tol);
        let recon = c - &(corr.pseudo_sqrt(0.0) * &corr.pseudo_sqrt(0.0).transpose());
        for i in 0..size {
            for j in 0..size {
                assert!(
                    recon[(i, j)].abs() <= tol,
                    "corr recon[{i},{j}]={}",
                    recon[(i, j)]
                );
            }
        }

        let fixing_times: Vec<Time> = (0..size).map(|i| 0.5 * i as Real).collect();
        let (a, b, c_param, d) = (0.2, 0.1, 2.1, 0.3);
        let vola = shared(
            LmLinearExponentialVolatilityModel::new(fixing_times.clone(), a, b, c_param, d)
                .unwrap(),
        );
        // Scalar path: live forward and expired forward.
        assert!(
            (vola.volatility_i(4, 0.5).unwrap()
                - ((a * (2.0 - 0.5) + d) * (-b * (2.0 - 0.5)).exp() + c_param))
                .abs()
                <= tol
        );
        assert_eq!(vola.volatility_i(1, 1.0).unwrap(), 0.0);
        assert!(vola.volatility_i(size, 0.0).is_err());

        let covar = LfmCovarianceProxy::new(Shared::clone(&vola), Shared::clone(&corr)).unwrap();

        let mut t = 0.0;
        while t < 4.6 {
            let recon =
                &covar.covariance(t) - &(&covar.diffusion(t) * &covar.diffusion(t).transpose());
            for i in 0..size {
                for j in 0..size {
                    assert!(
                        recon[(i, j)].abs() <= tol,
                        "covar recon t={t} [{i},{j}]={}",
                        recon[(i, j)]
                    );
                }
            }
            let volatility = vola.volatility(t);
            for k in 0..size {
                let expected = if (k as Real) > 2.0 * t {
                    let t_fix = fixing_times[k];
                    (a * (t_fix - t) + d) * (-b * (t_fix - t)).exp() + c_param
                } else {
                    0.0
                };
                assert!(
                    (expected - volatility[k]).abs() <= tol,
                    "vol t={t} k={k}: got {} expected {expected}",
                    volatility[k]
                );
            }
            t += 0.31;
        }
    }
}
