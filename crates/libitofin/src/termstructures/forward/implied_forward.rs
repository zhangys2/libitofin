//! Implied forward price and discount factor estimation from option quotes.
//!
//! Provides model-free estimation of forward prices $F^*$ and discount factors $D^*$
//! from two-sided option quotes with bid-ask spreads across:
//! - European cash/physical vanilla options (SPX, NDX, etc.) via Put-Call Parity:
//!   $$C(K) - P(K) = D \cdot (F - K)$$
//! - Cryptocurrency inverse options (Deribit BTC/ETH) via coin-numeraire linear parity:
//!   $$C_{\text{coin}}(K) - P_{\text{coin}}(K) = D_{\text{coin}} \left( 1 - \frac{K}{F} \right)$$
//! - American equity/ETF options (SPY, AAPL) via model-free ATM strike filtering and
//!   Merton early-exercise arbitrage bounds.

use crate::errors::QlResult;
use crate::types::{DiscountFactor, Rate, Real, Time};
use crate::{fail, require};

/// Market convention defining the quoting currency and payoff structure.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MarketConvention {
    /// European cash or physically-settled vanilla options (e.g. SPX, NDX).
    #[default]
    EuropeanVanilla,
    /// Coin-margined inverse options (e.g. Deribit BTC/ETH) where payoffs and quotes are in cryptocurrency.
    CryptoInverse,
    /// American options with early-exercise strike corridor filtering.
    AmericanEquity {
        /// Underlying spot price, used for ATM moneyness corridor filtering.
        spot: Real,
        /// Optional annualized ATM volatility to scale the corridor: $|\ln(K/S)| \le \kappa \sigma \sqrt{T}$.
        atm_vol: Option<Real>,
    },
}

/// A two-sided quote pair at a single strike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionQuotePair {
    pub strike: Real,
    pub call_bid: Real,
    pub call_ask: Real,
    pub put_bid: Real,
    pub put_ask: Real,
}

impl OptionQuotePair {
    pub fn new(strike: Real, call_bid: Real, call_ask: Real, put_bid: Real, put_ask: Real) -> Self {
        Self {
            strike,
            call_bid,
            call_ask,
            put_bid,
            put_ask,
        }
    }

    /// True if all prices and strike are positive, finite, and non-inverted ($ask \ge bid$).
    pub fn is_valid(&self) -> bool {
        self.strike > 0.0
            && self.strike.is_finite()
            && self.call_bid > 0.0
            && self.call_bid.is_finite()
            && self.call_ask >= self.call_bid
            && self.call_ask.is_finite()
            && self.put_bid > 0.0
            && self.put_bid.is_finite()
            && self.put_ask >= self.put_bid
            && self.put_ask.is_finite()
    }

    #[inline]
    pub fn call_mid(&self) -> Real {
        (self.call_bid + self.call_ask) * 0.5
    }

    #[inline]
    pub fn put_mid(&self) -> Real {
        (self.put_bid + self.put_ask) * 0.5
    }

    #[inline]
    pub fn call_spread(&self) -> Real {
        self.call_ask - self.call_bid
    }

    #[inline]
    pub fn put_spread(&self) -> Real {
        self.put_ask - self.put_bid
    }

    #[inline]
    pub fn combined_spread(&self) -> Real {
        self.call_spread() + self.put_spread()
    }
}

/// Qualitative status indicating health of the estimated forward and market quotes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForwardStatus {
    /// Clean fit, all arbitrage and consistency bounds satisfied.
    #[default]
    Valid,
    /// Strict synthetic bid-ask quotes are crossed ($F_{\text{bid}}^{\text{strict}} > F_{\text{ask}}^{\text{strict}}$).
    WarningCrossedSyntheticQuotes,
    /// Box-spread parity violated across adjacent liquid strikes.
    WarningBoxSpreadArbitrage,
    /// Implied discount factor outside plausible interval $[0.50, 1.25]$.
    WarningImplausibleDiscountFactor,
    /// Forward deviates from spot beyond threshold $|\ln(F/S)| > \delta_{\max}$.
    WarningSpotDeviationExceeded,
    /// American options early-exercise bounds widened.
    DegradedAmericanBounds,
}

/// Detailed diagnostic metrics accompanying the forward estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardDiagnostics {
    pub status: ForwardStatus,
    pub total_pairs_received: usize,
    pub pairs_used: usize,
    pub pairs_pruned: usize,
    pub is_crossed: bool,
    pub box_arbitrage_violations: usize,
    pub wls_rmse: Real,
    pub min_spread: Real,
    pub max_spread: Real,
}

/// Configuration options for the implied forward estimator.
#[derive(Debug, Clone)]
pub struct ImpliedForwardConfig {
    /// External discount factor $D = e^{-r T}$. If None, solved jointly from quotes.
    pub discount_factor: Option<DiscountFactor>,
    /// Underlying spot price for deviation validation and fallback.
    pub spot: Option<Real>,
    /// Minimum required valid pairs (default: 2).
    pub min_pairs: usize,
    /// Maximum pairs used in robust estimation (default: 50).
    pub max_pairs: usize,
    /// Relative spread floor regularizer $\epsilon = \text{spread\_floor} \cdot \max(K, 1.0)$ (default: 1e-4).
    pub spread_floor: Real,
    /// Maximum allowable relative deviation from spot $|F/S - 1.0|$ (default: 0.10).
    pub max_spot_deviation: Option<Real>,
    /// Multiplier for American ATM corridor: $|\ln(K/S)| \le \kappa \sigma \sqrt{T}$ (default: 0.5).
    pub american_kappa: Real,
}

impl Default for ImpliedForwardConfig {
    fn default() -> Self {
        Self {
            discount_factor: None,
            spot: None,
            min_pairs: 2,
            max_pairs: 50,
            spread_floor: 1e-4,
            max_spot_deviation: Some(0.10),
            american_kappa: 0.5,
        }
    }
}

/// Result of the implied forward estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpliedForwardResult {
    /// Weighted point estimate $F^*$.
    pub forward: Real,
    /// Strict synthetic no-arbitrage bid bound ($\max_{i} F_{\text{bid}, i}$).
    pub forward_bid_strict: Real,
    /// Strict synthetic no-arbitrage ask bound ($\min_{i} F_{\text{ask}, i}$).
    pub forward_ask_strict: Real,
    /// Robust synthetic spread-weighted bid.
    pub forward_bid_robust: Real,
    /// Robust synthetic spread-weighted ask.
    pub forward_ask_robust: Real,
    /// Implied or anchored discount factor $D$.
    pub discount_factor: DiscountFactor,
    /// Implied annualized cost of carry $(r - q) = -\frac{\ln(D)}{T}$.
    pub implied_carry_rate: Option<Rate>,
    /// Diagnostics on quote quality, crossed quotes, and box arbitrage.
    pub diagnostics: ForwardDiagnostics,
}

impl ImpliedForwardResult {
    /// Returns true if the estimate has `ForwardStatus::Valid`.
    pub fn is_valid(&self) -> bool {
        self.diagnostics.status == ForwardStatus::Valid
    }
}

/// High-level solver for calculating implied forward and discount factors.
pub struct ImpliedForward;

impl ImpliedForward {
    /// Calculates implied forward and discount factor from quote pairs.
    pub fn calculate(
        quotes: &[OptionQuotePair],
        convention: MarketConvention,
        config: &ImpliedForwardConfig,
        expiry_years: Time,
    ) -> QlResult<ImpliedForwardResult> {
        require!(
            expiry_years > 0.0 && expiry_years.is_finite(),
            "expiry_years must be positive and finite, got {expiry_years}"
        );
        require!(
            !quotes.is_empty(),
            "at least one option quote pair is required"
        );

        let total_received = quotes.len();

        // 1. Initial sanity filter: drop non-positive or inverted quotes
        let mut valid_pairs: Vec<OptionQuotePair> =
            quotes.iter().copied().filter(|q| q.is_valid()).collect();

        if let Some(s) = config.spot {
            valid_pairs.retain(|q| q.call_mid() < 0.5 * s && q.put_mid() < 0.5 * s);
        }

        // 2. Convention-specific strike filtering
        if let MarketConvention::AmericanEquity { spot, atm_vol } = convention {
            require!(
                spot > 0.0 && spot.is_finite(),
                "AmericanEquity spot must be positive, got {spot}"
            );
            let vol = atm_vol.unwrap_or(0.30);
            let window = config.american_kappa * vol * expiry_years.sqrt();
            valid_pairs.retain(|q| (q.strike / spot).ln().abs() <= window.max(0.05));
        }

        require!(
            valid_pairs.len() >= config.min_pairs,
            "insufficient valid strike pairs: expected at least {}, found {}",
            config.min_pairs,
            valid_pairs.len()
        );

        // Sort by strike ascending
        valid_pairs.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cap to max_pairs closest to spot (or median strike)
        if valid_pairs.len() > config.max_pairs {
            let center = config
                .spot
                .unwrap_or_else(|| valid_pairs[valid_pairs.len() / 2].strike);
            valid_pairs.sort_by(|a, b| {
                let da = (a.strike - center).abs();
                let db = (b.strike - center).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            valid_pairs.truncate(config.max_pairs);
            valid_pairs.sort_by(|a, b| {
                a.strike
                    .partial_cmp(&b.strike)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let pairs_used = valid_pairs.len();
        let pairs_pruned = total_received - pairs_used;

        match convention {
            MarketConvention::CryptoInverse => Self::calculate_crypto_inverse(
                &valid_pairs,
                config,
                expiry_years,
                total_received,
                pairs_pruned,
            ),
            MarketConvention::EuropeanVanilla | MarketConvention::AmericanEquity { .. } => {
                Self::calculate_linear(
                    &valid_pairs,
                    convention,
                    config,
                    expiry_years,
                    total_received,
                    pairs_pruned,
                )
            }
        }
    }

    /// Parity calculation for European vanilla and American equity options.
    fn calculate_linear(
        pairs: &[OptionQuotePair],
        convention: MarketConvention,
        config: &ImpliedForwardConfig,
        expiry_years: Time,
        total_received: usize,
        mut pairs_pruned: usize,
    ) -> QlResult<ImpliedForwardResult> {
        let n = pairs.len();
        let mut min_spread = Real::INFINITY;
        let mut max_spread = 0.0;

        // Compute weights
        let mut weights = Vec::with_capacity(n);
        for q in pairs {
            let cs = q.call_spread();
            let ps = q.put_spread();
            let spread_sum = cs + ps;
            if spread_sum < min_spread {
                min_spread = spread_sum;
            }
            if spread_sum > max_spread {
                max_spread = spread_sum;
            }
            let floor = config.spread_floor * q.strike.max(1.0);
            let w = 1.0 / (cs * cs + ps * ps + floor * floor);
            weights.push(w);
        }

        let total_weight: Real = weights.iter().sum();
        require!(
            total_weight > 0.0 && total_weight.is_finite(),
            "sum of regression weights is non-positive or non-finite"
        );
        for w in &mut weights {
            *w /= total_weight;
        }

        let (mut discount_factor, mut forward, mut wls_rmse, mut status) = match config
            .discount_factor
        {
            Some(df) => {
                require!(
                    df > 0.0 && df.is_finite(),
                    "discount_factor must be positive"
                );
                let mut sorted_by_spread: Vec<(Real, Real)> = pairs
                    .iter()
                    .map(|q| {
                        let spread = q.combined_spread();
                        let f_i = q.strike + (q.call_mid() - q.put_mid()) / df;
                        (spread, f_i)
                    })
                    .collect();
                sorted_by_spread
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let count = sorted_by_spread.len().min(config.max_pairs);
                let mut f_vals: Vec<Real> = sorted_by_spread[..count].iter().map(|e| e.1).collect();
                f_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let med_f = if count % 2 == 1 {
                    f_vals[count / 2]
                } else {
                    0.5 * (f_vals[count / 2 - 1] + f_vals[count / 2])
                };

                let mut sum_sq_err = 0.0;
                for (i, q) in pairs.iter().enumerate() {
                    let y = q.call_mid() - q.put_mid();
                    let model_y = df * (med_f - q.strike);
                    let err = y - model_y;
                    sum_sq_err += weights[i] * err * err;
                }
                (df, med_f, sum_sq_err.sqrt(), ForwardStatus::Valid)
            }
            None => {
                // Joint estimation via WLS
                let (df, fwd, rmse, st) = Self::fit_wls(pairs, &weights)?;
                (df, fwd, rmse, st)
            }
        };

        // Two-pass outlier pruning if RMSE > 0 and pairs > min_pairs + 1
        if wls_rmse > 0.0 && pairs.len() > config.min_pairs + 1 {
            let threshold = 3.5 * wls_rmse;
            let mut filtered_pairs = Vec::with_capacity(n);
            let mut filtered_weights = Vec::with_capacity(n);
            let mut pruned_any = false;

            for (i, q) in pairs.iter().enumerate() {
                let y = q.call_mid() - q.put_mid();
                let model_y = discount_factor * (forward - q.strike);
                if (y - model_y).abs() <= threshold {
                    filtered_pairs.push(*q);
                    filtered_weights.push(weights[i]);
                } else {
                    pruned_any = true;
                    pairs_pruned += 1;
                }
            }

            if pruned_any && filtered_pairs.len() >= config.min_pairs {
                let sum_w: Real = filtered_weights.iter().sum();
                for w in &mut filtered_weights {
                    *w /= sum_w;
                }
                if let Some(df) = config.discount_factor {
                    let mut sum_f = 0.0;
                    for (i, q) in filtered_pairs.iter().enumerate() {
                        let y = q.call_mid() - q.put_mid();
                        sum_f += filtered_weights[i] * (q.strike + y / df);
                    }
                    forward = sum_f;
                } else if let Ok((df, fwd, rmse, st)) =
                    Self::fit_wls(&filtered_pairs, &filtered_weights)
                {
                    discount_factor = df;
                    forward = fwd;
                    wls_rmse = rmse;
                    if st != ForwardStatus::Valid {
                        status = st;
                    }
                }
            }
        }

        // Synthetic bounds calculation
        let mut forward_bid_strict = Real::NEG_INFINITY;
        let mut forward_ask_strict = Real::INFINITY;
        let mut forward_bid_robust = 0.0;
        let mut forward_ask_robust = 0.0;

        for (i, q) in pairs.iter().enumerate() {
            let f_bid = q.strike + (q.call_bid - q.put_ask) / discount_factor;
            let f_ask = q.strike + (q.call_ask - q.put_bid) / discount_factor;

            if f_bid > forward_bid_strict {
                forward_bid_strict = f_bid;
            }
            if f_ask < forward_ask_strict {
                forward_ask_strict = f_ask;
            }

            forward_bid_robust += weights[i] * f_bid;
            forward_ask_robust += weights[i] * f_ask;
        }

        let is_crossed = forward_bid_strict > forward_ask_strict;
        if is_crossed && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningCrossedSyntheticQuotes;
        }

        // Box spread parity checks
        let mut box_violations = 0;
        for i in 0..pairs.len().saturating_sub(1) {
            let q1 = pairs[i];
            let q2 = pairs[i + 1];
            let box_mid = (q1.call_mid() - q1.put_mid()) - (q2.call_mid() - q2.put_mid());
            let box_expected = discount_factor * (q2.strike - q1.strike);
            let tol = 0.5 * (q1.combined_spread() + q2.combined_spread());
            if (box_mid - box_expected).abs() > tol.max(0.10) {
                box_violations += 1;
            }
        }
        if box_violations > pairs.len() / 2 && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningBoxSpreadArbitrage;
        }

        // Spot deviation check
        let spot_ref = config.spot.or(match convention {
            MarketConvention::AmericanEquity { spot, .. } => Some(spot),
            _ => None,
        });
        if let (Some(s), Some(max_dev)) = (spot_ref, config.max_spot_deviation) {
            let is_exceeded = (forward / s - 1.0).abs() > max_dev && status == ForwardStatus::Valid;
            if is_exceeded {
                status = ForwardStatus::WarningSpotDeviationExceeded;
            }
        }

        // American early exercise bounds check
        if let MarketConvention::AmericanEquity { .. } = convention {
            let mut f_min = Real::NEG_INFINITY;
            let mut f_max = Real::INFINITY;
            for q in pairs {
                let lb = q.strike + (q.call_bid - q.put_ask) / discount_factor;
                let ee_put = q.strike * (1.0 - discount_factor).max(0.0);
                let ub = q.strike + (q.call_ask - (q.put_bid - ee_put)) / discount_factor;
                if lb > f_min {
                    f_min = lb;
                }
                if ub < f_max {
                    f_max = ub;
                }
            }
            if (forward < f_min || forward > f_max) && status == ForwardStatus::Valid {
                status = ForwardStatus::DegradedAmericanBounds;
            }
        }

        let implied_carry_rate = if discount_factor > 0.0 && expiry_years > 0.0 {
            Some(-discount_factor.ln() / expiry_years)
        } else {
            None
        };

        Ok(ImpliedForwardResult {
            forward,
            forward_bid_strict,
            forward_ask_strict,
            forward_bid_robust,
            forward_ask_robust,
            discount_factor,
            implied_carry_rate,
            diagnostics: ForwardDiagnostics {
                status,
                total_pairs_received: total_received,
                pairs_used: pairs.len(),
                pairs_pruned,
                is_crossed,
                box_arbitrage_violations: box_violations,
                wls_rmse,
                min_spread,
                max_spread,
            },
        })
    }

    /// Parity calculation for coin-margined inverse cryptocurrency options (Deribit style).
    fn calculate_crypto_inverse(
        pairs: &[OptionQuotePair],
        config: &ImpliedForwardConfig,
        expiry_years: Time,
        total_received: usize,
        pairs_pruned: usize,
    ) -> QlResult<ImpliedForwardResult> {
        let n = pairs.len();
        let mut min_spread = Real::INFINITY;
        let mut max_spread = 0.0;

        let mut weights = Vec::with_capacity(n);
        for q in pairs {
            let cs = q.call_spread();
            let ps = q.put_spread();
            let spread_sum = cs + ps;
            if spread_sum < min_spread {
                min_spread = spread_sum;
            }
            if spread_sum > max_spread {
                max_spread = spread_sum;
            }
            // Coin space spread floor
            let floor = config.spread_floor;
            let w = 1.0 / (cs * cs + ps * ps + floor * floor);
            weights.push(w);
        }

        let total_weight: Real = weights.iter().sum();
        require!(
            total_weight > 0.0 && total_weight.is_finite(),
            "sum of regression weights is non-positive or non-finite"
        );
        for w in &mut weights {
            *w /= total_weight;
        }

        // Coin space linear regression:
        // y_i = C_btc - P_btc = D_coin - (D_coin / F) * K_i = alpha + beta * K_i
        let mut mean_k = 0.0;
        let mut mean_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            mean_k += weights[i] * q.strike;
            mean_y += weights[i] * y;
        }

        let mut var_k = 0.0;
        let mut cov_k_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let dk = q.strike - mean_k;
            let dy = y - mean_y;
            var_k += weights[i] * dk * dk;
            cov_k_y += weights[i] * dk * dy;
        }

        require!(
            var_k > 1e-12,
            "degenerate strike distribution in crypto inverse calculation"
        );

        let beta = cov_k_y / var_k;
        let alpha = mean_y - beta * mean_k;

        let mut status = ForwardStatus::Valid;
        let (d_coin, forward) = if beta < -1e-12 {
            let f = -alpha / beta;
            if f <= 0.0 || !f.is_finite() {
                fail!("calculated negative or infinite crypto forward: {f}");
            }
            (alpha.clamp(0.5, 1.5), f)
        } else {
            status = ForwardStatus::WarningImplausibleDiscountFactor;
            // Fallback: solve F assuming alpha = 1.0
            (1.0, mean_k / (1.0 - mean_y).max(1e-4))
        };

        // RMSE
        let mut sum_sq_err = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let pred = alpha + beta * q.strike;
            let err = y - pred;
            sum_sq_err += weights[i] * err * err;
        }
        let wls_rmse = sum_sq_err.sqrt();

        // Synthetic bounds in coin space:
        // 1 - K / F_bid = C_ask - P_bid => F_bid = K / (1 - (C_ask - P_bid))
        let mut forward_bid_strict = Real::NEG_INFINITY;
        let mut forward_ask_strict = Real::INFINITY;
        let mut forward_bid_robust = 0.0;
        let mut forward_ask_robust = 0.0;

        for (i, q) in pairs.iter().enumerate() {
            let diff_ask = q.call_ask - q.put_bid;
            let diff_bid = q.call_bid - q.put_ask;

            let f_ask = if 1.0 - diff_ask > 1e-4 {
                q.strike / (1.0 - diff_ask)
            } else {
                q.strike
            };

            let f_bid = if 1.0 - diff_bid > 1e-4 {
                q.strike / (1.0 - diff_bid)
            } else {
                q.strike
            };

            if f_bid > forward_bid_strict {
                forward_bid_strict = f_bid;
            }
            if f_ask < forward_ask_strict {
                forward_ask_strict = f_ask;
            }

            forward_bid_robust += weights[i] * f_bid;
            forward_ask_robust += weights[i] * f_ask;
        }

        let is_crossed = forward_bid_strict > forward_ask_strict;
        if is_crossed && status == ForwardStatus::Valid {
            status = ForwardStatus::WarningCrossedSyntheticQuotes;
        }

        // Spot deviation check if spot given
        if let (Some(s), Some(max_dev)) = (config.spot, config.max_spot_deviation) {
            let is_exceeded = (forward / s - 1.0).abs() > max_dev && status == ForwardStatus::Valid;
            if is_exceeded {
                status = ForwardStatus::WarningSpotDeviationExceeded;
            }
        }

        // Effective USD discount factor D = S / F
        let discount_factor = if let Some(s) = config.spot {
            s / forward
        } else {
            d_coin
        };

        let implied_carry_rate = if discount_factor > 0.0 && expiry_years > 0.0 {
            Some(-discount_factor.ln() / expiry_years)
        } else {
            None
        };

        Ok(ImpliedForwardResult {
            forward,
            forward_bid_strict,
            forward_ask_strict,
            forward_bid_robust,
            forward_ask_robust,
            discount_factor,
            implied_carry_rate,
            diagnostics: ForwardDiagnostics {
                status,
                total_pairs_received: total_received,
                pairs_used: pairs.len(),
                pairs_pruned,
                is_crossed,
                box_arbitrage_violations: 0,
                wls_rmse,
                min_spread,
                max_spread,
            },
        })
    }

    /// Fits (discount_factor, forward) via weighted least squares regression:
    /// $y_i \equiv C_{\text{mid}, i} - P_{\text{mid}, i} = \alpha + \beta K_i$ with $\beta = -D, \alpha = D \cdot F$.
    fn fit_wls(
        pairs: &[OptionQuotePair],
        weights: &[Real],
    ) -> QlResult<(DiscountFactor, Real, Real, ForwardStatus)> {
        let mut mean_k = 0.0;
        let mut mean_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            mean_k += weights[i] * q.strike;
            mean_y += weights[i] * y;
        }

        let mut var_k = 0.0;
        let mut cov_k_y = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let dk = q.strike - mean_k;
            let dy = y - mean_y;
            var_k += weights[i] * dk * dk;
            cov_k_y += weights[i] * dk * dy;
        }

        require!(
            var_k > 1e-12,
            "degenerate strike distribution: variance of strikes is zero"
        );

        let beta = cov_k_y / var_k;
        let alpha = mean_y - beta * mean_k;

        let df_candidate = -beta;
        let mut status = ForwardStatus::Valid;

        let (discount_factor, forward) = if (0.50..=1.25).contains(&df_candidate) {
            (df_candidate, alpha / df_candidate)
        } else {
            status = ForwardStatus::WarningImplausibleDiscountFactor;
            // Fallback to D = 1.0
            (1.0, mean_k + mean_y)
        };

        // Compute RMSE
        let mut sum_sq_err = 0.0;
        for (i, q) in pairs.iter().enumerate() {
            let y = q.call_mid() - q.put_mid();
            let pred = discount_factor * (forward - q.strike);
            let err = y - pred;
            sum_sq_err += weights[i] * err * err;
        }
        let rmse = sum_sq_err.sqrt();

        Ok((discount_factor, forward, rmse, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_european_parity_unanchored() {
        let true_forward = 5050.0;
        let true_df = 0.985;
        let expiry = 0.25;

        let strikes = vec![4900.0, 4950.0, 5000.0, 5050.0, 5100.0, 5150.0, 5200.0];
        let mut quotes = Vec::new();

        for &k in &strikes {
            // Parity mid diff: C - P = df * (F - K)
            let parity_diff = true_df * (true_forward - k);
            let base_c = if parity_diff > 0.0 {
                parity_diff + 50.0
            } else {
                50.0
            };
            let base_p = base_c - parity_diff;

            // Add 1.0 bid-ask spread
            quotes.push(OptionQuotePair::new(
                k,
                base_c - 0.5,
                base_c + 0.5,
                base_p - 0.5,
                base_p + 0.5,
            ));
        }

        let config = ImpliedForwardConfig::default();
        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, expiry)
                .unwrap();

        assert!(res.is_valid());
        assert!(
            (res.forward - true_forward).abs() < 1e-4,
            "forward={}",
            res.forward
        );
        assert!(
            (res.discount_factor - true_df).abs() < 1e-4,
            "df={}",
            res.discount_factor
        );
        assert!(!res.diagnostics.is_crossed);
        assert_eq!(res.diagnostics.box_arbitrage_violations, 0);
        assert!(res.forward_bid_strict <= res.forward);
        assert!(res.forward <= res.forward_ask_strict);
    }

    #[test]
    fn test_perfect_crypto_inverse_parity() {
        let true_forward = 65000.0;
        let expiry = 0.1;
        let strikes = vec![
            60000.0, 62000.0, 64000.0, 65000.0, 66000.0, 68000.0, 70000.0,
        ];

        let mut quotes = Vec::new();
        for &k in &strikes {
            // Coin parity diff: C_btc - P_btc = 1.0 - K / F
            let diff_btc: Real = 1.0 - k / true_forward;
            let base_c = 0.05 + diff_btc.max(0.0);
            let base_p = base_c - diff_btc;

            // Spread of 0.002 BTC
            quotes.push(OptionQuotePair::new(
                k,
                base_c - 0.001,
                base_c + 0.001,
                base_p - 0.001,
                base_p + 0.001,
            ));
        }

        let config = ImpliedForwardConfig {
            spot: Some(64500.0),
            ..Default::default()
        };

        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::CryptoInverse, &config, expiry)
                .unwrap();

        assert!(res.is_valid());
        assert!(
            (res.forward - true_forward).abs() < 1.0,
            "forward={}",
            res.forward
        );
        assert!(!res.diagnostics.is_crossed);
    }

    #[test]
    fn test_crossed_synthetic_quotes_flagged() {
        let expiry = 0.1;
        // Strike with inverted call quotes
        let quotes = vec![
            OptionQuotePair::new(5000.0, 100.0, 101.0, 80.0, 81.0),
            // Bad quote with crossed synthetic bounds
            OptionQuotePair::new(5050.0, 95.0, 95.5, 50.0, 50.5),
            OptionQuotePair::new(5100.0, 50.0, 51.0, 100.0, 101.0),
        ];

        let config = ImpliedForwardConfig {
            discount_factor: Some(1.0),
            ..Default::default()
        };

        let res =
            ImpliedForward::calculate(&quotes, MarketConvention::EuropeanVanilla, &config, expiry)
                .unwrap();

        // Should return a result but flag warning
        assert!(res.diagnostics.status != ForwardStatus::Valid);
    }

    #[test]
    fn test_american_equity_atm_filtering() {
        let spot = 100.0;
        let expiry = 0.25;

        // Wide array of strikes from 50 to 150
        let strikes = vec![50.0, 70.0, 90.0, 95.0, 100.0, 105.0, 110.0, 130.0, 150.0];
        let mut quotes = Vec::new();
        for &k in &strikes {
            let diff = 100.0 - k;
            quotes.push(OptionQuotePair::new(
                k,
                (diff + 10.0).max(1.0),
                (diff + 11.0).max(2.0),
                10.0,
                11.0,
            ));
        }

        let config = ImpliedForwardConfig::default();
        let res = ImpliedForward::calculate(
            &quotes,
            MarketConvention::AmericanEquity {
                spot,
                atm_vol: Some(0.20),
            },
            &config,
            expiry,
        )
        .unwrap();

        // Wings should have been pruned by ATM corridor
        assert!(res.diagnostics.pairs_pruned > 0);
        assert!(res.diagnostics.pairs_used < strikes.len());
    }
}
