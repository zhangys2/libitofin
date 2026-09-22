//! Quanto version of a vanilla option (`ql/instruments/quantovanillaoption.{hpp,cpp}`).

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionResults, OptionArguments, StrikedTypePayoff,
};
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Results from a quanto option calculation (the C++ `QuantoOptionResults<ResultsType>`).
#[derive(Default)]
pub struct QuantoOptionResults<R> {
    pub base: R,
    pub qvega: Option<Real>,
    pub qrho: Option<Real>,
    pub qlambda: Option<Real>,
}

impl<R: Results> Results for QuantoOptionResults<R> {
    fn reset(&mut self) {
        self.base.reset();
        self.qvega = None;
        self.qrho = None;
        self.qlambda = None;
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        self.base.as_instrument_results()
    }
}

/// Specialized results for quanto vanilla options (`ql/instruments/quantovanillaoption.hpp`).
pub type QuantoVanillaOptionResults = QuantoOptionResults<OneAssetOptionResults>;

/// Quanto vanilla option (`ql/instruments/quantovanillaoption.{hpp,cpp}`).
pub struct QuantoVanillaOption {
    base: InstrumentBase,
    payoff: Shared<dyn StrikedTypePayoff>,
    exercise: Shared<dyn Exercise>,
    settings: Shared<Settings<Date>>,
    greeks: Greeks,
    more_greeks: MoreGreeks,
    qvega: Option<Real>,
    qrho: Option<Real>,
    qlambda: Option<Real>,
}

macro_rules! impl_greek {
    ($fn:ident, $($path:ident).+, $desc:expr) => {
        #[doc = concat!("Sensitivity / greek `", $desc, "`.")]
        pub fn $fn(&mut self) -> QlResult<Real> {
            self.calculate()?;
            Self::greek(self.$($path).+, $desc)
        }
    };
}

impl QuantoVanillaOption {
    /// `QuantoVanillaOption(payoff, exercise)`.
    pub fn new(
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            payoff,
            exercise,
            settings,
            greeks: Greeks::default(),
            more_greeks: MoreGreeks::default(),
            qvega: None,
            qrho: None,
            qlambda: None,
        }
    }

    /// The payoff the option is written on.
    pub fn payoff(&self) -> &Shared<dyn StrikedTypePayoff> {
        &self.payoff
    }

    /// The exercise schedule.
    pub fn exercise(&self) -> &Shared<dyn Exercise> {
        &self.exercise
    }

    fn greek(value: Option<Real>, description: &str) -> QlResult<Real> {
        let Some(value) = value else {
            fail!("{description} not provided");
        };
        Ok(value)
    }

    impl_greek!(delta, greeks.delta, "delta");
    impl_greek!(gamma, greeks.gamma, "gamma");
    impl_greek!(theta, greeks.theta, "theta");
    impl_greek!(vega, greeks.vega, "vega");
    impl_greek!(rho, greeks.rho, "rho");
    impl_greek!(dividend_rho, greeks.dividend_rho, "dividend rho");
    impl_greek!(qvega, qvega, "quanto vega");
    impl_greek!(qrho, qrho, "quanto rho");
    impl_greek!(qlambda, qlambda, "quanto lambda");
}

impl Instrument for QuantoVanillaOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<OptionArguments>() else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(Shared::clone(&self.payoff));
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn setup_expired(&mut self) {
        self.base_mut().store_results(&InstrumentResults {
            value: Some(0.0),
            error_estimate: Some(0.0),
            ..Default::default()
        });
        self.greeks = Greeks {
            delta: Some(0.0),
            gamma: Some(0.0),
            theta: Some(0.0),
            vega: Some(0.0),
            rho: Some(0.0),
            dividend_rho: Some(0.0),
        };
        self.more_greeks = MoreGreeks::default();
        self.qvega = Some(0.0);
        self.qrho = Some(0.0);
        self.qlambda = Some(0.0);
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let any = results as &dyn Any;
        if let Some(qr) = any.downcast_ref::<QuantoVanillaOptionResults>() {
            self.greeks = qr.base.greeks;
            self.more_greeks = qr.base.more_greeks;
            self.qvega = qr.qvega;
            self.qrho = qr.qrho;
            self.qlambda = qr.qlambda;
            self.base_mut().store_results(&qr.base.instrument);
            return Ok(());
        }
        if let Some(r) = any.downcast_ref::<OneAssetOptionResults>() {
            self.greeks = r.greeks;
            self.more_greeks = r.more_greeks;
            let extras = &r.instrument.additional_results;
            self.qvega = extras.get("qvega").and_then(|v| v.downcast_ref()).copied();
            self.qrho = extras.get("qrho").and_then(|v| v.downcast_ref()).copied();
            self.qlambda = extras
                .get("qlambda")
                .and_then(|v| v.downcast_ref())
                .copied();
            self.base_mut().store_results(&r.instrument);
            return Ok(());
        }
        fail!("no greeks returned from pricing engine");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::PlainVanillaPayoff;
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::patterns::observable::{AsObservable, Observable};
    use crate::pricingengine::{GenericEngine, PricingEngine};
    use crate::pricingengines::vanilla::QuantoEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::May, 1998)
    }

    fn quote_handle(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    fn flat_rate(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )))
    }

    fn flat_vol(vol: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::new(
            today(),
            None,
            vol,
            Actual360::new(),
        )))
    }

    #[test]
    fn test_quanto_vanilla_option_haug_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(100.0),
            flat_rate(0.04),
            flat_rate(0.08),
            flat_vol(0.2),
        ));

        for (opt_type, expected_npv) in
            [(OptionType::Call, 5.3280 / 1.5), (OptionType::Put, 8.1636)]
        {
            let payoff = shared(PlainVanillaPayoff::new(opt_type, 105.0));
            let exercise = shared(EuropeanExercise::new(today() + 180));
            let mut opt = QuantoVanillaOption::new(payoff, exercise, Shared::clone(&settings));
            let engine = shared_mut(QuantoEuropeanEngine::new(
                Shared::clone(&process),
                flat_rate(0.05),
                flat_vol(0.10),
                quote_handle(0.3),
            ));
            opt.base_mut()
                .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

            let npv = opt.npv().unwrap();
            assert!(
                (npv - expected_npv).abs() <= 1e-4,
                "NPV error for {opt_type:?}: {npv} vs {expected_npv}"
            );
            for g in all_greeks(&mut opt) {
                assert!(g.unwrap().is_finite());
            }
        }
    }

    fn all_greeks(opt: &mut QuantoVanillaOption) -> [QlResult<Real>; 9] {
        [
            opt.delta(),
            opt.gamma(),
            opt.theta(),
            opt.vega(),
            opt.rho(),
            opt.dividend_rho(),
            opt.qvega(),
            opt.qrho(),
            opt.qlambda(),
        ]
    }

    fn dummy_option(settings: Shared<Settings<Date>>) -> QuantoVanillaOption {
        QuantoVanillaOption::new(
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0)),
            shared(EuropeanExercise::new(today() + 180)),
            settings,
        )
    }

    #[test]
    fn test_expired_quanto_option() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today() + 100);
        let mut opt = QuantoVanillaOption::new(
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0)),
            shared(EuropeanExercise::new(today())),
            settings,
        );
        assert!(opt.is_expired().unwrap());
        assert_eq!(opt.npv().unwrap(), 0.0);
        for g in all_greeks(&mut opt) {
            assert_eq!(g.unwrap(), 0.0);
        }
    }

    struct MockEngine<R: Results + Default> {
        base: GenericEngine<OptionArguments, R>,
        populate: fn(&mut R),
    }

    impl<R: Results + Default> AsObservable for MockEngine<R> {
        fn observable(&self) -> &Observable {
            self.base.observable()
        }
    }

    impl<R: Results + Default + 'static> PricingEngine for MockEngine<R> {
        fn arguments_mut(&mut self) -> &mut dyn Arguments {
            self.base.arguments_mut()
        }
        fn results(&self) -> &dyn Results {
            self.base.results()
        }
        fn reset(&mut self) {
            self.base.reset();
        }
        fn calculate(&mut self) -> QlResult<()> {
            (self.populate)(self.base.results_mut());
            Ok(())
        }
    }

    #[test]
    fn test_missing_quanto_greeks() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut opt = dummy_option(settings);

        let engine = shared_mut(MockEngine {
            base: GenericEngine::new(OptionArguments::default(), OneAssetOptionResults::default()),
            populate: |res| {
                res.instrument.value = Some(10.0);
                res.greeks.delta = Some(0.5);
            },
        });
        opt.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        assert_eq!(opt.npv().unwrap(), 10.0);
        assert_eq!(opt.delta().unwrap(), 0.5);
        for (g, msg) in [
            (opt.qvega(), "quanto vega not provided"),
            (opt.qrho(), "quanto rho not provided"),
            (opt.qlambda(), "quanto lambda not provided"),
            (opt.gamma(), "gamma not provided"),
        ] {
            assert_eq!(g.unwrap_err().message(), msg);
        }
    }

    #[test]
    fn test_quanto_vanilla_option_results() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut opt = dummy_option(settings);

        let engine = shared_mut(MockEngine {
            base: GenericEngine::new(
                OptionArguments::default(),
                QuantoVanillaOptionResults::default(),
            ),
            populate: |res| {
                res.base.instrument.value = Some(12.5);
                res.base.greeks.delta = Some(0.6);
                res.qvega = Some(1.23);
                res.qrho = Some(2.34);
                res.qlambda = Some(3.45);
            },
        });
        opt.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        assert_eq!(opt.npv().unwrap(), 12.5);
        assert_eq!(opt.delta().unwrap(), 0.6);
        assert_eq!(opt.qvega().unwrap(), 1.23);
        assert_eq!(opt.qrho().unwrap(), 2.34);
        assert_eq!(opt.qlambda().unwrap(), 3.45);
    }
}
