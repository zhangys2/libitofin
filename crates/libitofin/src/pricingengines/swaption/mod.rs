//! Swaption pricing engines.
//!
//! Port of `ql/pricingengines/swaption/`: the shared
//! [`BlackStyleSwaptionEngine`] template with its shifted-lognormal
//! ([`BlackSwaptionEngine`]) and normal ([`BachelierSwaptionEngine`])
//! instantiations, the model-based [`JamshidianSwaptionEngine`] (European
//! swaption pricing under Hull-White via the Jamshidian decomposition),
//! [`G2SwaptionEngine`] (European swaption under G2++),
//! [`FdG2SwaptionEngine`] (finite-difference Bermudan/European under G2++),
//! [`FdHullWhiteSwaptionEngine`] (finite-difference Bermudan/European under
//! Hull–White), and [`TreeG2SwaptionEngine`] (tree Bermudan/European under G2++).

mod blackswaptionengine;
mod discretizedswap;
mod discretizedswaption;
mod fdg2swaptionengine;
mod fdhullwhiteswaptionengine;
mod g2swaptionengine;
mod jamshidianswaptionengine;
mod treeg2swaptionengine;
mod treeswaptionengine;

pub use blackswaptionengine::{
    BachelierSpec, BachelierSwaptionEngine, Black76Spec, BlackStyleSpec, BlackStyleSwaptionEngine,
    BlackSwaptionEngine, CashAnnuityModel,
};
pub use discretizedswap::DiscretizedSwap;
pub use discretizedswaption::DiscretizedSwaption;
pub use fdg2swaptionengine::FdG2SwaptionEngine;
pub use fdhullwhiteswaptionengine::FdHullWhiteSwaptionEngine;
pub use g2swaptionengine::G2SwaptionEngine;
pub use jamshidianswaptionengine::JamshidianSwaptionEngine;
pub use treeg2swaptionengine::TreeG2SwaptionEngine;
pub use treeswaptionengine::TreeSwaptionEngine;
