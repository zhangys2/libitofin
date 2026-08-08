//! Swaption pricing engines.
//!
//! Port of `ql/pricingengines/swaption/`: the shared
//! [`BlackStyleSwaptionEngine`] template with its shifted-lognormal
//! ([`BlackSwaptionEngine`]) and normal ([`BachelierSwaptionEngine`])
//! instantiations, the model-based [`JamshidianSwaptionEngine`] (European
//! swaption pricing under Hull-White via the Jamshidian decomposition),
//! [`G2SwaptionEngine`] (European swaption under G2++), and
//! [`FdG2SwaptionEngine`] (finite-difference Bermudan/European under G2++).

mod blackswaptionengine;
mod discretizedswap;
mod discretizedswaption;
mod fdg2swaptionengine;
mod g2swaptionengine;
mod jamshidianswaptionengine;
mod treeswaptionengine;

pub use blackswaptionengine::{
    BachelierSpec, BachelierSwaptionEngine, Black76Spec, BlackStyleSpec, BlackStyleSwaptionEngine,
    BlackSwaptionEngine, CashAnnuityModel,
};
pub use discretizedswap::DiscretizedSwap;
pub use discretizedswaption::DiscretizedSwaption;
pub use fdg2swaptionengine::FdG2SwaptionEngine;
pub use g2swaptionengine::G2SwaptionEngine;
pub use jamshidianswaptionengine::JamshidianSwaptionEngine;
pub use treeswaptionengine::TreeSwaptionEngine;
