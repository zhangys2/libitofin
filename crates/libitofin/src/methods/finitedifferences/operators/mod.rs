//! Finite-difference operators.
//!
//! Port of `ql/methods/finitedifferences/operators/`. This ticket lands the
//! indexing primitives every operator is built on -
//! [`FdmLinearOpIterator`] and [`FdmLinearOpLayout`] - plus the
//! [`FdmLinearOp`] contract the operators themselves implement.

mod fdmblackscholesop;
mod fdmg2op;
mod fdmhestonop;
mod fdmlinearop;
mod fdmlinearopcomposite;
mod fdmlinearopiterator;
mod fdmlinearoplayout;
mod firstderivativeop;
mod ninepointlinearop;
mod secondderivativeop;
mod secondordermixedderivativeop;
mod triplebandlinearop;

pub use fdmblackscholesop::FdmBlackScholesOp;
pub use fdmg2op::FdmG2Op;
pub use fdmhestonop::FdmHestonOp;
pub use fdmlinearop::FdmLinearOp;
pub use fdmlinearopcomposite::FdmLinearOpComposite;
pub use fdmlinearopiterator::FdmLinearOpIterator;
pub use fdmlinearoplayout::FdmLinearOpLayout;
pub use firstderivativeop::first_derivative_op;
pub use ninepointlinearop::{NinePointCoeffs, NinePointLinearOp};
pub use secondderivativeop::second_derivative_op;
pub use secondordermixedderivativeop::second_order_mixed_derivative_op;
pub use triplebandlinearop::TripleBandLinearOp;
