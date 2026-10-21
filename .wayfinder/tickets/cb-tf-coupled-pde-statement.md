---
title: Tsiveriotis–Fernandes coupled PDE statement
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
resolution: "Discretize the Tsiveriotis–Fernandes value split (equity E at r, cash C at r+s, U=E+C). Do not copy the blended discount in tf_stepback. Flat-spread agreement is the s=0 and never-converted oracles; a residual at partial conversion is the Goldman-versus-TF gap."
map: ../maps/convertible-bond-equity-credit.md
---

# Tsiveriotis–Fernandes coupled PDE statement

## Question

What coupled initial-boundary-value problem — unknowns, the \(r\) versus \(r+s\) discounts, and the terminal, conversion, call, and put conditions — should the finite-difference engine discretize so a flat credit spread agrees with `BinomialConvertibleEngine`?

## Context

`DiscretizedConvertible` already blends one discount, \(p_{\mathrm{conv}} r + (1-p_{\mathrm{conv}})(r+s)\), on a Cox–Ross–Rubinstein tree. Tsiveriotis and Fernandes (1998) write a coupled system that splits credit-risky cash flows from risk-free conversion rights. This ticket states the continuous problem the new engine will discretize. It does not choose the mesh or the code layout. The contract is already settled by [PDE contract features](cb-pde-contract-features.md): voluntary conversion, issuer call, holder put, and fixed coupons. Coefficients are already settled by [Flat versus term-structure coefficients](cb-flat-versus-term-structure.md): flat \(r\), \(q\), and \(\sigma\) frozen at the conversion exercise date. Soft call, discrete dividends, and floating coupons are outside this engine.

## Resolution

Resolved 2026-10-19 by [TF coupled PDE statement](6c557a04-2921-4729-92f2-520afe9d0d40).

Discretize the Tsiveriotis–Fernandes value split. Do not discretize the blended discount in `tf_stepback`.

Stock: \(dS=(r-q)S\,dt+\sigma S\,dW\). Operator \(\mathcal{L}=\partial_t+(r-q)S\partial_S+\tfrac12\sigma^2 S^2\partial_{SS}\). Unknowns \(E\) (equity, discounted at \(r\)) and \(C\) (cash, discounted at \(r+s\)). Price \(U=E+C\). Continuation:

\[
\mathcal{L}E-rE=0,\qquad \mathcal{L}C-(r+s)C=0.
\]

Equivalently \(\mathcal{L}U-rU-sC=0\) plus the cash equation. \(S\downarrow 0\): \(E\to 0\), \(C\) is the straight bond at \(r+s\). \(S\uparrow\infty\): \(C\to 0\), \(E\sim\kappa S\).

At maturity, with \(N^\star\) the call/put-adjusted redemption plus any maturity coupon: \(U=\max(\kappa S, N^\star)\). Converted nodes are \((\kappa S, 0)\); cash nodes are \((0, N^\star)\).

On an exercise date, in engine order: issuer call \(\min(U,\max(K_{\mathrm{call}},\kappa S))\) when conversion is alive (cash call sets \(E=0\)); holder put \(\max(U,K_{\mathrm{put}})\) (binding put is cash); fixed coupon added to \(C\) only; voluntary conversion \(\max(U,\kappa S)\) sets \(E=\kappa S\), \(C=0\). American conversion holds throughout the exercise interval; European and Bermudan conversion only on exercise dates. Call, put, and coupons stay on their dates. \(K_{\mathrm{call}}\) and \(K_{\mathrm{put}}\) are the dirty strikes from `ConvertibleBondBase::setup_arguments`.

Flat-spread comparison is total value \(U\) at the settlement spot. It matches the binomial engine when \(s=0\) and when conversion never binds. Where \(s>0\) and conversion is uncertain, the tree discounts at \(r+s(1-p)\) and this PDE discounts \(E\) at \(r\) and \(C\) at \(r+s\). That residual is the Goldman-versus-Tsiveriotis–Fernandes gap. A local hazard would replace \(s\) in the cash discount only; \(\lambda(S)\) is not part of this statement.
