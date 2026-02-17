; Determinism probe for SMT solvers.
; Intentionally minimal: check-sat only.
(set-logic QF_UF)
(declare-fun a () Bool)
(assert a)
(check-sat)
