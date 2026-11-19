# Technical execution repairs

## Bootstrap random-number consumption order

During internal audit after initial primary metric generation, the Phase 1 bootstrap-weight generator was found to consume its RNG in study-first, replicate-second order. The frozen Phase 0.3 helper consumes it in replicate-first, study-second order. The initial procedure used the correct sampling units and distribution, but its realized seed-specific draws differed from the frozen implementation.

The implementation now calls the immutable Phase 0.3 patient_bootstrap_indices function directly for every replicate. All formal confidence intervals are recomputed with the same frozen seed 170329 and B=2000. Preliminary outputs are retained in logs/bootstrap_draw_order_repair. This repair aligns implementation with the existing protocol and does not select a bootstrap method using performance. The original implementation is retained in protocol/EXECUTED_CODE_SNAPSHOT/statistical_analysis.py.

## Worker allocation

The robustness coordinator was interrupted at its scheduler boundary to increase independent-case concurrency from two to six workers. Its currently running outer cases are allowed to finish and checkpoint before coordinator exit. Resume verifies every existing checkpoint's engine code hash, case signature and prediction/model hashes. Seeds, fitting settings and all numerical-library thread limits remain frozen. This changes resource allocation only.

## Probability clipping scope

Independent final audit found that the log-loss probability clipping was also applied to ranking metrics. This introduced artificial ties for extreme probabilities in the source-scale sensitivity. AUROC and average precision now use the original saved probabilities. Brier score and threshold evaluation also use original probabilities. The frozen 1e-15 clipping applies only to log loss. All formal metrics and confidence intervals are regenerated from unchanged predictions using the same frozen bootstrap draws. Pre-repair tables and implementation are retained in logs/probability_clipping_repair.
