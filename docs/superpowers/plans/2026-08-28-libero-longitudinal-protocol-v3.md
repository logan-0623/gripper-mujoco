# LIBERO Longitudinal Protocol v3 Implementation Plan

1. Add failing tests for checkpoint discovery, condition planning, CLI routing, and
   cross-condition runtime validation.
2. Generalize the existing latent extractor to accept an explicit checkpoint and
   output directory while preserving protocol-v2 behavior.
3. Add `longitudinal plan`, `longitudinal extract`, and `longitudinal inspect` commands.
4. Record a deterministic runtime fingerprint in each protocol-v3 cache/report.
5. Run targeted and full LIBERO representation-study tests.
6. Document the exact server commands and mark execution-dependent evidence as not
   yet run.
