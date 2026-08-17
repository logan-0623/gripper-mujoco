# Graph Runtime Language Batch Fix

## Problem

`FrozenGraphRuntime.predict_token()` performs single-sample Graph v2 inference for
both cache generation and closed-loop rollout. `Vocabulary.encode()` returns
one-dimensional language token and mask arrays with shape `[tokens]`, while
`MuJoCoGraphEstimator.forward()` intentionally requires batched language inputs
with shape `[batch, tokens]`. Training does not expose the mismatch because the
PyTorch DataLoader adds the batch dimension.

The runtime currently batches RGB, state, and previous-graph inputs, but passes
language tokens and masks without a batch dimension. A real predicted Graph
checkpoint therefore raises `ValueError: language tokens and mask must share
shape [batch, tokens]` during Graph control cache generation.

## Design

Keep the tokenizer and model contracts unchanged. At the single-sample runtime
boundary, add a leading dimension to both encoded arrays before converting them
to tensors. The model will receive `[1, max_language_tokens]`, matching the batch
size already used by RGB, state, and previous graph.

Do not make the model accept unbatched tensors and do not change
`Vocabulary.encode()`: both alternatives would broaden shared interfaces and
could hide caller mistakes.

## Verification

Strengthen the existing `FrozenGraphRuntime` test double so it records and
asserts all multimodal input shapes. The regression test must fail on the current
implementation because language tensors are one-dimensional, then pass after
the runtime-boundary fix. Run the focused Graph control feature tests followed by
the complete test suite.

Existing Graph estimator checkpoints and comparison results remain valid. Cache
generation is atomic, so the failed attempt leaves no reusable partial cache;
the user reruns only `graph_control cache` after updating the code.
