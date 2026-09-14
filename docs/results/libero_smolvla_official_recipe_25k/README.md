# SmolVLA official-recipe 25k capability check

Closed-loop evaluation of the locally trained 25,000-step checkpoint on
LIBERO Spatial tasks 0--3. Each task uses initial-state IDs 10--19 under the
same inference and event-recording contract as
`../libero_smolvla_official_reference_v2`.

| Task | Success | Contact | Stable grasp | Geometric lift | Supported lift | Drop | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8/10 | 10/10 | 8/10 | 9/10 | 8/10 | 0/10 | 0/10 |
| 1 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 2/10 | 2/10 |
| 2 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 0/10 |
| 3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 2/10 | 0/10 |
| **Total** | **38/40** | **40/40** | **38/40** | **39/40** | **38/40** | **4/40** | **2/40** |

The matched official checkpoint scored 30/40 on these same task/state pairs.
The observed paired success difference is +20 percentage points: nine pairs
changed from official failure to reproduced-model success and one changed in
the opposite direction. An exploratory episode-level paired bootstrap with
200,000 resamples and seed 2057736129 gives a percentile 95% interval of
[+7.5, +35.0] percentage points. This bounded 40-pair result does not establish
equivalence or superiority over the official model across LIBERO.

Training completed 25,000/25,000 steps in 5:54:46. Logged losses at steps
200/5k/10k/15k/20k/25k were 1.737/0.446/0.387/0.336/0.311/0.311. The evaluated
checkpoint was `checkpoints/025000/pretrained_model`; the launcher contract is
repository commit `e0631f5`.

Source archive SHA-256:
`df7e4839db7957cadd9ded90f5629e78c8e86bf68af6fc6f51ae1ea1da1fa07e`.
