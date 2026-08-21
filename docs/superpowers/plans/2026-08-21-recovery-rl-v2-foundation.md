# Recovery RL v2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, resumable recovery-RL protocol that calibrates a 30–50% SFT recovery baseline, validates the residual interface with Compact Oracle-State, screens PPO against SAC, and selects an anchoring configuration before any formal representation comparison.

**Architecture:** Preserve the completed v1 pipeline and add focused v2 modules under `interaction_vla/representation_study/rl/`. A provenance-bound case sampler reconstructs nominal, perturbation, and recovery starts; a 36D Oracle-State feeds independent actor and critic encoders; PPO and SAC implement one shared backend contract; immutable gates determine whether the formal ACT study may begin.

**Tech Stack:** Python 3.12, PyTorch 2.10, NumPy, MuJoCo, LeRobot 0.6.1, PyYAML, pytest.

---

## File map

Create:

- `interaction_vla/representation_study/rl/v2_config.py`: strict v2 YAML schema and loader.
- `interaction_vla/representation_study/rl/distributions.py`: split-bound case manifests and mixture sampling.
- `interaction_vla/representation_study/rl/oracle_state.py`: 36D codec and normalization.
- `interaction_vla/representation_study/rl/rewards.py`: terminal, potential, and residual reward terms.
- `interaction_vla/representation_study/rl/actors.py`: Oracle and latent residual actors.
- `interaction_vla/representation_study/rl/critics.py`: independent PPO value and SAC twin-Q modules.
- `interaction_vla/representation_study/rl/anchoring.py`: nominal replay and latent drift losses.
- `interaction_vla/representation_study/rl/replay.py`: CPU image replay shards for SAC and exact resume.
- `interaction_vla/representation_study/rl/snapshots.py`: immutable time snapshots.
- `interaction_vla/representation_study/rl/ppo_v2.py`: separated-critic PPO backend.
- `interaction_vla/representation_study/rl/sac.py`: standard squashed-Gaussian SAC backend.
- `interaction_vla/representation_study/rl/evaluation_v2.py`: paired nominal/recovery evaluator.
- `interaction_vla/representation_study/rl/gates.py`: distribution, backend, Oracle, and anchoring decisions.
- `interaction_vla/representation_study/rl/protocol.py`: stage orchestration.
- Mac and CUDA v2 configs and focused pytest files.

Modify:

- `interaction_vla/representation_study/cli.py`: add the `recovery-rl` command family.
- `interaction_vla/representation_study/rl/environment.py`: accept reconstructed cases without changing v1 defaults.
- `interaction_vla/physics_data.py`: expose reusable physical perturbation preparation.
- `README.md`: document gate order only after verification.

Do not modify completed outputs or v1 report schemas.

### Task 1: Strict v2 configuration and output isolation

**Files:**
- Create: `interaction_vla/representation_study/rl/v2_config.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_rl_v2_config.py`
- Create: `configs/representation_study/recovery_rl_v2_act_macos.yaml`
- Create: `configs/representation_study/recovery_rl_v2_act_linux_cuda.yaml`

- [ ] **Step 1: Write failing schema tests**

```python
def test_v2_config_rejects_v1_output_root(tmp_path: Path) -> None:
    raw = minimal_v2_mapping(tmp_path)
    raw["output_dir"] = "outputs/representation_study/icra"
    with pytest.raises(ValueError, match="immutable v1 root"):
        RecoveryRLV2Config.from_mapping(raw, config_path=tmp_path / "v2.yaml")


def test_v2_config_fixes_distribution_and_snapshot_contract(tmp_path: Path) -> None:
    config = RecoveryRLV2Config.from_mapping(
        minimal_v2_mapping(tmp_path), config_path=tmp_path / "v2.yaml"
    )
    assert config.distribution.probabilities == pytest.approx((0.50, 0.30, 0.20))
    assert config.snapshot_steps == (0, 4096, 8192, 12288, 16384, 20480)
    assert config.oracle_state_dim == 36
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_rl_v2_config.py
```

Expected: collection fails because `rl.v2_config` does not exist.

- [ ] **Step 3: Implement immutable dataclasses and loader**

```python
@dataclass(frozen=True)
class RecoveryDistributionConfig:
    recovery_probability: float
    perturbation_probability: float
    nominal_probability: float
    calibration_seed_count: int
    training_seed_count: int
    curve_case_count: int
    final_case_count: int
    severity_candidates: tuple[float, ...]
    minimum_accepted_per_kind: int

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (
            self.recovery_probability,
            self.perturbation_probability,
            self.nominal_probability,
        )


@dataclass(frozen=True)
class PPOV2Config:
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    actor_learning_rate: float
    value_learning_rate: float
    gae_lambda: float
    clip_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float


@dataclass(frozen=True)
class SACConfig:
    replay_capacity: int
    warmup_steps: int
    batch_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    temperature_learning_rate: float
    polyak: float
    updates_per_environment_step: int
    target_entropy: float


@dataclass(frozen=True)
class RecoveryRLV2Config:
    config_path: Path
    output_dir: Path
    bridge_config: Path
    sft_checkpoint: str
    continued_sft_checkpoint: str
    device: str
    distribution: RecoveryDistributionConfig
    screen_steps: int
    formal_steps: int
    snapshot_steps: tuple[int, ...]
    oracle_state_dim: int
    residual_scale: tuple[float, ...]
    gamma: float
    progress_coefficient: float
    residual_coefficient: float
    nominal_anchor_coefficient: float
    latent_anchor_coefficient: float
    ppo: PPOV2Config
    sac: SACConfig
    seed: int

    def __post_init__(self) -> None:
        if self.output_dir.as_posix().rstrip("/") == "outputs/representation_study/icra":
            raise ValueError("Recovery RL v2 cannot target the immutable v1 root")
        if self.oracle_state_dim != 36:
            raise ValueError("Compact Oracle-State must have width 36")
        if self.snapshot_steps != (0, 4096, 8192, 12288, 16384, 20480):
            raise ValueError("formal snapshot schedule is incompatible")
```

Use the existing strict pattern directly:

```python
def _only(raw: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(sorted(unknown)))
```

Validate all numeric fields for finiteness, positive counts, valid probabilities,
seven residual scales, and platform device values.

Register backend hyperparameters in immutable nested config records instead of
reading v1 defaults implicitly. PPO uses rollout steps 256, four update epochs,
minibatch 64, actor/value learning rates `3e-4`, GAE lambda `0.95`, clip coefficient
`0.2`, entropy coefficient `0.01`, and gradient norm `1.0`. SAC uses replay capacity
100,000, warmup 1,024, batch size 256, actor/critic/temperature learning rates
`3e-4`, Polyak coefficient `0.005`, one gradient update per environment step, and
target entropy `-7`. Both use gamma `0.99`; reject silent unknown parameters.

- [ ] **Step 4: Add the two fixed profiles**

The Mac profile contains:

```yaml
schema_version: recovery_rl_v2
output_dir: outputs/representation_study/icra_rl_v2
bridge_config: configs/lerobot_act_recovery_macos.yaml
sft_checkpoint: outputs/graph_control/graph_v2_pilot/runs/seed_0/flat/checkpoint
continued_sft_checkpoint: outputs/representation_study/icra/sft/act/continued_sft/checkpoint
device: auto
distribution:
  recovery_probability: 0.50
  perturbation_probability: 0.30
  nominal_probability: 0.20
  calibration_seed_count: 50
  training_seed_count: 200
  curve_case_count: 30
  final_case_count: 50
  severity_candidates: [0.50, 0.75, 1.00]
  minimum_accepted_per_kind: 10
screen_steps: 8192
formal_steps: 20480
snapshot_steps: [0, 4096, 8192, 12288, 16384, 20480]
oracle_state_dim: 36
residual_scale: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]
gamma: 0.99
progress_coefficient: 0.10
residual_coefficient: 0.01
nominal_anchor_coefficient: 1.0
latent_anchor_coefficient: 0.10
ppo:
  rollout_steps: 256
  update_epochs: 4
  minibatch_size: 64
  actor_learning_rate: 0.0003
  value_learning_rate: 0.0003
  gae_lambda: 0.95
  clip_coefficient: 0.20
  entropy_coefficient: 0.01
  max_grad_norm: 1.0
sac:
  replay_capacity: 100000
  warmup_steps: 1024
  batch_size: 256
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.0003
  temperature_learning_rate: 0.0003
  polyak: 0.005
  updates_per_environment_step: 1
  target_entropy: -7.0
seed: 2057736129
```

The CUDA profile changes only device, bridge config, and output root to
`outputs/representation_study/icra_rl_v2_cuda`.

- [ ] **Step 5: Run tests and commit**

Run the Task 1 test command; expect all tests to pass.

```bash
git add interaction_vla/representation_study/rl/v2_config.py \
  tests/interaction_vla/representation_study/test_recovery_rl_v2_config.py \
  configs/representation_study/recovery_rl_v2_act_macos.yaml \
  configs/representation_study/recovery_rl_v2_act_linux_cuda.yaml
git commit -m "feat: define isolated recovery RL v2 config"
```

### Task 2: Provenance-bound distribution manifests

**Files:**
- Create: `interaction_vla/representation_study/rl/distributions.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_distributions.py`

- [ ] **Step 1: Write failing manifest and sampler tests**

```python
def test_source_seed_never_crosses_distribution_partitions() -> None:
    manifest = build_case_manifest(seed=7, calibration=8, training=16, curve=6, final=6)
    groups = [set(manifest.source_seeds(name)) for name in manifest.partition_names]
    assert all(a.isdisjoint(b) for index, a in enumerate(groups) for b in groups[index + 1 :])


def test_mixture_sampler_is_resume_exact() -> None:
    sampler = RecoveryCaseSampler(fake_manifest(), probabilities=(0.5, 0.3, 0.2), seed=9)
    state = sampler.state_dict()
    expected = [sampler.next_case().case_id for _ in range(20)]
    sampler.load_state_dict(state)
    assert [sampler.next_case().case_id for _ in range(20)] == expected
```

- [ ] **Step 2: Verify RED**

Run the new test file; expect import failure for `rl.distributions`.

- [ ] **Step 3: Implement immutable records and canonical JSON**

```python
@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    partition: str
    family: Literal["recovery", "perturbation", "nominal"]
    source_seed: int
    variant_id: int
    object_count: int
    layout: str
    phase: str
    intervention_kind: str
    severity: float


@dataclass(frozen=True)
class RecoveryCaseManifest:
    schema_version: str
    distribution_version: str
    cases: tuple[RecoveryCase, ...]
    source_hash: str

    def partition(self, name: str) -> tuple[RecoveryCase, ...]:
        return tuple(case for case in self.cases if case.partition == name)
```

Use `np.random.SeedSequence` with separate integer tags for split, family, and
variant generation. Sort by case id before hashing canonical JSON.

- [ ] **Step 4: Implement the exact mixture sampler**

`RecoveryCaseSampler` validates probabilities sum to one, samples only from the
training partition, and serializes its NumPy bit-generator state plus manifest hash.
Loading a state for another manifest raises `ValueError`.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_distributions.py
git add interaction_vla/representation_study/rl/distributions.py \
  tests/interaction_vla/representation_study/test_recovery_distributions.py
git commit -m "feat: add split-bound recovery distributions"
```

### Task 3: Reconstructible recovery and perturbation resets

**Files:**
- Modify: `interaction_vla/representation_study/rl/environment.py`
- Modify: `interaction_vla/physics_data.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_case_runtime.py`

- [ ] **Step 1: Write failing runtime tests**

```python
@pytest.mark.parametrize("family", ["nominal", "perturbation", "recovery"])
def test_reset_case_reconstructs_finite_state(family: str, runtime_factory) -> None:
    runtime = runtime_factory()
    first = runtime.reset_case(example_case(family))
    first_qpos = runtime.env.data.qpos.copy()
    second = runtime.reset_case(example_case(family))
    assert np.array_equal(runtime.env.data.qpos, first_qpos)
    assert np.isfinite(first.oracle_state).all()
    assert second.case_id == first.case_id


def test_perturbation_uses_actions_not_direct_qpos_assignment(monkeypatch, runtime_factory) -> None:
    runtime = runtime_factory()
    monkeypatch.setattr(runtime.env, "set_state", lambda *_: pytest.fail("direct state write"), raising=False)
    runtime.reset_case(example_case("perturbation"))
```

- [ ] **Step 2: Verify RED**

Run the test file; expect `ResidualMujocoRuntime` has no `reset_case`.

- [ ] **Step 3: Add physical preparation helpers**

Refactor the existing recovery prefix code only enough to expose it. Add
`prefix_steps: int` to `PreparedRecoveryStart`, initialize a local counter before
its expert loop, increment it after each `env.step`, and return it with the accepted
start. This is an additive schema change; preserve every existing recovery field.
Construct one `PhysicsScriptedExpert(self.env.physics)` in
`ResidualMujocoRuntime.__init__` and reset it for every case.

Map the three admitted recovery kinds and the abstract perturbation phases without
implicit string arithmetic:

```python
RECOVERY_KIND_INDEX = {
    kind.value: index
    for index, kind in enumerate(PhysicsRecoveryKind)
    if kind is not PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE
}
PERTURBATION_TRIGGER_PHASE = {
    "approach_offset": PhysicsExpertPhase.APPROACH,
    "grasp_offset": PhysicsExpertPhase.CLOSE,
    "lift_offset": PhysicsExpertPhase.LIFT,
}


def recovery_spec_from_case(case: RecoveryCase) -> PhysicsRecoverySpec:
    try:
        kind_index = RECOVERY_KIND_INDEX[case.intervention_kind]
    except KeyError as error:
        raise ValueError(
            f"case has an unsupported recovery kind: {case.intervention_kind}"
        ) from error
    return make_physics_recovery_spec(
        case.source_seed, case.variant_id, kind_index=kind_index
    )


def advance_expert_to_phase(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    case: RecoveryCase,
) -> tuple[SceneSnapshot, int]:
    trigger = PERTURBATION_TRIGGER_PHASE[case.intervention_kind]
    snapshot = env.reset(
        seed=case.source_seed,
        object_count=case.object_count,
        layout_mode=case.layout,
    )
    expert.reset(seed=case.source_seed)
    prefix_steps = 0
    while expert.phase is not trigger:
        transition = env.step(
            expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        )
        prefix_steps += 1
        snapshot = transition.snapshot
        if transition.done:
            raise PhysicsRecoveryRejected(
                f"perturbation_trigger_not_reached:{transition.reason.value}"
            )
    return snapshot, prefix_steps


def apply_phase_perturbation(
    env: FrankaContactEnv,
    snapshot: SceneSnapshot,
    *,
    case: RecoveryCase,
) -> SceneSnapshot:
    rng = np.random.default_rng(
        np.random.SeedSequence((case.source_seed, case.variant_id, 0x50545632))
    )
    direction = rng.normal(size=2)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    action = np.zeros(7, dtype=np.float32)
    action[:2] = direction * min(float(case.severity), 1.0)
    result = env.advance_intervention(action, substeps=env.physics.substeps)
    if result.physics_failure is not None:
        raise PhysicsRecoveryRejected(
            f"physics_failure_during_{case.intervention_kind}:"
            f"{result.physics_failure}"
        )
    if result.controller_diagnostics is not None and result.controller_diagnostics.ik_limited:
        raise PhysicsRecoveryRejected(f"ik_limited_during_{case.intervention_kind}")
    return result.snapshot
```

Then expose the unified preparation record:

```python
@dataclass(frozen=True)
class PreparedInteractionStart:
    snapshot: SceneSnapshot
    case_id: str
    family: str
    intervention_kind: str
    severity: float
    prefix_steps: int


def prepare_interaction_start(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    case: RecoveryCase,
) -> PreparedInteractionStart:
    if case.family == "nominal":
        snapshot = env.reset(
            seed=case.source_seed,
            object_count=case.object_count,
            layout_mode=case.layout,
        )
        expert.reset(seed=case.source_seed)
        return PreparedInteractionStart(
            snapshot, case.case_id, case.family, case.intervention_kind,
            case.severity, 0,
        )
    if case.family == "recovery":
        prepared = prepare_physics_recovery_start(
            env,
            expert,
            spec=recovery_spec_from_case(case),
            object_count=case.object_count,
            source_split=case.partition,
            layout_mode=case.layout,
        )
        return PreparedInteractionStart(
            prepared.snapshot, case.case_id, case.family,
            case.intervention_kind, case.severity, prepared.prefix_steps,
        )
    snapshot, prefix_steps = advance_expert_to_phase(
        env, expert, case=case
    )
    snapshot = apply_phase_perturbation(
        env, snapshot, case=case
    )
    return PreparedInteractionStart(
        snapshot, case.case_id, case.family, case.intervention_kind,
        case.severity, prefix_steps,
    )
```

Recovery reuses `prepare_physics_recovery_start` for the three admitted kinds.
Perturbation advances the expert to the declared phase and applies bounded Cartesian
actions through `advance_intervention`. No direct qpos or object-pose assignment is
allowed.

- [ ] **Step 4: Add `reset_case` without changing v1 `reset`**

Factor common episode-counter initialization into `_finish_reset(snapshot)` and
instantiate `CompactOracleStateCodec` from the manifest-bound normalization in the
v2 runtime factory. Add:

```python
@dataclass(frozen=True)
class InteractionReset:
    case_id: str
    observation: dict[str, object]
    oracle_state: np.ndarray


def reset_case(self, case: RecoveryCase) -> InteractionReset:
    prepared = prepare_interaction_start(self.env, self.expert, case=case)
    self._finish_reset(prepared.snapshot)
    oracle = self.oracle_codec.encode_runtime(self.env, prepared, case)
    assert self.current_observation is not None
    return InteractionReset(case.case_id, self.current_observation, oracle)
```

The old `reset(...)` signature and behavior remain byte-compatible.

- [ ] **Step 5: Run focused tests and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_case_runtime.py \
  tests/interaction_vla/representation_study/test_rl_environment.py \
  tests/interaction_vla/test_physics_recovery.py
git add interaction_vla/representation_study/rl/environment.py \
  interaction_vla/physics_data.py \
  tests/interaction_vla/representation_study/test_recovery_case_runtime.py
git commit -m "feat: reconstruct recovery RL starts"
```

### Task 4: Compact Oracle-State and reward terms

**Files:**
- Create: `interaction_vla/representation_study/rl/oracle_state.py`
- Create: `interaction_vla/representation_study/rl/rewards.py`
- Create: `tests/interaction_vla/representation_study/test_oracle_state.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_rewards.py`

- [ ] **Step 1: Write failing Oracle width and invariance tests**

```python
def test_oracle_state_is_finite_width_36(scene_fixture) -> None:
    encoded = CompactOracleStateCodec().encode(**scene_fixture)
    assert encoded.shape == (36,)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()


def test_oracle_state_is_global_frame_invariant(scene_fixture) -> None:
    codec = CompactOracleStateCodec()
    original = codec.encode(**scene_fixture)
    transformed = codec.encode(**globally_transform(scene_fixture, yaw=0.7, translation=(1.0, -0.3, 0.0)))
    np.testing.assert_allclose(original, transformed, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Write failing reward tests**

```python
@pytest.mark.parametrize("reason,expected", [("success", 1.0), ("dropped", -1.0), ("wrong_object", -1.0), ("timeout", 0.0)])
def test_terminal_reward_signs(reason: str, expected: float) -> None:
    assert terminal_reward(reason) == expected


def test_reward_matches_registered_decomposition() -> None:
    terms = recovery_reward(
        reason="running", previous_potential=-0.5, next_potential=-0.4,
        residual=np.ones(7), residual_scale=np.full(7, 0.1), gamma=0.99,
    )
    assert terms.total == pytest.approx(0.10 * (0.99 * -0.4 + 0.5) - 0.01 * 0.07)
```

- [ ] **Step 3: Verify RED**

Run both new test files; expect missing modules.

- [ ] **Step 4: Implement the fixed Oracle slices**

```python
ORACLE_STATE_WIDTH = 36
ORACLE_SLICES = {
    "gripper_target": slice(0, 10),
    "target_receptacle": slice(10, 16),
    "interaction": slice(16, 20),
    "distractor": slice(20, 21),
    "phase": slice(21, 27),
    "intervention": slice(27, 34),
    "recovery": slice(34, 36),
}
```

Use the existing quaternion-to-rotation and 6D rotation codecs. Store normalization
constants in `OracleNormalization.to_json()`. Reject non-finite values, invalid
one-hot groups, and out-of-range normalized scalars.

- [ ] **Step 5: Implement auditable reward terms**

```python
@dataclass(frozen=True)
class RewardTerms:
    terminal: float
    progress: float
    residual: float

    @property
    def total(self) -> float:
        return self.terminal + self.progress + self.residual


def recovery_reward(
    *,
    reason: str,
    previous_potential: float,
    next_potential: float,
    residual: object,
    residual_scale: object,
    gamma: float,
    progress_coefficient: float = 0.10,
    residual_coefficient: float = 0.01,
) -> RewardTerms:
    progress = progress_coefficient * (gamma * next_potential - previous_potential)
    scaled = np.asarray(residual) * np.asarray(residual_scale)
    cost = -residual_coefficient * float(np.dot(scaled, scaled))
    result = RewardTerms(terminal_reward(reason), progress, cost)
    if not np.isfinite(result.total):
        raise ValueError("recovery reward is non-finite")
    return result
```

Call this only from v2 runtime steps. Preserve v1 `sparse_task_reward`.

- [ ] **Step 6: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_oracle_state.py \
  tests/interaction_vla/representation_study/test_recovery_rewards.py
git add interaction_vla/representation_study/rl/oracle_state.py \
  interaction_vla/representation_study/rl/rewards.py \
  interaction_vla/representation_study/rl/environment.py \
  tests/interaction_vla/representation_study/test_oracle_state.py \
  tests/interaction_vla/representation_study/test_recovery_rewards.py
git commit -m "feat: add Oracle state and recovery rewards"
```

### Task 5: Independent actors, critics, and anchoring

**Files:**
- Create: `interaction_vla/representation_study/rl/actors.py`
- Create: `interaction_vla/representation_study/rl/critics.py`
- Create: `interaction_vla/representation_study/rl/anchoring.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_actor_critic.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_anchoring.py`

- [ ] **Step 1: Write failing gradient-isolation tests**

```python
def test_critic_loss_never_reaches_actor() -> None:
    actor = LatentResidualActor(latent_dim=32)
    critic = OracleValueCritic(state_dim=36)
    critic(torch.randn(8, 36)).square().mean().backward()
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_sac_q_loss_does_not_update_visual_actor() -> None:
    actor = LatentResidualActor(latent_dim=32)
    critics = OracleTwinQ(state_dim=36, action_dim=7)
    residual = actor.sample(torch.randn(8, 32), deterministic=False).residual.detach()
    q1, q2 = critics(torch.randn(8, 36), residual)
    (q1.square().mean() + q2.square().mean()).backward()
    assert all(parameter.grad is None for parameter in actor.parameters())
```

- [ ] **Step 2: Write failing anchoring tests**

```python
def test_nominal_anchor_prefers_zero_residual() -> None:
    residual = torch.tensor([[0.2, -0.1]])
    assert nominal_residual_loss(residual).item() == pytest.approx(0.025)


def test_latent_anchor_stops_gradient_on_sft_target() -> None:
    current = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    latent_drift_loss(current, target).backward()
    assert current.grad is not None
    assert target.grad is None
```

- [ ] **Step 3: Verify RED**

Run both test files; expect missing actor, critic, and anchoring modules.

- [ ] **Step 4: Implement actor and critic contracts**

```python
@dataclass(frozen=True)
class ActorSample:
    residual: torch.Tensor
    log_prob: torch.Tensor


class ResidualActor(nn.Module):
    def sample(self, observation: torch.Tensor, *, deterministic: bool) -> ActorSample:
        hidden = self.encoder(observation)
        mean = self.mean(hidden)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        distribution = Normal(mean, std)
        raw = mean if deterministic else distribution.rsample()
        residual = torch.tanh(raw)
        correction = torch.log1p(-residual.square() + 1.0e-6)
        log_prob = (distribution.log_prob(raw) - correction).sum(dim=-1)
        return ActorSample(residual=residual, log_prob=log_prob)


class OracleResidualActor(ResidualActor):
    def __init__(self, state_dim: int = 36, action_dim: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))


class LatentResidualActor(ResidualActor):
    def __init__(self, latent_dim: int, action_dim: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 256), nn.Tanh())
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))


class OracleValueCritic(nn.Module):
    def __init__(self, state_dim: int = 36) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(self.encoder(state)).squeeze(-1)


class OracleTwinQ(nn.Module):
    def __init__(self, state_dim: int = 36, action_dim: int = 7) -> None:
        super().__init__()
        width = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((state, residual), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)
```

Move the Jacobian-corrected tanh Gaussian into the actor module without importing
the v1 shared actor-critic trunk. Critics own independent encoders and never receive
actor or ACT modules in their constructors.

- [ ] **Step 5: Implement anchoring and cache binding**

```python
def nominal_residual_loss(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().mean()


def latent_drift_loss(current: torch.Tensor, sft_target: torch.Tensor) -> torch.Tensor:
    return (current - sft_target.detach()).square().mean()
```

`NominalAnchorCache` binds dataset fingerprint, SFT checkpoint, tap id, case ids,
latent dtype, and payload SHA-256. Sampling is deterministic and resumable.

- [ ] **Step 6: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_actor_critic.py \
  tests/interaction_vla/representation_study/test_recovery_anchoring.py
git add interaction_vla/representation_study/rl/actors.py \
  interaction_vla/representation_study/rl/critics.py \
  interaction_vla/representation_study/rl/anchoring.py \
  tests/interaction_vla/representation_study/test_recovery_actor_critic.py \
  tests/interaction_vla/representation_study/test_recovery_anchoring.py
git commit -m "feat: separate and anchor recovery RL actors"
```

### Task 6: Replay and immutable snapshots

**Files:**
- Create: `interaction_vla/representation_study/rl/replay.py`
- Create: `interaction_vla/representation_study/rl/snapshots.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_snapshots.py`

- [ ] **Step 1: Write failing replay and snapshot tests**

```python
def test_replay_round_trip_preserves_sample_sequence(tmp_path: Path) -> None:
    replay = RecoveryReplay(root=tmp_path / "replay", capacity=32, seed=5)
    for index in range(20):
        replay.add(example_transition(index))
    state = replay.state_dict()
    expected = replay.sample(8).transition_ids
    replay.load_state_dict(state)
    assert replay.sample(8).transition_ids == expected


def test_completed_snapshot_is_immutable(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.save(step=4096, payload=example_payload(), binding="abc")
    with pytest.raises(FileExistsError, match="immutable"):
        store.save(step=4096, payload=example_payload(), binding="abc")
```

- [ ] **Step 2: Verify RED**

Run the snapshot test file; expect missing modules.

- [ ] **Step 3: Implement sharded CPU replay**

Store images as uint8, numeric arrays as float32, and task strings once per case.
Every completed shard has a canonical manifest hash. `state_dict()` contains
capacity, cursor, size, RNG, and ordered shard hashes. Missing or mismatched shards
block resume.

- [ ] **Step 4: Implement atomic snapshot storage**

```python
SNAPSHOT_STEPS = (0, 4096, 8192, 12288, 16384, 20480)

class SnapshotStore:
    def save(self, *, step: int, payload: Mapping[str, object], binding: str) -> Path:
        if step not in SNAPSHOT_STEPS:
            raise ValueError("snapshot step is not registered")
        destination = self.root / f"step_{step:06d}"
        if (destination / "COMPLETED").is_file():
            raise FileExistsError(f"snapshot is immutable: {destination}")
        return self._write_atomic(destination, payload=payload, binding=binding)
```

The payload binds actor, critic, target critics, optimizers, temperature, replay,
sampler, global RNG, config, distribution, and Oracle normalization. Write
`COMPLETED` only after the directory is atomically installed.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_snapshots.py
git add interaction_vla/representation_study/rl/replay.py \
  interaction_vla/representation_study/rl/snapshots.py \
  tests/interaction_vla/representation_study/test_recovery_snapshots.py
git commit -m "feat: checkpoint recovery replay exactly"
```

### Task 7: PPO v2 with an isolated value path

**Files:**
- Create: `interaction_vla/representation_study/rl/ppo_v2.py`
- Create: `tests/interaction_vla/representation_study/test_ppo_v2.py`

- [ ] **Step 1: Write a failing finite-update test**

```python
def test_ppo_v2_update_is_finite_and_critic_isolated() -> None:
    backend = make_tiny_ppo_v2(latent_dim=16, state_dim=36)
    report = backend.update(fake_on_policy_batch(64, latent_dim=16, state_dim=36))
    assert np.isfinite(report.policy_loss)
    assert np.isfinite(report.value_loss)
    assert report.critic_gradient_on_actor == 0.0
```

- [ ] **Step 2: Verify RED**

Run the PPO v2 test; expect import failure.

- [ ] **Step 3: Implement separate optimizer passes**

```python
critic_optimizer.zero_grad(set_to_none=True)
value_loss.backward()
critic_optimizer.step()

actor_optimizer.zero_grad(set_to_none=True)
(policy_loss + nominal_anchor + latent_anchor).backward()
actor_optimizer.step()
```

Compute advantages from detached critic values. Retain clipped probability ratio,
tanh Jacobian, entropy, KL, and clip-fraction diagnostics. Value loss must never be
part of the actor backward graph.

- [ ] **Step 4: Add exact-resume test**

Compare uninterrupted two-update CPU weights with a save/load between updates.
Require bitwise-equal actor and critic states and equal case-sampler sequence.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_ppo_v2.py
git add interaction_vla/representation_study/rl/ppo_v2.py \
  tests/interaction_vla/representation_study/test_ppo_v2.py
git commit -m "feat: isolate PPO policy and value learning"
```

### Task 8: SAC off-policy backend

**Files:**
- Create: `interaction_vla/representation_study/rl/sac.py`
- Create: `tests/interaction_vla/representation_study/test_sac.py`

- [ ] **Step 1: Write failing SAC equation tests**

```python
def test_sac_target_uses_minimum_target_q() -> None:
    target = sac_target(
        reward=torch.tensor([1.0]), done=torch.tensor([0.0]),
        next_log_prob=torch.tensor([-0.2]), q1=torch.tensor([3.0]),
        q2=torch.tensor([2.0]), alpha=torch.tensor(0.1), gamma=0.99,
    )
    assert target.item() == pytest.approx(1.0 + 0.99 * (2.0 + 0.02))


def test_one_sac_update_is_finite() -> None:
    backend = make_tiny_sac(latent_dim=16, state_dim=36)
    report = backend.update(fake_replay_batch(64, latent_dim=16, state_dim=36))
    assert all(np.isfinite(value) for value in asdict(report).values())
```

- [ ] **Step 2: Verify RED**

Run the SAC tests; expect import failure.

- [ ] **Step 3: Implement standard SAC update order**

Implement twin Oracle-State Q critics, Polyak targets, squashed-Gaussian residual
actor, and automatic entropy temperature. Update critics first, actor plus anchors
second, temperature third, and targets last. Q-loss inputs detach the actor and ACT
latent; actor loss permits gradients through the selected actor path only.

```python
def sac_target(
    *,
    reward: torch.Tensor,
    done: torch.Tensor,
    next_log_prob: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    with torch.no_grad():
        soft_value = torch.minimum(q1, q2) - alpha.detach() * next_log_prob
        return reward + gamma * (1.0 - done) * soft_value
```

The replay row contains current/next Oracle-State, current/next detached ACT latent,
residual, decomposed reward, terminal mask, case id, and distribution family. Never
reconstruct targets from observations under a changed policy checkpoint.

- [ ] **Step 4: Test isolation and exact resume**

Assert Q-only backward leaves actor and ACT gradients absent. Compare uninterrupted
and resumed CPU updates including target networks, temperature optimizer, replay,
and sampler states.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_sac.py
git add interaction_vla/representation_study/rl/sac.py \
  tests/interaction_vla/representation_study/test_sac.py
git commit -m "feat: add SAC residual adaptation backend"
```

### Task 9: Paired evaluator and scientific gates

**Files:**
- Create: `interaction_vla/representation_study/rl/evaluation_v2.py`
- Create: `interaction_vla/representation_study/rl/gates.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_rl_gates.py`

- [ ] **Step 1: Write failing gate tests**

```python
def test_oracle_gate_requires_recovery_gain_and_retention() -> None:
    passed = oracle_gate(sft_recovery=0.40, rl_recovery=0.52, sft_nominal=0.70, rl_nominal=0.61)
    failed = oracle_gate(sft_recovery=0.40, rl_recovery=0.55, sft_nominal=0.70, rl_nominal=0.59)
    assert passed.passed is True
    assert failed.passed is False


def test_backend_selection_uses_variance_when_auc_is_tied() -> None:
    result = select_backend(ppo=screen(auc=(0.50, 0.54)), sac=screen(auc=(0.515, 0.516)))
    assert result.selected_backend == "sac"


def test_calibration_selects_rate_nearest_band_center() -> None:
    selected = select_calibrated_severity(
        {0.50: 0.62, 0.75: 0.47, 1.00: 0.32}, target=(0.30, 0.50)
    )
    assert selected.severity == 0.75
```

- [ ] **Step 2: Verify RED**

Run the gate test file; expect missing evaluator and gate modules.

- [ ] **Step 3: Implement paired case evaluation**

```python
def evaluate_case_manifest(
    policy: EvaluationPolicy,
    runtime: ResidualMujocoRuntime,
    cases: Sequence[RecoveryCase],
    *,
    policy_seed: int,
) -> EvaluationReport:
```

The evaluator iterates the fixed case sequence, seeds policy noise, resets by case
id, and writes one row per episode containing family, intervention kind, source
seed, success, termination, steps, reward terms, residual norm, clipping,
smoothness, and IK scale. Aggregate nominal and recovery separately and reject
missing or duplicated case ids.

- [ ] **Step 4: Implement pure gate decisions**

```python
@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    reasons: tuple[str, ...]
    inputs: dict[str, object]
    selected_backend: str | None = None
```

Implement distribution `[0.30, 0.50]`, lexicographic backend selection, Oracle
`+0.10/-0.10`, and anchoring recovery-AUC/retention gates exactly as specified.
Bind every input report hash in the output JSON.

For calibration, evaluate frozen SFT on the same calibration source seeds for every
registered severity. Discard candidates failing the per-kind acceptance minimum;
among candidates inside `[0.30, 0.50]`, select recovery success closest to `0.40`,
then the lower severity on an exact tie. If no candidate is inside the band, write a
failing distribution gate without mutating the candidate grid. Freeze the chosen
severity, all rejected-case rows, and the final case manifest hash in
`gates/distribution.json`.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_rl_gates.py
git add interaction_vla/representation_study/rl/evaluation_v2.py \
  interaction_vla/representation_study/rl/gates.py \
  tests/interaction_vla/representation_study/test_recovery_rl_gates.py
git commit -m "feat: gate recovery RL evidence"
```

### Task 10: Gate-ordered protocol and CLI

**Files:**
- Create: `interaction_vla/representation_study/rl/protocol.py`
- Modify: `interaction_vla/representation_study/cli.py`
- Modify: `tests/interaction_vla/representation_study/test_cli.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_protocol.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

```python
@pytest.mark.parametrize("command", ["calibrate", "screen", "oracle-gate", "anchor-screen"])
def test_recovery_rl_commands_parse(command: str) -> None:
    args = build_parser().parse_args(["recovery-rl", command, "--config", "v2.yaml"])
    assert args.family == "recovery-rl"
    assert args.command == command
```

- [ ] **Step 2: Verify RED**

Run CLI and protocol tests; expect argparse rejection.

- [ ] **Step 3: Implement gate-ordered orchestration**

```python
def run_recovery_command(config: RecoveryRLV2Config, command: str, *, resume: bool) -> dict[str, object]:
    if command == "calibrate":
        return calibrate_distribution(config)
    require_passing_gate(config.output_dir / "gates/distribution.json")
    if command == "screen":
        return run_algorithm_screen(config, resume=resume)
    require_passing_gate(config.output_dir / "gates/backend.json")
    if command == "oracle-gate":
        return build_oracle_gate(config)
    require_passing_gate(config.output_dir / "gates/oracle.json")
    if command == "anchor-screen":
        return run_anchor_screen(config, resume=resume)
    raise ValueError(f"unknown recovery RL command: {command}")
```

Training commands accept `--resume`. A completed compatible stage returns its
report; an incompatible output raises without deletion or overwrite.

- [ ] **Step 4: Add unique integration tests**

Use tiny budgets with mocked rendering but real actor, critic, replay, snapshot, and
gate code. Cover one Oracle-PPO update and resume, one Oracle-SAC update and resume,
one paired evaluation, and deterministic backend selection. Do not duplicate unit
smokes in publication reports.

- [ ] **Step 5: Document command order**

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl calibrate --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl screen --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl oracle-gate --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl anchor-screen --config "$CONFIG"
```

- [ ] **Step 6: Run focused and full verification**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study

HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q
```

Expected: zero failures; environment-dependent skips remain explicit.

- [ ] **Step 7: Commit orchestration**

```bash
git add interaction_vla/representation_study/cli.py \
  interaction_vla/representation_study/rl/protocol.py \
  tests/interaction_vla/representation_study/test_cli.py \
  tests/interaction_vla/representation_study/test_recovery_protocol.py \
  README.md
git commit -m "feat: orchestrate recovery RL protocol gates"
```

## Foundation completion gate

Do not begin formal ACT implementation until:

- v1 outputs are byte-for-byte untouched;
- calibration produces an immutable manifest and SFT recovery success in `[0.30, 0.50]`;
- Oracle-PPO and Oracle-SAC finish or emit structured failure reports;
- backend selection is reproducible from JSON alone;
- the Oracle interface gate passes;
- an anchoring variant satisfies recovery AUC and nominal retention constraints;
- the full test suite passes with a writable `HF_HOME`.
