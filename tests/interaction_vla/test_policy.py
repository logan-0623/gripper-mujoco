from __future__ import annotations

import torch
import pytest

from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.models.encoders import scene_graphs_to_batch
from interaction_vla.models.policy import ActionPolicy, build_action_policy


def sample_inputs():
    env = KinematicTabletopEnv(max_objects=5)
    builder = SceneGraphBuilder(max_objects=5)
    graphs = []
    proprioception = []
    for seed, object_count in ((1, 2), (2, 3)):
        graphs.append(builder.build(env.reset(seed=seed, object_count=object_count)))
        proprioception.append(torch.from_numpy(env.proprioception()))
    return scene_graphs_to_batch(graphs), torch.stack(proprioception).float()


def test_all_representations_emit_bounded_four_dimensional_actions() -> None:
    batch, proprioception = sample_inputs()
    for representation in ("flat", "graph", "proprio"):
        policy = build_action_policy(
            representation=representation,
            max_nodes=8,
            max_edges=56,
            node_feature_dim=23,
            edge_feature_dim=10,
            graph_hidden_dim=24,
            embedding_dim=16,
            policy_hidden_dim=32,
            message_rounds=1,
        )
        actions = policy(None if representation == "proprio" else batch, proprioception)

        assert actions.shape == (2, 4)
        assert torch.all(actions[:, :3].abs() <= 0.04 + 1e-7)
        assert torch.all((actions[:, 3] >= 0.0) & (actions[:, 3] <= 1.0))


def test_flat_and_graph_use_the_same_action_head_architecture() -> None:
    flat = build_action_policy(representation="flat", graph_hidden_dim=24, embedding_dim=16)
    graph = build_action_policy(representation="graph", graph_hidden_dim=24, embedding_dim=16)

    assert isinstance(flat, ActionPolicy)
    assert isinstance(graph, ActionPolicy)
    flat_shapes = {name: value.shape for name, value in flat.action_head.state_dict().items()}
    graph_shapes = {name: value.shape for name, value in graph.action_head.state_dict().items()}
    assert flat_shapes == graph_shapes


def test_physics_policy_emits_bounded_seven_dimensional_commands() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    builder = SceneGraphBuilder(max_objects=5, feature_schema="physics_v2")
    graphs = [builder.build(env.reset(seed=seed, object_count=2)) for seed in (1, 2)]
    batch = scene_graphs_to_batch(graphs)
    proprioception = torch.zeros((2, 23), dtype=torch.float32)
    policy = build_action_policy(
        representation="graph",
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=18,
        graph_hidden_dim=24,
        embedding_dim=16,
        policy_hidden_dim=32,
        message_rounds=1,
        proprio_dim=23,
        action_dim=7,
        action_mode="cartesian_7d",
    )

    actions = policy(batch, proprioception)

    assert actions.shape == (2, 7)
    assert torch.all(actions[:, :6].abs() <= 1.0 + 1e-7)
    assert torch.all((actions[:, 6] >= 0.0) & (actions[:, 6] <= 1.0))


def test_action_policy_predicts_a_bounded_h8_chunk() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    builder = SceneGraphBuilder(max_objects=5, feature_schema="physics_v2")
    batch = scene_graphs_to_batch(
        [builder.build(env.reset(seed=seed, object_count=2)) for seed in (1, 2)]
    )
    proprioception = torch.zeros((2, 23), dtype=torch.float32)
    policy = build_action_policy(
        representation="graph",
        max_nodes=8,
        max_edges=56,
        node_feature_dim=23,
        edge_feature_dim=18,
        graph_hidden_dim=24,
        embedding_dim=16,
        policy_hidden_dim=32,
        message_rounds=1,
        proprio_dim=23,
        action_dim=7,
        action_mode="cartesian_7d",
        action_horizon=8,
    )

    chunk = policy.predict_action_chunk(batch, proprioception)

    assert chunk.shape == (2, 8, 7)
    assert torch.all(chunk[:, :, :6].abs() <= 1.0 + 1e-7)
    assert torch.all((chunk[:, :, 6] >= 0.0) & (chunk[:, :, 6] <= 1.0))
    assert policy(batch, proprioception).shape == (2, 8, 7)


def test_h1_forward_contract_is_backward_compatible() -> None:
    batch, proprioception = sample_inputs()
    policy = build_action_policy(
        representation="flat",
        graph_hidden_dim=24,
        embedding_dim=16,
        action_horizon=1,
    )

    assert policy.predict_action_chunk(batch, proprioception).shape == (2, 1, 4)
    assert policy(batch, proprioception).shape == (2, 4)


def test_flat_and_graph_share_identical_non_encoder_initialization_for_h8() -> None:
    kwargs = {
        "graph_hidden_dim": 24,
        "embedding_dim": 16,
        "policy_hidden_dim": 32,
        "proprio_dim": 23,
        "action_dim": 7,
        "action_mode": "cartesian_7d",
        "action_horizon": 8,
    }
    torch.manual_seed(13)
    flat = build_action_policy(representation="flat", **kwargs)
    torch.manual_seed(13)
    graph = build_action_policy(representation="graph", **kwargs)

    for name in ("action_head", "proprio_encoder"):
        flat_state = getattr(flat, name).state_dict()
        graph_state = getattr(graph, name).state_dict()
        assert flat_state.keys() == graph_state.keys()
        for key in flat_state:
            torch.testing.assert_close(flat_state[key], graph_state[key], atol=0, rtol=0)


def test_policy_rejects_an_action_dimension_mode_mismatch() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        build_action_policy(
            representation="graph",
            action_dim=4,
            action_mode="cartesian_7d",
        )


def test_policy_backpropagates_through_scene_and_proprioception() -> None:
    batch, proprioception = sample_inputs()
    proprioception.requires_grad_(True)
    policy = build_action_policy(
        representation="graph",
        graph_hidden_dim=24,
        embedding_dim=16,
        message_rounds=1,
    )

    policy(batch, proprioception).sum().backward()

    assert proprioception.grad is not None
    assert any(parameter.grad is not None for parameter in policy.scene_encoder.parameters())


def test_unknown_representation_fails_early() -> None:
    try:
        build_action_policy(representation="vector")
    except ValueError as error:
        assert "representation" in str(error)
    else:
        raise AssertionError("unknown representation should fail")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_flat_and_graph_training_steps_run_on_mps() -> None:
    batch, proprioception = sample_inputs()
    for representation in ("flat", "graph"):
        policy = build_action_policy(
            representation=representation,
            graph_hidden_dim=16,
            embedding_dim=8,
            policy_hidden_dim=8,
            message_rounds=1,
        ).to("mps")
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        actions = policy(batch.to("mps"), proprioception.to("mps"))
        loss = actions.square().mean()
        loss.backward()
        optimizer.step()

        assert actions.shape == (2, 4)
        assert torch.isfinite(actions).all()
