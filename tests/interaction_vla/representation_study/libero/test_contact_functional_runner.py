import json

from scripts import run_contact_functional_exploratory as runner


def test_runner_pairs_phase_observer_seed_and_frame_trace(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({
        "center_checkpoint_sha256": "checkpoint-hash",
        "candidates": [{"id": "contact_0", "tap": "expert_late"},
                       {"id": "matched_random_0", "tap": "expert_late"}],
    }))
    monkeypatch.setattr(runner, "_tree_sha256", lambda _: "checkpoint-hash")
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(runner, "summarize_closed_loop", lambda *_: {"complete": True})

    output = tmp_path / "output"
    runner.run(checkpoint, candidates, output, tasks=[0], initial_state_offset=20,
               episodes=1, initial_state_count=50, dose=.5)
    plan = json.loads((output / "plan.json").read_text())
    assert plan["deployed_prefix"] == 10
    assert len(commands) == 3
    assert all("--record-frame-trace" in command and
               "--policy-noise-seed-base" in command for command in commands)
    assert "--observe-only" in commands[0]
    assert "--observe-only" not in commands[1]
    assert "--match-candidate-id" in commands[2]
