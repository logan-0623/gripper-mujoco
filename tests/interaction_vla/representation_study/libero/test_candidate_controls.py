from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.libero.candidate_controls import mask_batch, action_metrics, postprocess_actions, validate_split
from interaction_vla.representation_study.libero.phase_edit import PhaseEdit
from interaction_vla.representation_study.libero.flow_trace import FlowEdit, install_flow_edit


def test_matched_masks_are_local_equal_area_and_do_not_mutate_input():
    image=torch.arange(3*12*12,dtype=torch.float32).reshape(1,3,12,12)/432
    batch={"observation.images.camera1":image,"observation.state":torch.ones(1,8)}
    spec={"states":{"s":{"camera":"observation.images.camera1","regions":{"target":[0,0,4,4],"control":[8,8,12,12]}}}}
    original=image.clone()
    for fill in ("mean","blur"):
        result=mask_batch(batch,["s"],spec,"target",fill)
        assert not torch.equal(result["observation.images.camera1"][:,:,:4,:4],original[:,:,:4,:4])
        torch.testing.assert_close(result["observation.images.camera1"][:,:,4:],original[:,:,4:])
        torch.testing.assert_close(image,original)
    spec["states"]["s"]["regions"]["control"]=[8,8,11,12]
    with pytest.raises(ValueError,match="areas"):
        mask_batch(batch,["s"],spec,"target","mean")


def test_action_window_and_postprocessing_contract():
    x=np.zeros((2,50,7)); x[:,10:]=100
    metrics=action_metrics(x,10)
    assert np.all(metrics["deployed_full_rms"]==0)
    assert np.all(metrics["plan_full_rms"]>0)
    def post(v):
        assert v.shape==(100,7)
        return v+1
    np.testing.assert_allclose(postprocess_actions(post,x,"cpu"),x+1)


def test_task_split_requires_unseen_validation_tasks():
    records = [NS(state_id="train", suite="spatial", task_id=0, source_episode_id="a"),
               NS(state_id="valid", suite="spatial", task_id=1, source_episode_id="b")]
    split = NS(assignments={"train": "train", "valid": "validation"})
    validate_split(records, split, ["train"], ["valid"], "task")
    records[1].task_id = 0
    with pytest.raises(ValueError, match="task leakage"):
        validate_split(records, split, ["train"], ["valid"], "task")
    validate_split(records, split, ["train"], ["valid"], "episode")
    records[1].source_episode_id = "a"
    with pytest.raises(ValueError, match="episode leakage"):
        validate_split(records, split, ["train"], ["valid"], "episode")


def test_phase_trigger_causal_quota_reset_and_observe_only():
    gate=PhaseEdit("contact",1)
    with pytest.raises(RuntimeError): gate.begin_chunk()
    gate.reset(NS(frame_index=0,finger_contact_groups=()))
    assert not gate.begin_chunk()
    gate.update(NS(frame_index=7,finger_contact_groups=("left",)))
    assert gate.begin_chunk()
    assert not gate.begin_chunk()
    assert gate.summary()["trigger_steps"]==[7]
    gate.reset(NS(frame_index=0,finger_contact_groups=()))
    assert not gate.summary()["triggered"]
    gate.observe_only=True
    gate.update(NS(frame_index=3,finger_contact_groups=("left",)))
    assert not gate.begin_chunk()
    assert gate.summary()["triggered"] and gate.summary()["edited_stage_calls"]==0
    before=PhaseEdit("pre_contact",3)
    before.reset(NS(frame_index=0,finger_contact_groups=()))
    assert before.begin_chunk()
    before.update(NS(frame_index=1,finger_contact_groups=("left",)))
    before.update(NS(frame_index=2,finger_contact_groups=()))
    assert not before.begin_chunk()


def test_phase_is_latched_for_whole_chunk_and_counts_actual_edits():
    layer=torch.nn.Identity(); policy=NS(config=NS(num_steps=3))
    gate=PhaseEdit("contact",1)
    gate.reset(NS(frame_index=4,finger_contact_groups=("left",)))
    edit=FlowEdit("expert_late",(0,2),torch.ones(2),.5)
    handle=install_flow_edit(policy,layer,edit,begin_chunk=gate.begin_chunk,on_edit=gate.on_edit)
    try:
        values=[layer(torch.zeros(1,1,2)) for _ in range(6)]
    finally: handle.remove()
    assert [float(x.sum()) for x in values]==[1,0,1,0,0,0]
    assert gate.summary()["edited_stage_calls"]==2


def test_phase_pilot_freezes_six_runs_and_checks_actual_state(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from interaction_vla.representation_study.libero import phase_pilot
    offline=tmp_path/"offline"
    for step in ("005000","025000"):
        folder=offline/step/"expert_late"; folder.mkdir(parents=True)
        (folder/"candidates.json").write_text('{}')
        (folder.parent/"effects.npz").write_bytes(b"fixture")
        (folder.parent/"report.json").write_text(json.dumps({"complete":True,"effects_sha256":phase_pilot.file_hash(folder.parent/"effects.npz")}))
    calls=[]
    def execute(command,check):
        assert check and "--environment-phase" in command and "--eval.n_episodes=1" in command
        calls.append(command)
        path=Path(command[command.index("--events-output")+1]); path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"episodes":[{"initial_state_id":10,"task_id":0,"phase_edit":{"trigger_steps":[]}}]}))
    monkeypatch.setattr(phase_pilot.subprocess,"run",execute)
    phase_pilot.run(offline,tmp_path/"checkpoints",tmp_path/"result")
    assert len(calls)==6
    assert sum("--observe-only" in c for c in calls)==2
    assert json.loads((tmp_path/"result/report.json").read_text())["complete"]


def test_context_bootstrap_is_task_stratified_and_reports_inconclusive_small_tasks():
    from interaction_vla.representation_study.libero.context_dependence import _bootstrap

    rows = [
        {"task": ["suite", 0], "interaction_episode": "a", "difference": 1.0},
        {"task": ["suite", 0], "interaction_episode": "b", "difference": 1.0},
        {"task": ["suite", 1], "interaction_episode": "c", "difference": -1.0},
        {"task": ["suite", 1], "interaction_episode": "d", "difference": -1.0},
    ]
    result = _bootstrap(rows, seed=0, samples=200)
    assert result["observed_task_macro"] == 0.0
    assert result["decision"] == "inconclusive"
    assert result["tasks"] == 2


def test_context_matching_prefers_same_episode_and_never_uses_effects():
    from interaction_vla.representation_study.libero.context_dependence import _match

    def row(state_id, frame, episode_id, contact):
        return NS(state_id=state_id, suite="suite", task_id=0,
                  source_episode_id=episode_id, frame_index=frame,
                  observation=NS(robot_state=np.zeros(2)),
                  labels=NS(contact=NS(gripper_target=contact),
                             geometry=NS(gripper_target_distance=0.1)))

    records = [row("i", 10, "ep", True), row("r_same", 11, "ep", False),
               row("r_other", 10, "other", False)]
    _, pairs, _ = _match(records, ["i", "r_same", "r_other"])
    assert pairs[0]["reference_state_id"] == "r_same"
    assert pairs[0]["same_episode"]
