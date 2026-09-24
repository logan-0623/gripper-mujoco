import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from interaction_vla.representation_study.libero.contact_subspace import _direction
from interaction_vla.representation_study.libero.joint_evidence import run


def test_contact_direction_is_finite_unit_and_train_label_dependent():
    rng=np.random.default_rng(4); x=rng.normal(size=(32,8)); y=(x[:,0]>0).astype(float)
    direction, score=_direction(x,y,1.)
    assert np.isfinite(direction).all() and np.isclose(np.linalg.norm(direction),1.)
    assert direction[0] > 0 and score > .1
    with pytest.raises(ValueError): _direction(x,np.ones(32),1.)


def test_joint_report_marks_missing_behavior_evidence(tmp_path):
    folder=tmp_path/"latent"; folder.mkdir(); (folder/"effects.npz").write_bytes(b"x")
    import hashlib
    digest=hashlib.sha256(b"x").hexdigest()
    report={"complete":True,"effects_sha256":digest,"target_id":"contact_0","control_ids":["matched_random_0"],"checkpoint_sha256":"x","summary":{
        "expert_late:contact_0/deployed_full_rms":{"task_macro":.2},
        "expert_late:matched_random_0/deployed_full_rms":{"task_macro":.1}}}
    (folder/"report.json").write_text(json.dumps(report))
    result=run([folder/"report.json"],output=tmp_path/"joint.json")
    assert result["latent"][0]["delta_A_contact_minus_random"] == pytest.approx(.1)
    assert result["behavior"]["delta_SR_interaction_minus_background"] is None
