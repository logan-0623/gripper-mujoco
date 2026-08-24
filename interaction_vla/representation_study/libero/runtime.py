from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .annotation import PrivilegedFrame
from .contacts import SemanticContactMap, contact_labels_from_pairs
from .replay import relocate_model_asset_paths
from .task_semantics import GoalAtom, TaskSemantics, TaskSemanticsRegistry


def _model_name2id(model: object, kind: str, name: str) -> int:
    legacy = getattr(model, f"{kind}_name2id", None)
    if callable(legacy):
        return int(legacy(name))
    accessor = getattr(model, kind, None)
    if callable(accessor):
        return int(accessor(name).id)
    try:
        import mujoco

        object_type = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "site": mujoco.mjtObj.mjOBJ_SITE,
        }[kind]
        identifier = int(mujoco.mj_name2id(model, object_type, name))
    except (ImportError, KeyError, TypeError) as error:
        raise ValueError(f"cannot resolve MuJoCo {kind} name: {name}") from error
    if identifier < 0:
        raise ValueError(f"cannot resolve MuJoCo {kind} name: {name}")
    return identifier


def _model_id2name(model: object, kind: str, identifier: int) -> str | None:
    legacy = getattr(model, f"{kind}_id2name", None)
    if callable(legacy):
        value = legacy(identifier)
        return None if value is None else str(value)
    accessor = getattr(model, kind, None)
    if callable(accessor):
        value = accessor(identifier).name
        return None if value is None else str(value)
    try:
        import mujoco

        object_type = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "site": mujoco.mjtObj.mjOBJ_SITE,
        }[kind]
        value = mujoco.mj_id2name(model, object_type, int(identifier))
    except (ImportError, KeyError, TypeError) as error:
        raise ValueError(f"cannot resolve MuJoCo {kind} id: {identifier}") from error
    return None if value is None else str(value)


def _evaluate_raw_goal_state(
    domain: object, goal_states: tuple[tuple[str, ...], ...]
) -> bool:
    """Evaluate the original LIBERO predicates without ontology normalization.

    The measurement schema calls the BDDL predicate ``in`` "inside", but
    LIBERO's predicate registry only accepts the original ``in`` token.
    Keeping simulator evaluation on the raw BDDL atoms prevents that semantic
    normalization from changing environment behavior.
    """
    if not goal_states:
        raise ValueError("LIBERO task has no raw goal states to evaluate")
    evaluator = getattr(domain, "_eval_predicate", None)
    if not callable(evaluator):
        raise ValueError("LIBERO domain does not expose predicate evaluation")
    return all(bool(evaluator(list(state))) for state in goal_states)


class LiberoOffscreenSimulator:
    """Thin, lazy wrapper around the official LIBERO replay environment."""

    replay_validation_vector_name = "qpos"

    def __init__(
        self,
        *,
        suite: str,
        task_id: int,
        task_name: str,
        language: str,
        bddl_path: str | Path,
        seed: int,
        control_freq: int,
    ) -> None:
        try:
            import robosuite
            from libero.libero import get_assets_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as error:  # pragma: no cover - Linux optional dependency
            raise RuntimeError("hf-libero is required for deterministic replay") from error
        self._robosuite_root = Path(robosuite.__file__).resolve().parent
        self._libero_assets_root = Path(get_assets_path()).resolve()
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=256,
            camera_widths=256,
            camera_names=["agentview", "robot0_eye_in_hand"],
            control_freq=control_freq,
            hard_reset=True,
            ignore_done=True,
        )
        self.env.seed(seed)
        self._observation = self.env.reset()
        domain = getattr(self.env, "env", self.env)
        self._raw_goal_states = tuple(
            tuple(str(item) for item in value)
            for value in domain.parsed_problem["goal_state"]
        )
        raw_goals = tuple(
            GoalAtom(value[0].lower(), value[1:])
            for value in self._raw_goal_states
        )
        if not raw_goals:
            raise ValueError(f"LIBERO task {task_name} has no parsed goal atoms")
        source = str(getattr(domain, "workspace_name", "kitchen_table"))
        distractors = tuple(
            str(name)
            for name in getattr(domain, "objects_dict", {})
            if str(name) not in {raw_goals[0].arguments[0], *raw_goals[0].arguments[1:]}
        )
        self.semantics = TaskSemanticsRegistry.default().resolve(
            suite=suite,
            task_id=task_id,
            task_name=task_name,
            language=language,
            goal_atoms=raw_goals,
            source=source,
            distractors=distractors,
        )

    @property
    def domain(self):
        return getattr(self.env, "env", self.env)

    @property
    def sim(self):
        value = getattr(self.env, "sim", None)
        return value if value is not None else self.domain.sim

    @property
    def robots(self):
        value = getattr(self.env, "robots", None)
        return value if value is not None else self.domain.robots

    def close(self) -> None:
        self.env.close()

    def reset_from_xml_string(self, xml: str) -> None:
        relocated_xml = relocate_model_asset_paths(
            xml,
            robosuite_root=self._robosuite_root,
            libero_assets_root=self._libero_assets_root,
        )
        target = (
            self.env
            if callable(getattr(self.env, "reset_from_xml_string", None))
            else self.domain
        )
        target.reset_from_xml_string(relocated_xml)
        self.sim.reset()

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        regeneration_target = next(
            (
                target
                for target in (self.env, self.domain)
                if callable(getattr(target, "regenerate_obs_from_state", None))
            ),
            None,
        )
        if regeneration_target is not None:
            self._observation = regeneration_target.regenerate_obs_from_state(state)
        else:
            self.sim.set_state_from_flattened(state)
            self.sim.forward()
            self._observation = self.domain._get_observations()

    def get_state_flattened(self) -> np.ndarray:
        return np.asarray(self.sim.get_state().flatten(), dtype=np.float64)

    def replay_validation_vector(self, state: np.ndarray) -> np.ndarray:
        """Select positions from a legacy MjSimState flattened vector.

        Privileged labels are generated from teacher-forced recorded states.
        One-step replay fidelity is therefore gated on configuration (qpos),
        while velocity mismatch remains outside the scientific label path.
        """
        values = np.asarray(state, dtype=np.float64)
        nq = int(self.sim.model.nq)
        if values.ndim != 1 or len(values) < 1 + nq:
            raise ValueError("flattened LIBERO state is too short for qpos validation")
        return values[1 : 1 + nq].copy()

    def step(self, action: np.ndarray) -> None:
        self._observation, _, _, _ = self.env.step(action)

    def observation(self) -> dict[str, object]:
        result = dict(self._observation)
        result["_privileged_frame"] = self._privileged_frame()
        return result

    def contacts(self) -> tuple[tuple[str, str], ...]:
        model = self.sim.model
        data = self.sim.data
        result: list[tuple[str, str]] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            left = _model_id2name(model, "geom", int(contact.geom1))
            right = _model_id2name(model, "geom", int(contact.geom2))
            if left is not None and right is not None:
                result.append((str(left), str(right)))
        return tuple(result)

    def _entity_object(self, name: str):
        for attribute in ("objects_dict", "fixtures_dict"):
            mapping = getattr(self.domain, attribute, {})
            if name in mapping:
                return mapping[name]
        return None

    def _pose(self, name: str) -> np.ndarray:
        model = self.sim.model
        data = self.sim.data
        entity = self._entity_object(name)
        if entity is not None:
            body_name = str(getattr(entity, "root_body", getattr(entity, "name", name)))
            body_id = _model_name2id(model, "body", body_name)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = np.asarray(data.body_xpos[body_id])
            pose[:3, :3] = np.asarray(data.body_xmat[body_id]).reshape(3, 3)
            return pose
        sites = getattr(self.domain, "object_sites_dict", {})
        site = sites.get(name)
        site_name = str(getattr(site, "name", name))
        try:
            site_id = _model_name2id(model, "site", site_name)
        except Exception as error:
            raise ValueError(f"cannot resolve LIBERO semantic entity pose: {name}") from error
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(data.site_xpos[site_id])
        pose[:3, :3] = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
        return pose

    def _gripper_pose(self) -> np.ndarray:
        robot = self.robots[0]
        site_id = int(getattr(robot, "eef_site_id"))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(self.sim.data.site_xpos[site_id])
        pose[:3, :3] = np.asarray(self.sim.data.site_xmat[site_id]).reshape(3, 3)
        return pose

    def _entity_geoms(self, name: str) -> frozenset[str]:
        entity = self._entity_object(name)
        if entity is not None:
            values = frozenset(str(item) for item in getattr(entity, "contact_geoms", ()))
            if values:
                return values
        normalized = name.lower().replace("kitchen_", "").replace("_", "")
        model = self.sim.model
        matches: list[str] = []
        for index in range(int(model.ngeom)):
            geom_name = _model_id2name(model, "geom", index)
            if geom_name is None:
                continue
            candidate = str(geom_name).lower().replace("_", "")
            if normalized and normalized in candidate:
                matches.append(str(geom_name))
        return frozenset(matches)

    def _finger_groups(self) -> dict[str, frozenset[str]]:
        gripper = self.robots[0].gripper
        important = getattr(gripper, "important_geoms", {})
        if not isinstance(important, Mapping):
            important = {}
        left = getattr(gripper, "left_finger_geoms", ()) or important.get(
            "left_finger", important.get("left_fingerpad", ())
        )
        right = getattr(gripper, "right_finger_geoms", ()) or important.get(
            "right_finger", important.get("right_fingerpad", ())
        )
        result = {
            "left": frozenset(str(item) for item in left),
            "right": frozenset(str(item) for item in right),
        }
        if not result["left"] or not result["right"]:
            raise ValueError("cannot resolve both LIBERO gripper finger geom groups")
        return result

    def _surface_distance(self, left: Iterable[str], right: Iterable[str]) -> float:
        model = self.sim.model
        data = self.sim.data
        distances: list[float] = []
        for left_name in left:
            for right_name in right:
                try:
                    left_id = _model_name2id(model, "geom", left_name)
                    right_id = _model_name2id(model, "geom", right_name)
                except Exception:
                    continue
                try:
                    import mujoco

                    distance = float(
                        mujoco.mj_geomDistance(
                            model, data, left_id, right_id, 1.0, None
                        )
                    )
                    if np.isfinite(distance):
                        distances.append(max(distance, 0.0))
                        continue
                except (ImportError, TypeError):
                    pass
                center = float(np.linalg.norm(data.geom_xpos[left_id] - data.geom_xpos[right_id]))
                radii = float(model.geom_rbound[left_id] + model.geom_rbound[right_id])
                distances.append(max(center - radii, 0.0))
        return min(distances) if distances else float("inf")

    def _normalized_aperture(self) -> float:
        observation = self._observation
        qpos = observation.get("robot0_gripper_qpos")
        if qpos is None:
            raise ValueError("LIBERO observation has no robot0_gripper_qpos")
        values = np.abs(np.asarray(qpos, dtype=np.float64))
        gripper = self.robots[0].gripper
        maxima: list[float] = []
        for joint in getattr(gripper, "joints", ()):
            joint_id = _model_name2id(self.sim.model, "joint", str(joint))
            maxima.append(float(np.max(np.abs(self.sim.model.jnt_range[joint_id]))))
        denominator = float(np.sum(maxima)) if maxima else 0.0
        if denominator <= 0:
            raise ValueError("LIBERO gripper joints have no finite aperture range")
        return float(np.clip(values.sum() / denominator, 0.0, 1.0))

    def _privileged_frame(self) -> PrivilegedFrame:
        semantics: TaskSemantics = self.semantics
        target_geoms = self._entity_geoms(semantics.target)
        goal_geoms = self._entity_geoms(str(semantics.goal))
        source_geoms = self._entity_geoms(str(semantics.source))
        fingers = self._finger_groups()
        contact_labels = contact_labels_from_pairs(
            self.contacts(),
            SemanticContactMap(
                target_geoms=target_geoms,
                goal_geoms=goal_geoms,
                source_geoms=source_geoms,
                finger_geoms=fingers,
            ),
        )
        all_fingers = frozenset(geom for geoms in fingers.values() for geom in geoms)
        gripper_pose = self._gripper_pose()
        target_pose = self._pose(semantics.target)
        goal_pose = self._pose(str(semantics.goal))
        goal_satisfied = _evaluate_raw_goal_state(self.domain, self._raw_goal_states)
        return PrivilegedFrame(
            frame_index=-1,
            gripper_pose=gripper_pose,
            target_pose=target_pose,
            goal_pose=goal_pose,
            gripper_target_surface_distance=self._surface_distance(all_fingers, target_geoms),
            target_goal_surface_distance=(
                self._surface_distance(target_geoms, goal_geoms)
                if goal_geoms
                else float(np.linalg.norm(target_pose[:3, 3] - goal_pose[:3, 3]))
            ),
            finger_contact_groups=contact_labels.finger_groups,
            target_goal_contact=contact_labels.target_goal,
            target_source_contact=contact_labels.target_source,
            gripper_aperture=self._normalized_aperture(),
            source_supported=contact_labels.target_source,
            goal_satisfied=goal_satisfied,
        )
