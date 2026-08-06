# Franka Panda Assets

`franka_emika_panda/` is copied from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.

Only that model directory was imported. Its upstream `LICENSE`, `README.md`,
MJCF files, and meshes are preserved. The model is distributed under the
Apache License 2.0 stated in `franka_emika_panda/LICENSE`.

`franka_emika_panda/franka_tabletop.xml` is project-owned integration code. It
adds the interaction experiment's table, five object slots, receptacle, and
Agent/Wrist/Side/Top cameras. It lives beside the official XML so MuJoCo resolves
the upstream relative mesh directory correctly.

`panda_integration.xml` is an integration copy of the pinned upstream
`panda.xml`. Its only change is removal of the nine-value `home` keyframe,
because the five added free joints increase scene `qpos` from 9 to 44. The
original `panda.xml` remains preserved unchanged.
