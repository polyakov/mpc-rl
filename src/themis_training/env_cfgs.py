"""THEMIS velocity environment configurations."""
from __future__ import annotations
import math
from typing import TYPE_CHECKING
import torch
from themis_training.themis.themis_constants import (
  get_themis_robot_cfg,
  JOINT_NAMES_EXPR,
)
from themis_training import phase_mdp
from themis_training import mpc_grf_mdp
from themis_training.mpc_grf_mdp import LocoMPCCommandCfg, LocoManipMPCCommandCfg
from themis_training import push_box_mdp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensor,
  ContactSensorCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import SceneEntityCfg, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.envs.mdp import dr
from mjlab import terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
def _feet_too_near(
  env: ManagerBasedRlEnv,
  min_distance: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalise feet being closer than *min_distance* (XY plane).

  Returns ``max(0, min_distance - d_xy)`` per environment, where *d_xy* is
  the horizontal Euclidean distance between the two foot sites.  The penalty
  is zero when the feet are at least *min_distance* apart and grows linearly
  as they approach each other.
  """
  asset = env.scene[asset_cfg.name]
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  # Horizontal (XY) distance only – height difference is irrelevant.
  diff_xy = foot_pos[:, 0, :2] - foot_pos[:, 1, :2]  # [B, 2]
  dist_xy = torch.norm(diff_xy, dim=-1)  # [B]
  return torch.clamp(min_distance - dist_xy, min=0.0)
def _gait_symmetry(
  env: ManagerBasedRlEnv,
  left_sensor_name: str,
  right_sensor_name: str,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward alternating biped gait (L/R feet anti-phase).

  Returns 1 when exactly one foot is in the air (swing-stance alternation)
  and 0 when both feet are simultaneously grounded or simultaneously airborne.
  The reward is gated on a non-zero velocity command so the robot is not
  penalised or rewarded for standing still.
  """
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]
  left_air_time = left_sensor.data.current_air_time   # [B, 1]
  right_air_time = right_sensor.data.current_air_time  # [B, 1]
  assert left_air_time is not None and right_air_time is not None
  left_in_air = left_air_time[:, 0] > 0
  right_in_air = right_air_time[:, 0] > 0
  # XOR: reward only when exactly one foot is swinging.
  alternating = (left_in_air ^ right_in_air).float()
  # Gate reward by command magnitude so we don't reward symmetry at rest.
  command = env.command_manager.get_command(command_name)
  if command is not None:
    cmd_mag = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    alternating *= (cmd_mag > command_threshold).float()
  return alternating
def themis_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create THEMIS rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # Set raycast sensor frame to THEMIS base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "BASE_LINK"

  cfg.scene.entities = {"robot": get_themis_robot_cfg()}

  # Foot sites for height / clearance tracking.
  site_names = ("left_foot", "right_foot")

  # Foot geoms for ground contact sensor (mesh geoms named in XML).
  geom_names = ("left_foot_collision", "right_foot_collision")

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # --------------- actions ---------------
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=tuple(JOINT_NAMES_EXPR),
    scale=0.5,
    use_default_offset=True,
    preserve_order=True,
  )

  # --------------- observations ---------------
  cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=tuple(JOINT_NAMES_EXPR), preserve_order=True)},
  )
  cfg.observations["critic"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=tuple(JOINT_NAMES_EXPR), preserve_order=True)},
  )
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=tuple(JOINT_NAMES_EXPR), preserve_order=True)},
  )
  cfg.observations["critic"].terms["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=tuple(JOINT_NAMES_EXPR), preserve_order=True)},
  )

  # --------------- viewer ---------------
  cfg.viewer.body_name = "BASE_LINK"

  # --------------- commands ---------------
  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.45  # above the head (~1.18 m standing + margin)

  # --------------- reward references ---------------
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("BASE_LINK",)

  # Domain randomization: ±20% PD gains 
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )

  # Domain randomization: joint armature (scale default armature by 0.5–1.5).
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "ranges": (0.5, 1.5),
      "operation": "scale",
    },
  )

  # Domain randomization: ±20% body inertia via pseudo-inertia parameterization.
  # alpha scales mass & inertia by e^(2α): α=-0.112 → 0.8×, α=0.091 → 1.2×.
  # body_names=(".*",) covers all bodies including BASE_LINK.
  cfg.events["body_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.112, 0.091),
    },
  )

  # Pose reward: match default joint angles per-joint-group.
  # Tighter std keeps joints near nominal; looser allows natural motion.
  cfg.rewards["pose"].params["asset_cfg"].joint_names = [
    r".*HIP_YAW.*",
    r".*HIP_ROLL.*",
    r".*HIP_PITCH.*",
    r".*KNEE.*",
    r".*ANKLE.*",
    r".*SHOULDER.*",
    r".*ELBOW.*",
    r".*WRIST.*",
    r".*HEAD.*",
  ]
  cfg.rewards["pose"].params["std_standing"] = {
    # Lower body
    r".*HIP_PITCH.*":  0.025,
    r".*HIP_ROLL.*":   0.15,
    r".*HIP_YAW.*":    0.15,
    r".*KNEE.*":       0.05,
    r".*ANKLE.*":      0.05,
    # Upper body
    r".*SHOULDER.*":   0.05,
    r".*ELBOW.*":      0.05,
    r".*WRIST.*":      0.1,
    r".*HEAD.*":       0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body
    r".*HIP_PITCH.*":  0.1,
    r".*HIP_ROLL.*":   0.15,
    r".*HIP_YAW.*":    0.15,
    r".*KNEE.*":       0.35,
    r".*ANKLE.*":      0.25,
    # Upper body – arms swing during walking
    r".*SHOULDER.*":   0.2,
    r".*ELBOW.*":      0.2,
    r".*WRIST.*":      0.3,
    r".*HEAD.*":       0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body
    r".*HIP_PITCH.*":  0.25,
    r".*HIP_ROLL.*":   0.25,
    r".*HIP_YAW.*":    0.25,
    r".*KNEE.*":       0.6,
    r".*ANKLE.*":      0.35,
    # Upper body
    r".*SHOULDER.*":   0.35,
    r".*ELBOW.*":      0.35,
    r".*WRIST.*":      0.4,
    r".*HEAD.*":       0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("BASE_LINK",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("BASE_LINK",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  # cfg.rewards["angular_momentum"].weight = -0.0
  cfg.rewards["angular_momentum"].weight = -0.02 #default


  # --- Stepping frequency / periodicity tuning ---
  # air_time: rewards each foot being airborne between threshold_min and
  # threshold_max seconds.  Enabling this (weight > 0) is the primary knob
  # for periodic gait.  Raising threshold_min forces a longer minimum swing
  # phase per step, which directly lowers stepping frequency:
  #   threshold_min ~0.10 s → fast gait (~4–5 Hz)
  #   threshold_min ~0.20 s → moderate gait (~3 Hz)
  #   threshold_min ~0.30 s → slow, deliberate gait (~1.5–2 Hz)  ← target
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.30   # min swing per step 
  cfg.rewards["air_time"].params["threshold_max"] = 0.70   # max swing per step 

  # action_rate_l2: penalises ||a_t - a_{t-1}||².  A larger negative weight
  # suppresses rapid oscillatory actions between timesteps, further smoothing
  # the motion and discouraging high-frequency leg flapping.
  cfg.rewards["action_rate_l2"].weight = -0.25

  # Penalize self-collision between the two hip abad links (prevent them
  # from rubbing against each other to stabilise walking).
  cfg.rewards.pop("self_collisions", None)
  hip_abad_collision_cfg = ContactSensorCfg(
    name="hip_abad_collision",
    primary=ContactMatch(mode="body", pattern="HIP_ABAD_L", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="HIP_ABAD_R", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["hip_abad_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": hip_abad_collision_cfg.name, "force_threshold": 10.0},
  )

  # Penalize feet contact: prevent feet from touching each other during gait.
  feet_feet_collision_cfg = ContactSensorCfg(
    name="feet_feet_collision",
    primary=ContactMatch(mode="geom", pattern="FOOT_L_.*", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="FOOT_R_.*", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["feet_feet_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": feet_feet_collision_cfg.name, "force_threshold": 5.0},
  )

  # Penalize right foot hitting left tibia (cross-leg kick during swing).
  right_foot_left_tibia_cfg = ContactSensorCfg(
    name="right_foot_left_tibia_collision",
    primary=ContactMatch(mode="geom", pattern="FOOT_R_.*", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="left_tibia_collision", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["right_foot_left_tibia_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": right_foot_left_tibia_cfg.name, "force_threshold": 5.0},
  )

  # Penalize left foot hitting right tibia (cross-leg kick during swing).
  left_foot_right_tibia_cfg = ContactSensorCfg(
    name="left_foot_right_tibia_collision",
    primary=ContactMatch(mode="geom", pattern="FOOT_L_.*", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="right_tibia_collision", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["left_foot_right_tibia_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": left_foot_right_tibia_cfg.name, "force_threshold": 5.0},
  )

  # Penalize left femur hitting left upperwrist (hand slapping own thigh).
  left_femur_wrist_cfg = ContactSensorCfg(
    name="left_femur_wrist_collision",
    primary=ContactMatch(mode="body", pattern="FEMUR_L", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="UPPERWRIST_L", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["left_femur_wrist_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": left_femur_wrist_cfg.name, "force_threshold": 5.0},
  )

  # Penalize right femur hitting right upperwrist (hand slapping own thigh).
  right_femur_wrist_cfg = ContactSensorCfg(
    name="right_femur_wrist_collision",
    primary=ContactMatch(mode="body", pattern="FEMUR_R", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="UPPERWRIST_R", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.rewards["right_femur_wrist_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": right_femur_wrist_cfg.name, "force_threshold": 5.0},
  )

  # Penalize feet being too close together (minimum lateral spacing).
  cfg.rewards["feet_too_near"] = RewardTermCfg(
    func=_feet_too_near,
    weight=-0.50,
    params={
      "min_distance": 0.15,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # --------------- contact sensors ---------------
  # Non-foot body parts that should NOT touch the ground (illegal contact).
  non_contact_geom_names = (
    "trunk_collision",
    "left_tibia_collision",
    "right_tibia_collision",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=non_contact_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Per-foot sensors for gait-symmetry reward (L/R tracked independently).
  left_foot_ground_cfg = ContactSensorCfg(
    name="left_foot_ground",
    primary=ContactMatch(mode="geom", pattern="left_foot_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  right_foot_ground_cfg = ContactSensorCfg(
    name="right_foot_ground",
    primary=ContactMatch(mode="geom", pattern="right_foot_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    hip_abad_collision_cfg,
    feet_feet_collision_cfg,
    right_foot_left_tibia_cfg,
    left_foot_right_tibia_cfg,
    left_femur_wrist_cfg,
    right_femur_wrist_cfg,
    left_foot_ground_cfg,
    right_foot_ground_cfg,
  )

  # Symmetric periodic gait: reward L/R foot alternation (anti-phase stride).
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=_gait_symmetry,
    weight=0.00,
    params={
      "left_sensor_name": left_foot_ground_cfg.name,
      "right_sensor_name": right_foot_ground_cfg.name,
    },
  )

  # --------------- terminations ---------------
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # --------------- play mode overrides ---------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
def themis_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create THEMIS flat terrain velocity configuration."""
  cfg = themis_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  del cfg.observations["actor"].terms["base_lin_vel"] 
  
  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  # cfg.curriculum["command_vel"] = CurriculumTermCfg(
  #   func=mdp.commands_vel,
  #   params={
  #     "command_name": "twist",
  #     "velocity_stages": [
  #       {"step": 0, "lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.25, 0.25)},
  #       {"step": 10000 * 24, "lin_vel_x": (-0.75, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.50, 0.50)},
  #       {"step": 20000 * 24, "lin_vel_x": (-1.0, 1.25), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.00, 1.00)},
  #     ],
  #   },
  # )


  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.75, 0.75), "ang_vel_z": (-1.0, 1.0)},
      ],
    },
  )


  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.75, 1.5)
    twist_cmd.ranges.lin_vel_y = (-0.75, 0.75)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.terminations.pop("root_height", None)
    cfg.terminations.pop("knee_height", None)
    cfg.terminations.pop("fell_over", None)

  # ── Velocity tracking-score metrics via the command term ────────────────
  # Swap the ``twist`` command for one that publishes additional exp-kernel
  # tracking scores (``tracking_lin_score`` / ``tracking_ang_score``) under
  # ``Metrics/twist/...`` alongside the inherited ``error_vel_xy`` /
  # ``error_vel_yaw`` L2 errors.  Same body-frame errors and ``std`` as the
  # ``track_linear_velocity`` / ``track_angular_velocity`` rewards, so the
  # metric is directly comparable across base-RL and MPC-guided runs (and
  # is reported regardless of whether the matching reward is enabled).
  old_twist = cfg.commands["twist"]
  assert isinstance(old_twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = mpc_grf_mdp.TrackingMetricsVelocityCommandCfg.from_uniform(
    old_twist,
    metric_std_lin=math.sqrt(0.25),
    metric_std_ang=math.sqrt(0.5),
  )

  return cfg
# Gait clock period shared by observation and reward (seconds).
_GAIT_PERIOD = 0.9
# Per-foot phase offsets as fractions of the period.
# [0.0, 0.5] → left foot at phase 0, right foot half a period later
# (standard alternating biped walk).
_FOOT_OFFSETS = [0.0, 0.5]
# sin threshold separating stance (sin < threshold) from swing (sin ≥ threshold).
# 0.5 → exactly 50 % duty cycle; slightly above 0.5 gives small stance bias.
_GAIT_THRESHOLD = 0.5
def _apply_phase_features(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add phase observation and phase-locked gait reward to an existing cfg."""

  site_names = ("left_foot", "right_foot")

  # ── Observations: add gait-clock signal to actor input ──────────────────
  # The 2-D (sin, cos) vector lets the policy read off the current phase of
  # the gait cycle.  The clock freezes (decays to 0 = both-feet-stance) when
  # the velocity command is near zero, signalling the policy to stand still.
  #
  # The actor uses the stateful `phase` class which advances the shared
  # env._gait_phase buffer.  The critic uses `phase_readout` which only
  # reads the buffer to avoid double-advancing per step.
  cfg.observations["actor"].terms["phase"] = ObservationTermCfg(
    func=phase_mdp.phase,
    params={
      "period": _GAIT_PERIOD,
      "command_name": "twist",
      "command_threshold": 0.1,
      "stance_decay": 0.95,
    },
  )
  cfg.observations["critic"].terms["phase"] = ObservationTermCfg(
    func=phase_mdp.phase_readout,
    params={
      "period": _GAIT_PERIOD,
      "command_name": "twist",
    },
  )

  # ── Rewards: replace air-time with phase-locked gait reward ─────────────
  # Disable the old air-time reward so the two don't conflict.
  cfg.rewards["air_time"].weight = 0.0

  # feet_gait: reward each foot for being in contact / in the air exactly when
  # the gait clock says it should be.  Uses the combined feet_ground_contact
  # sensor whose data.found tensor has shape [B, 2] — one column per foot
  # geom (left_foot_collision, right_foot_collision) in declaration order.
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=phase_mdp.feet_gait,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "period": _GAIT_PERIOD,
      "offset": _FOOT_OFFSETS,
      "threshold": _GAIT_THRESHOLD,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  # stand_still: penalise joint deviation from the default pose when the
  # commanded velocity is near zero (robot should stand neutrally).
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=phase_mdp.stand_still,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  # stand_still_vel: penalise joint velocities when the robot should be
  # standing still — directly suppresses shakiness / oscillation at rest.
  # cfg.rewards["stand_still_vel"] = RewardTermCfg(
  #   func=phase_mdp.stand_still_vel,
  #   weight=-0.5,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
  #     "command_name": "twist",
  #     "command_threshold": 0.1,
  #   },
  # )

  # foot_clearance: target swing height matches the unitree-rl-lab default.
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.15
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names

  return cfg
def themis_phase_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """THEMIS flat terrain config with phase-clock observation and gait reward.

  Builds on :func:`themis_flat_env_cfg` and adds:

  * A sinusoidal gait-clock ``phase`` observation (period = 0.6 s).
  * A ``foot_gait`` reward that directly scores each foot on following the
    clock-derived stance/swing schedule.
  * A ``stand_still`` penalty for joint drift during zero-velocity commands.
  * A ``foot_flat_orientation`` reward, a ``cop_forward`` reward (with
    heel/toe ground sensors), and tightened ANKLE pose std, matching the
    non-MPC reward structure of :func:`themis_mpc_grf_v2_flat_env_cfg` so
    the two configs can be compared on shared-reward growth.
  """
  cfg = themis_flat_env_cfg(play=play)
  cfg = _apply_phase_features(cfg)
  _apply_v2_shared_non_mpc_features(cfg)
  return cfg
def _apply_v2_shared_non_mpc_features(
  cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
  """Shared non-MPC enrichments used by the v2 reward family (terrain-agnostic).

  Tightens the ankle pose std, raises ``foot_flat_orientation`` to weight 1,
  adds heel/toe ground contact sensors, and adds the ``cop_forward`` reward.
  Both :func:`themis_phase_flat_env_cfg` and
  :func:`themis_mpc_grf_v2_rough_env_cfg` apply this so the v2 non-MPC reward
  structure is shared across flat and rough terrains.
  """
  cfg.rewards["pose"].params["std_walking"][r".*ANKLE.*"] = 0.15
  cfg.rewards["pose"].params["std_running"][r".*ANKLE.*"] = 0.20

  cfg.rewards["foot_flat_orientation"] = RewardTermCfg(
    func=mpc_grf_mdp.foot_flat_orientation,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      "sigma": 0.15,
    },
  )

  heel_ground_cfg = ContactSensorCfg(
    name="heel_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "FOOT_L_heel_in", "FOOT_L_heel_out", "FOOT_L_heel_center",
        "FOOT_R_heel_in", "FOOT_R_heel_out", "FOOT_R_heel_center",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  toe_ground_cfg = ContactSensorCfg(
    name="toe_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "FOOT_L_toe_in", "FOOT_L_toe_out", "FOOT_L_toe_in2", "FOOT_L_toe_out2",
        "FOOT_R_toe_in", "FOOT_R_toe_out", "FOOT_R_toe_in2", "FOOT_R_toe_out2",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    heel_ground_cfg,
    toe_ground_cfg,
  )

  cfg.rewards["cop_forward"] = RewardTermCfg(
    func=mpc_grf_mdp.cop_forward_reward,
    weight=1.0,
    params={
      "heel_sensor_name": "heel_ground_contact",
      "toe_sensor_name": "toe_ground_contact",
      "target_ratio": 0.5,
      "sigma": 0.25,
    },
  )

  return cfg
def _apply_mpc_grf_features(
  cfg: ManagerBasedRlEnvCfg, play: bool,
) -> ManagerBasedRlEnvCfg:
  """Add the v1 MPC-GRF feature stack to ``cfg`` (terrain-agnostic).

  Adds the centroidal ``loco_mpc`` command term, the MPC-guided rewards
  (``mpc_grf_tracking`` at weight 0, ``mpc_ang_mom``, ``mpc_com_tracking``,
  ``foot_flat_orientation``) and the critic-only MPC reference observations.
  Used by both the flat (:func:`themis_mpc_grf_flat_env_cfg`) and rough
  (:func:`themis_mpc_grf_v2_rough_env_cfg`) variants.
  """
  # ── Locomotion MPC command term ─────────────────────────────────────────
  # Runs the centroidal MPC at every policy step and stores the first-step
  # optimal foot forces in the command buffer.  These are NOT fed to the
  # policy; they serve purely as a reference for the zero-weight reward below.
  cfg.commands["loco_mpc"] = LocoMPCCommandCfg(
    debug_vis=play,
    asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    mpc_dt=0.07,            # 50 × 0.07 s = 3.5 s ≈ 5 full gait cycle lookahead
    mpc_horizon=10,
    mass=37.0,
    hip_width=0.15,          # ±0.1 m body-frame y offset for predicted touchdowns
    gait_period=_GAIT_PERIOD,  # must match _GAIT_PERIOD used by the phase clock
    duty_factor=0.5,
    vel_cmd_name="twist",
    grf_sensor_name="feet_ground_contact",
    run_every_n_steps=5,    # 10 Hz  (policy = 50 Hz → every 5 steps)
  )

  # ── Zero-weight GRF tracking reward (verification only) ─────────────────
  # Compares actual foot contact forces to the MPC optimal reference.
  # Weight = 0 → no training signal; value is logged for inspection.
  cfg.rewards["mpc_grf_tracking"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_grf_tracking,
    weight=0.0,
    params={
      "command_name": "loco_mpc",
      "grf_sensor_name": "feet_ground_contact",
    },
  )

  # ── MPC-guided angular-momentum regulation reward ─────────────────────
  # Replace the generic angular_momentum penalty with the MPC-guided version
  # that penalises deviations from the centroidal MPC's predicted k trajectory.
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["mpc_ang_mom"] = RewardTermCfg(
    func=mpc_grf_mdp.MpcAngMomTracking,
    weight=0.05,
    params={
      "command_name": "loco_mpc",
      "w_k":    1.0,
      "w_kdot": 0.01,
      # qx, qy > qz: penalise roll/pitch ang-mom deviations more than yaw
      "q_k":    (1.0, 1.0, 0.5),
      "q_kdot": (0.1, 0.1, 0.05),
    },
  )

  # ── MPC CoM tracking reward ───────────────────────────────────────────
  # Exponential-kernel reward for tracking the MPC-predicted CoM position
  # and velocity.  Uses position + velocity error; peaks at 1 when perfect.
  cfg.rewards["mpc_com_tracking"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_com_tracking,
    weight=1.0,
    params={
      "command_name": "loco_mpc",
      "w_pos": 1.0,
      "w_vel": 0.5,
    },
  )

  # ── Foot flatness regularization ─────────────────────────────────────────
  # Reward the foot contact normal (local Z) staying aligned with world Z,
  # active during both swing and stance.  Discourages ankle twist and toe/
  # heel drop throughout the stride.
  cfg.rewards["foot_flat_orientation"] = RewardTermCfg(
    func=mpc_grf_mdp.foot_flat_orientation,
    weight=0.3,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
      "sigma": 0.15,
    },
  )

  # ── MPC reference observations (critic only) ───────────────────────────
  # The actor keeps its original observation space (deployable without MPC).
  # The critic sees MPC targets to better estimate value under the MPC prior.
  cfg.observations["critic"].terms["mpc_com_ref"] = ObservationTermCfg(
    func=mpc_grf_mdp.mpc_com_ref,
    params={"command_name": "loco_mpc"},
  )
  cfg.observations["critic"].terms["mpc_k_ref"] = ObservationTermCfg(
    func=mpc_grf_mdp.mpc_ang_mom_ref,
    params={"command_name": "loco_mpc"},
  )


  return cfg
def themis_mpc_grf_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """THEMIS phase-flat env with a centroidal locomotion MPC running in parallel.

  Inherits all of :func:`themis_phase_flat_env_cfg` (phase clock, gait reward,
  stand-still penalty, velocity curriculum) without modifying any reward
  weights.  Adds:

  * A :class:`~mpc_grf_mdp.LocoMPCCommand` term (``"loco_mpc"``) that runs
    the centroidal QP-MPC at every policy step (50 Hz, sparse relative to
    physics at 200 Hz).  The MPC is locomotion-only — no hand forces.
  * A ``mpc_grf_tracking`` reward (weight **0.0**) that computes and logs
    the RMS error between the MPC-optimal foot GRFs and the actual contact
    forces.  The zero weight means no training signal is added; the reward
    value can be monitored to verify the MPC pipeline is running correctly.

  Policy training is therefore identical to ``Mjlab-Velocity-Phase-Flat-Themis``.

  MPC configuration
  ─────────────────
  * Horizon dt = 0.02 s (matches policy step_dt = 50 Hz)
  * N = 10 horizon steps → 0.2 s lookahead
  * Contact schedule derived from ``env._gait_phase``
  * Reference: linear CoM trajectory at commanded velocity, zero angular
    momentum
  * Warm-started from previous solution; reset on episode termination
  """
  return _apply_mpc_grf_features(themis_phase_flat_env_cfg(play=play), play=play)
def themis_mpc_grf_v2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """THEMIS flat env: MPC-primary reward structure with phase-aligned contacts.

  Builds on :func:`themis_mpc_grf_flat_env_cfg` with a cleaner separation of
  concerns:

  * **Velocity tracking** is removed as an explicit reward; it is implicit in
    ``mpc_com_tracking`` via the MPC's linear-momentum reference
    ``l_ref = m·v_cmd``.  The CoM-tracking weight is raised and ``w_vel``
    increased so the policy receives a clear signal for both position and
    velocity.

  * **Yaw-rate tracking** is removed; it is captured by ``mpc_ang_mom`` via
    the MPC's angular-momentum reference ``k_ref_z = I_zz · ω_z_cmd``.  The
    yaw weight in ``q_k`` is raised to equal the roll/pitch weights.

  * **body_ang_vel** is removed; it is subsumed by ``mpc_ang_mom``, which
    penalises angular-momentum rate deviations from the MPC reference.

  * **air_time** is deleted (was already zeroed by the phase env).

  Reward changes vs. v1
  ─────────────────────
  Removed:
    * ``track_linear_velocity``   (→ ``mpc_com_tracking``)
    * ``track_angular_velocity``  (→ ``mpc_ang_mom``)
    * ``body_ang_vel``            (→ ``mpc_ang_mom`` k_dot regularisation)
    * ``air_time``                (was 0; deleted for clarity)

  Kept as gait regularization:
    * ``foot_gait``         +1.0   phase-locked stance/swing (phase consistency)
    * ``foot_clearance``    -4.0   swing foot must clear target height
    * ``foot_swing_height`` -0.25  penalise insufficient swing height
    * ``foot_slip``         -0.1   penalise sliding during stance
    * ``soft_landing``      -1e-5  penalise large touchdown impact forces

  Kept as regularization:
    * ``pose``, ``action_rate_l2``, ``dof_pos_limits``, ``stand_still``
    * all self-collision terms, ``feet_too_near``

  MPC rewards (updated weights):
    * ``mpc_com_tracking``  +1.5  (was 0.25) — now primary locomotion reward
    * ``mpc_ang_mom``       +0.05 — unchanged weight; yaw q_k raised to 1.0
    * ``mpc_grf_tracking``  +0.5  — enabled (was 0.0 logging-only in v1)
  """
  return _apply_mpc_grf_v2_features(themis_mpc_grf_flat_env_cfg(play=play))
def _apply_mpc_grf_v2_features(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Apply the v2 reward overrides on top of a v1 MPC-GRF cfg (terrain-agnostic).

  Removes the base-RL velocity-tracking / body-ang-vel / air-time rewards
  (now subsumed by the MPC tracking terms), promotes ``mpc_com_tracking``
  to the primary locomotion driver, adds ``mpc_com_vel_tracking`` /
  ``mpc_ang_vel_tracking`` (multi-lookahead), raises foot-flatness, adds
  the (disabled) ``mpc_foot_placement`` term, and bumps the MPC solve
  cadence to every policy step.  Used by both flat and rough v2 configs.
  """
  # ── Remove rewards now carried by MPC tracking ───────────────────────────
  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards.pop("body_ang_vel", None)
  cfg.rewards.pop("air_time", None)
  cfg.rewards.pop("foot_slip", None)

  # ── Promote MPC CoM tracking: now the primary locomotion driving reward ───
  # w_vel increased so velocity error dominates over position error; the MPC
  # reference l_ref = m·v_cmd encodes the commanded velocity directly.
  # Multi-point horizon sampling (Option A): the reward averages squared
  # errors at 5 horizon fractions spanning [0.0, 1.0], giving the policy a
  # trajectory-shaped signal rather than tracking a single instant.  The
  # far-horizon samples carry lower weight so the near-term (tractable)
  # target still dominates the gradient.
  # ── Split CoM tracking into pos-only and vel-only terms ──────────────────
  cfg.rewards["mpc_com_tracking"].weight = 2.0
  cfg.rewards["mpc_com_tracking"].params["w_pos"] = 1.0
  cfg.rewards["mpc_com_tracking"].params["w_vel"] = 0.0   # vel handled below
  cfg.rewards["mpc_com_tracking"].params["lookahead_fracs"] = (0.0, 0.25, 0.5, 0.75, 1.0)
  cfg.rewards["mpc_com_tracking"].params["lookahead_weights"] = (0.5, 0.25, 0.15, 0.07, 0.03)
  # cfg.rewards["mpc_com_tracking"].params["lookahead_weights"] = (0.2, 0.2, 0.2, 0.2, 0.2)

  cfg.rewards["mpc_com_vel_tracking"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_com_vel_tracking,
    weight=2.0,                                            # equal to pos term
    params={
      "command_name":      "loco_mpc",
      # w_vel = 1/sigma^2 with sigma = 0.5 m/s — matches the base-RL
      # ``track_linear_velocity`` kernel ``exp(-||e||^2 / 0.25)`` at every
      # horizon landmark, so per-landmark velocity errors are penalised on
      # the same scale as the base-RL command-tracking reward.
      "w_vel":             4.0,
      "lookahead_fracs":   (0.0, 0.25, 0.5, 0.75, 1.0),
      "lookahead_weights": (0.5, 0.25, 0.15, 0.07, 0.03),
      # "lookahead_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
    },
  )

  # ── MPC angular-velocity tracking (replaces base-RL track_angular_velocity)
  # The centroidal MPC carries angular momentum k, not ω directly; we
  # extract ω^mpc = I^{-1} k^mpc using the same trunk-inertia approximation
  # I_approx that MpcAngMomTracking already uses.  Multi-landmark profile
  # mirrors mpc_com_vel_tracking so the prediction-landmark annealing is
  # consistent across linear / angular MPC supervision.
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards["mpc_ang_vel_tracking"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_ang_vel_tracking,
    weight=4.0,                                            # equal to lin-vel term
    params={
      "command_name":      "loco_mpc",
      # w_ang = 1/sigma^2 with sigma = sqrt(0.5) rad/s — matches the
      # base-RL ``track_angular_velocity`` kernel ``exp(-||e||^2 / 0.5)``
      # at every horizon landmark.
      "w_ang":             2.0,
      "lookahead_fracs":   (0.0, 0.25, 0.5, 0.75, 1.0),
      "lookahead_weights": (0.5, 0.25, 0.15, 0.07, 0.03),
      # "lookahead_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
    },
  )

  # ── MPC angular-momentum tracking: raise yaw weight to absorb ang vel ───
  # q_k[2] (yaw) raised to equal roll/pitch so yaw-rate commands produce a
  # training signal comparable to the removed track_angular_velocity reward.
  # Same multi-point horizon sampling as mpc_com_tracking.
  cfg.rewards["mpc_ang_mom"].weight = 0.05
  cfg.rewards["mpc_ang_mom"].params["q_k"] = (1.0, 1.0, 1.0)
  cfg.rewards["mpc_ang_mom"].params["lookahead_fracs"] = (0.0, 0.25, 0.5, 0.75, 1.0)
  cfg.rewards["mpc_ang_mom"].params["lookahead_weights"] = (0.5, 0.25, 0.15, 0.07, 0.03)
  # cfg.rewards["mpc_ang_mom"].params["lookahead_weights"] = (0.2, 0.2, 0.2, 0.2, 0.2)

  # ── GRF tracking: enable training signal (was logging-only in v1) ─────────
  cfg.rewards["mpc_grf_tracking"].weight = 0.002

  # ── Foot flatness regularization ─────────────────────────────────────────
  cfg.rewards["foot_flat_orientation"].weight = 1.0

  # ── MPC foot placement tracking ──────────────────────────────────────
  # Reward swing feet for approaching the MPC's Raibert-heuristic predicted
  # landing positions (XY).  Provides a direct spatial gradient for WHERE
  # to step, complementing mpc_com_tracking which only signals CoM error.
  cfg.rewards["mpc_foot_placement"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_foot_placement_tracking,
    weight=1.0,
    params={
      "command_name": "loco_mpc",
      "sigma": 0.2,
    },
  )

  # ── CoP forward reward & ankle-pose tightening are inherited from the
  # phase env so Phase and V2 share identical non-MPC reward structure.

  # ── MPC solve cadence: 10 Hz → 50 Hz ─────────────────────────────────
  # The v1 base sets ``run_every_n_steps=5`` (10 Hz at the 50 Hz policy
  # rate).  Run at every policy step so the value targets the policy reads
  # via ``mpc_com_ref`` / ``mpc_k_ref`` are always one-step-fresh and the
  # warm-start is one policy step old (better ADMM convergence).
  # Horizon stays at the v1 default (10 × 0.07 s = 0.7 s lookahead).
  loco_mpc_cmd = cfg.commands["loco_mpc"]
  assert isinstance(loco_mpc_cmd, LocoMPCCommandCfg)
  loco_mpc_cmd.mpc_horizon       = 10
  loco_mpc_cmd.run_every_n_steps = 5
  # ── Solver: AOT-compiled preconditioned JAX PiMPC ────────────────────────
  # V2 trains with the JAX path by default — typically 5-10× faster than the
  # PyTorch PiMPC at the same accuracy thanks to cost equilibration + AOT
  # compile + Nesterov.  Requires the matching JAX install in the env (see
  # ``jax[cuda12]==0.10.0`` in the project README / .venv).
  loco_mpc_cmd.solver_type       = "jax_pimpc"

  return cfg
# Commanded twist is restricted to forward-only (no lateral / backward / yaw)
# in push mode.  vx ∈ [0, _PUSH_BOX_VX_MAX]; ModeGatedVelocityCommand zeros
# lin_vel_y and ang_vel_z for every push env.
_PUSH_BOX_VX_MAX: float = 0.75
def _themis_mpc_push_box_base(play: bool = False) -> ManagerBasedRlEnvCfg:
  """[Private helper] THEMIS push-box scene on top of the MPC-GRF v2 base.

  Builds the shared push-box scaffold — box entity, mode-gated forward-only
  twist command, hand/leg contact sensors, privileged critic obs, push
  rewards, terminations — used by the loco-manipulation envs.  Not a
  registered task on its own; consumers are
  :func:`themis_loco_manip_mpc_push_box_flat_env_cfg` and
  :func:`themis_loco_manip_no_mpc_push_box_flat_env_cfg`.

  Inherited from the v2 base:
    * MPC-primary reward structure (``mpc_com_tracking``, ``mpc_ang_mom``,
      ``mpc_foot_placement``, ``mpc_grf_tracking``) and MPC reference critic
      observations (``mpc_com_ref``, ``mpc_k_ref``).
    * Phase clock, ``foot_gait``, ``stand_still`` regularisation,
      ``foot_flat_orientation``, CoP forward reward, etc.
    * ``track_linear_velocity`` / ``track_angular_velocity`` residuals.

  Added on top (mirrors the perceptive push-box env):
    * Fixed 1 m × 1 m × 1.3 m box spawned in front of the robot.
    * ``ModeGatedVelocityCommand`` — forward-only twist (no lateral /
      backward / yaw).  All envs are push-mode.
    * Hand-box and leg-box contact sensors.
    * Push rewards: ``hand_box_contact``, ``push_velocity_match``,
      ``robot_box_velocity_match``, ``robot_box_xy_distance``,
      ``robot_box_yaw``, ``leg_box_collision``, ``hand_slip``.
    * Push terminations: ``robot_far_from_box``, ``box_toppled``,
      ``box_pitched``.
    * Privileged critic observations: ``box_pose_rel``, ``box_lin_vel``,
      ``box_size``, ``hand_box_contact`` (15 dims total — matches the
      perceptive env so both fine-tune runs exercise the same critic
      privileged signal).  The critic's first linear layer will
      re-initialise during :func:`themis_training.finetune` (shape filter);
      the actor input is unchanged from the MPC-v2 actor so it loads fully.
  """
  cfg = themis_mpc_grf_v2_flat_env_cfg(play=play)

  # ── Add the box entity alongside the robot ────────────────────────────
  # Loco-manipulation tasks initialise from the loco-manip default pose
  # (arms lowered/extended for pushing); locomotion uses the walking pose.
  cfg.scene.entities = {
    "robot": get_themis_robot_cfg(loco_manip=True),
    "box": EntityCfg(spec_fn=push_box_mdp.get_push_box_spec),
  }

  # ── Mode-gated twist command (forward-only, every env) ────────────────
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.__class__ = push_box_mdp.ModeGatedVelocityCommandCfg
  twist_cmd.ranges.lin_vel_x = (0.0, _PUSH_BOX_VX_MAX)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.rel_standing_envs = 0.0

  # Pin command ranges via the curriculum (single stage, no ramp).  Only
  # lin_vel_x[1] matters; vy/wz are hard-zeroed by the mode-gated resample.
  assert cfg.curriculum is not None
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (0.0, _PUSH_BOX_VX_MAX), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
      ],
    },
  )

  # ── Tighten the "stand-still" deadzone so the policy only enters stance
  # when cmd is very close to zero (same tweak as the perceptive env).
  _PB_CMD_THRESH = 0.02
  if "phase" in cfg.observations["actor"].terms:
    cfg.observations["actor"].terms["phase"].params["command_threshold"] = _PB_CMD_THRESH
  if "foot_gait" in cfg.rewards:
    cfg.rewards["foot_gait"].params["command_threshold"] = _PB_CMD_THRESH
  if "stand_still" in cfg.rewards:
    cfg.rewards["stand_still"].params["command_threshold"] = _PB_CMD_THRESH

  # ── Hand-box and leg-box contact sensors ──────────────────────────────
  lhand_box_contact_cfg = ContactSensorCfg(
    name="lhand_box_contact",
    primary=ContactMatch(mode="geom", pattern="hand_contact_L", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )
  rhand_box_contact_cfg = ContactSensorCfg(
    name="rhand_box_contact",
    primary=ContactMatch(mode="geom", pattern="hand_contact_R", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )
  leg_box_contact_cfg = ContactSensorCfg(
    name="leg_box_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "left_tibia_collision",
        "right_tibia_collision",
        "left_foot_collision",
        "right_foot_collision",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    lhand_box_contact_cfg,
    rhand_box_contact_cfg,
    leg_box_contact_cfg,
  )

  # ── Privileged critic observations (NO chest point cloud) ─────────────
  # Same 15-dim privileged box state as the perceptive env so a fine-tune
  # run re-initialises the critic's first layer consistently.
  cfg.observations["critic"].terms["box_pose_rel"] = ObservationTermCfg(
    func=push_box_mdp.box_pose_rel_priv,
    params={"box_name": "box", "robot_name": "robot"},
  )
  cfg.observations["critic"].terms["box_lin_vel"] = ObservationTermCfg(
    func=push_box_mdp.box_lin_vel_priv,
    params={"box_name": "box", "robot_name": "robot"},
  )
  cfg.observations["critic"].terms["box_size"] = ObservationTermCfg(
    func=push_box_mdp.box_size_priv,
    params={"box_name": "box", "geom_name": "box_geom"},
  )
  cfg.observations["critic"].terms["hand_box_contact"] = ObservationTermCfg(
    func=push_box_mdp.hand_box_contact_priv,
    params={
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
    },
  )

  # ── Rewards (added to MPC-v2 base) ────────────────────────────────────
  # Increase robot CoM / velocity tracking weights.  ``track_linear_velocity``
  # / ``track_angular_velocity`` were present in the v1 base this helper was
  # originally written against, but the v2 helper pops them in favour of
  # ``mpc_com_vel_tracking`` / ``mpc_ang_vel_tracking``.  Guard so the
  # registration import doesn't crash on v2 — for new push-box training you
  # likely want to bump the MPC-driven equivalents instead.
  if "track_linear_velocity" in cfg.rewards:
    cfg.rewards["track_linear_velocity"].weight  = 3.0
  if "track_angular_velocity" in cfg.rewards:
    cfg.rewards["track_angular_velocity"].weight = 2.5
  cfg.rewards["mpc_com_tracking"].weight        = 3.0

  cfg.rewards["hand_box_contact"] = RewardTermCfg(
    func=push_box_mdp.hand_box_contact,
    weight=1.0,
    params={
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
      "both_hands_bonus": 0.5,
    },
  )
  # Box trajectory tracking (multi-step, mirrors mpc_com_tracking).
  cfg.rewards["box_com_tracking"] = RewardTermCfg(
    func=push_box_mdp.box_com_tracking,
    weight=4.0,
    params={
      "command_name": "loco_mpc",
      "box_name": "box",
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
      "w_pos": 2.0,
      "w_vel": 0.5,
      "lookahead_fracs": (0.0, 0.25, 0.5, 0.75, 1.0),
      "lookahead_weights": (0.35, 0.25, 0.20, 0.12, 0.08),
    },
  )
  cfg.rewards["push_velocity_match"] = RewardTermCfg(
    func=push_box_mdp.push_velocity_match,
    weight=5.0,
    params={
      "command_name": "twist",
      "box_name": "box",
      "robot_name": "robot",
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
      "sigma": 0.3,
    },
  )
  cfg.rewards["robot_box_velocity_match"] = RewardTermCfg(
    func=push_box_mdp.robot_box_velocity_match,
    weight=2.5,
    params={
      "box_name": "box",
      "robot_name": "robot",
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
      "sigma": 0.3,
    },
  )
  cfg.rewards["robot_box_xy_distance"] = RewardTermCfg(
    func=push_box_mdp.robot_box_xy_distance_cost,
    weight=-1.0,
    params={
      "box_name": "box",
      "robot_name": "robot",
      "target_distance": 0.5,
    },
  )
  cfg.rewards["robot_box_yaw"] = RewardTermCfg(
    func=push_box_mdp.robot_box_yaw_cost,
    weight=-0.5,
    params={
      "box_name": "box",
      "robot_name": "robot",
    },
  )
  cfg.rewards["leg_box_collision"] = RewardTermCfg(
    func=push_box_mdp.leg_box_collision_cost,
    weight=-1.0,
    params={"sensor_name": "leg_box_contact"},
  )
  cfg.rewards["hand_slip"] = RewardTermCfg(
    func=push_box_mdp.HandSlipPenalty,
    weight=-0.5,
    params={
      "lhand_sensor": "lhand_box_contact",
      "rhand_sensor": "rhand_box_contact",
      "box_name": "box",
      "robot_name": "robot",
      "reach_distance": 0.35,
      "camera_site_name": "chest_camera",
    },
  )

  # ── Events: mode flag, box pose + friction ────────────────────────────
  cfg.events["init_push_mode"] = EventTermCfg(
    mode="startup",
    func=push_box_mdp.init_push_mode,
    params={"push_fraction": 1.0},
  )
  cfg.events["reset_box"] = EventTermCfg(
    mode="reset",
    func=push_box_mdp.reset_box_pose_and_mass,
    params={
      "x_range": (0.9, 1.5),           # spawn in front of the robot
      "y_range": (0.0, 0.0),
      "yaw_range": (0.0, 0.0),
      "mass_range": (3.0, 15.0),
      "z_clearance": 0.03,             # centre z = half_h + clearance = 0.50 + 0.03 = 0.53 m
      "box_name": "box",
      "geom_name": "box_geom",
    },
  )
  cfg.events["reset_box_friction"] = EventTermCfg(
    mode="reset",
    func=push_box_mdp.randomize_box_friction_curriculum,
    params={
      "default_friction_range": (0.5, 1.5),
      "box_name": "box",
      "geom_name": "box_geom",
    },
  )

  # ── Terminations ──────────────────────────────────────────────────────
  cfg.terminations["robot_far_from_box"] = TerminationTermCfg(
    func=push_box_mdp.robot_far_from_box,
    params={
      "max_distance": 2.0,
      "box_name": "box",
      "robot_name": "robot",
    },
  )
  cfg.terminations["box_toppled"] = TerminationTermCfg(
    func=push_box_mdp.box_toppled,
    params={
      "max_tilt_rad": 1.0,
      "box_name": "box",
    },
  )
  cfg.terminations["box_pitched"] = TerminationTermCfg(
    func=push_box_mdp.box_pitched,
    params={
      "max_pitch_rad": 0.1745,   # 10 degrees
      "box_name": "box",
    },
  )

  return cfg
def themis_loco_manip_mpc_push_box_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """THEMIS push-box env with the loco-manipulation MPC (hand forces).

  Builds on :func:`_themis_mpc_push_box_base` and replaces the
  locomotion-only ``LocoMPCCommand`` with :class:`LocoManipMPCCommand`,
  which adds two 3-D hand push-force decision variables to the centroidal QP.

  New additions over the MPC push-box base
  ─────────────────────────────────────────
  * **``body_box_contact`` sensor** — net-force contact sensor covering all
    robot-box contacts (via ``.*_collision`` geom pattern).  Its net force
    is used to compute the non-hand centroidal disturbance (other_body_force).
    Hand geoms are NOT included here; they are handled via the existing
    ``lhand_box_contact`` / ``rhand_box_contact`` sensors.
  * **``loco_mpc`` command** replaced by :class:`LocoManipMPCCommandCfg`
    which solves the 18-D loco-manip QP (feet + hands).

  Robot MJCF prerequisite
  ───────────────────────
  ``left_hand`` and ``right_hand`` sites must exist at pos ``(0.15, 0, 0)``
  inside the ``LOWERWRIST_L_contact`` / ``LOWERWRIST_R_contact`` bodies
  (already added to ``themis_29dof.xml``).
  """
  cfg = _themis_mpc_push_box_base(play=play)

  # ── All-body robot-box net-force sensor (non-hand geoms) ─────────────
  # Pattern covers legs / torso collision geoms; hand_contact_* are intentionally
  # excluded so that subtracting hand forces gives a clean other-body estimate.
  body_box_contact_cfg = ContactSensorCfg(
    name="body_box_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "left_tibia_collision",
        "right_tibia_collision",
        "left_foot_collision",
        "right_foot_collision",
        "left_upperarm_collision",
        "right_upperarm_collision",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (body_box_contact_cfg,)

  # ── Swap locomotion-only MPC command for loco-manipulation MPC ────────
  del cfg.commands["loco_mpc"]
  cfg.commands["loco_mpc"] = LocoManipMPCCommandCfg(
    debug_vis=play,
    mpc_dt=0.07,
    mpc_horizon=10,          # 10 × 0.07 s = 0.7 s lookahead (~1 gait cycle)
    mass=37.0,
    gait_period=_GAIT_PERIOD,
    run_every_n_steps=5,     # MPC runs at policy rate (50 Hz)
    # Hand geometry
    hand_site_names=("left_hand", "right_hand"),
    lhand_box_sensor_name="lhand_box_contact",
    rhand_box_sensor_name="rhand_box_contact",
    body_box_sensor_name="body_box_contact",
    # Box physics
    box_mass=8.0,
    mu_ground=0.5,
    # Hand MPC costs
    mu_hand=0.6,
    f_hand_max=300.0,
    R_f_hand=1e-4,
    R_hand_balance=1e-3,
    # Batched JAX PiMPC backend (per-env per-step dynamics, 18-D feet+hands QP)
    solver_type="jax_pimpc",
  )

  # ── 50/50 push/walk split ──────────────────────────────────────────────
  # Walk-only environments automatically park the box 100 m away (handled
  # by reset_box_pose_and_mass in push_box_mdp) and are shielded from
  # push-specific terminations and rewards by _push_mask().
  cfg.events["init_push_mode"].params["push_fraction"] = 0.5

  # ── Velocity commands: full 3-D for walk envs, vx-only for push envs ──
  # The base mpc_push_box env locks vy=0 and wz=0 globally.  Here we restore
  # full 3-D ranges so the 50% walk-only envs learn omnidirectional gait.
  # ModeGatedVelocityCommand still forces vy=0, wz=0 for push envs at
  # resample time, so push-mode behaviour is unchanged.
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.75)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.75, 0.75)
  # Update curriculum to match full locomotion range (push envs are overridden
  # by ModeGatedVelocityCommand and are unaffected by these ranges).
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.5, 0.75), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.75, 0.75)},
  ]

  cfg.events["reset_box"] = EventTermCfg(
    mode="reset",
    func=push_box_mdp.reset_box_pose_and_mass,
    params={
      "x_range": (0.9, 1.5),
      "y_range": (0.0, 0.0),
      "yaw_range": (0.0, 0.0),
      "mass_range": (3.0, 15.0),
      "z_clearance": 0.03,
      "box_name": "box",
      "geom_name": "box_geom",
    },
  )

  cfg.events["reset_box_friction"].params["default_friction_range"] = (0.05, 0.05)
  cfg.curriculum["push_box_friction"] = CurriculumTermCfg(
    func=push_box_mdp.push_box_mass_size_curriculum,
    params={
      "mass_size_stages": [
        {"step":    0, "friction_range": (0.05, 0.05)},
        {"step": 3000, "friction_range": (0.10, 0.10)},
        {"step": 5000, "friction_range": (0.25, 0.25)},
        {"step": 7000, "friction_range": (0.50, 0.50)},
      ],
    },
  )

  # ── Foot slip penalty: 10× stronger to prevent stance slide under push ─
  # Guarded: the v2 locomotion base may have removed ``foot_slip`` entirely.
  if "foot_slip" in cfg.rewards:
    cfg.rewards["foot_slip"].weight = -2.5

  # ── Arm default-pose penalty: tighten std to keep arms near nominal ───
  # The base env allows wide arm swing during walking (std ≈ 0.2–0.3 rad),
  # which causes bad arm posture during loco-manipulation.  Tightening the
  # arm stds to standing values and raising the overall pose weight strongly
  # penalises deviations so the robot keeps a compact, push-ready arm pose.
  pose_params = cfg.rewards["pose"].params
  for std_key in ("std_walking", "std_running"):
    if std_key in pose_params:
      pose_params[std_key][r".*SHOULDER.*"] = 0.05
      pose_params[std_key][r".*ELBOW.*"]    = 0.05
      pose_params[std_key][r".*WRIST.*"]    = 0.10
  cfg.rewards["pose"].weight = 5.0

  # ── Arm/neck deviation termination ────────────────────────────────────
  # Hard constraint: terminate when any arm or neck joint deviates more
  # than 1.2 rad (≈68°) from default.  The termination_penalty reward
  # (weight -20) provides an additional negative signal at episode end.
  # Applies to both push-mode and walk-only envs.
  cfg.terminations["arm_joint_deviation"] = TerminationTermCfg(
    func=push_box_mdp.arm_joint_deviation,
    params={
      "max_deviation_rad": 0.25,
      "robot_name": "robot",
    },
  )

  # ── Box-velocity stall termination (push-mode only) ───────────────────
  # Terminate when the box speed stays below 50% of the commanded velocity
  # for more than 5 s.  Prevents the policy from idling against an unmovable
  # box.  Walk-only envs are excluded via _push_mask().
  cfg.terminations["box_velocity_stall"] = TerminationTermCfg(
    func=push_box_mdp.box_velocity_stall,
    params={
      "min_speed_fraction": 0.5,
      "grace_period_s": 5.0,
      "activate_after_step": 2500,
      "box_name": "box",
      "command_name": "twist",
    },
  )

  # ── Hand force tracking reward ─────────────────────────────────────────
  # Encourages the robot to apply the MPC-optimal hand push forces when in
  # contact with the box.  Gated by push_mode so walk-only envs are unaffected.
  cfg.rewards["mpc_hand_force_tracking"] = RewardTermCfg(
    func=mpc_grf_mdp.mpc_hand_force_tracking,
    weight=1.0,
    params={
      "command_name": "loco_mpc",
      "lhand_sensor_name": "lhand_box_contact",
      "rhand_sensor_name": "rhand_box_contact",
      "sigma": 50.0,
    },
  )

  return cfg
