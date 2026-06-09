"""THEMIS TH02-A7 29-DOF humanoid constants.

Motor specifications and joint assignments are the single source of truth
for all actuator parameters used across training, sim2sim, and sim2real.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


# ══════════════════════════════════════════════════════════════════════════════
# Motor Specifications
# ══════════════════════════════════════════════════════════════════════════════
# Four physical actuator types on THEMIS. Every joint is assigned to exactly
# one motor type, which determines its kp, kd, torque limit, velocity limit,
# and rotor inertia (armature).
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MotorSpec:
  """Physical motor specification for a THEMIS actuator type."""
  stiffness: float       # kp — position gain (N·m/rad)
  damping: float         # kd — velocity gain (N·m·s/rad)
  effort_limit: float    # max torque (N·m)
  velocity_limit: float  # max joint velocity (rad/s)
  armature: float        # rotor inertia (kg·m²)


# koala — head / neck
KOALA = MotorSpec(
  stiffness=10.0,
  damping=1.0,
  effort_limit=10.5,
  velocity_limit=20.0,
  armature=0.0182,
)

# koala-mb — arms and ankles
# NOTE: ankle effort_limit set to 36 N·m.  kp/kd intentionally unchanged.  See
# the explicit action-scale pin after THEMIS_ACTION_SCALE is built — leaving
# the formula `0.25 * effort / kp` unmodified would change the tracking-env
# ankle scale away from the intended 0.5, which is *not* what we want.
KOALA_MB_ANKLE = MotorSpec(
  stiffness=20.0,
  damping=3.0,
  effort_limit=36.0,
  velocity_limit=20.0,
  armature=0.009,
)

KOALA_MB_ARM = MotorSpec(
  stiffness=15.0,
  damping=2.0,
  effort_limit=20.0,
  velocity_limit=20.0,
  armature=0.009,
)

# panda-p — hip yaw and hip roll
PANDA_P = MotorSpec(
  stiffness=35.0,
  damping=5.0,
  effort_limit=67.0,
  velocity_limit=15.0,
  armature=0.0297,
)

# kodiak — hip pitch and knee
KODIAK = MotorSpec(
  stiffness=35.0,
  damping=8.0,
  effort_limit=180.0,
  velocity_limit=10.0,
  armature=0.1153,
)


# ── Joint → Motor assignment ────────────────────────────────────────────────
# Every joint maps to exactly one motor type.  Change the motor assignment
# here and all downstream configs (training, sim2sim, sim2real) update.
# ─────────────────────────────────────────────────────────────────────────────

JOINT_MOTOR: dict[str, MotorSpec] = {
  # Right leg
  "HIP_YAW_R":      PANDA_P,
  "HIP_ROLL_R":     PANDA_P,
  "HIP_PITCH_R":    KODIAK,
  "KNEE_PITCH_R":   KODIAK,
  "ANKLE_PITCH_R":  KOALA_MB_ANKLE,
  "ANKLE_ROLL_R":   KOALA_MB_ANKLE,
  # Left leg
  "HIP_YAW_L":      PANDA_P,
  "HIP_ROLL_L":     PANDA_P,
  "HIP_PITCH_L":    KODIAK,
  "KNEE_PITCH_L":   KODIAK,
  "ANKLE_PITCH_L":  KOALA_MB_ANKLE,
  "ANKLE_ROLL_L":   KOALA_MB_ANKLE,
  # Right arm
  "SHOULDER_PITCH_R": KOALA_MB_ARM,
  "SHOULDER_ROLL_R":  KOALA_MB_ARM,
  "SHOULDER_YAW_R":   KOALA_MB_ARM,
  "ELBOW_PITCH_R":    KOALA_MB_ARM,
  "ELBOW_YAW_R":      KOALA_MB_ARM,
  "WRIST_PITCH_R":    KOALA_MB_ARM,
  "WRIST_YAW_R":      KOALA_MB_ARM,
  # Left arm
  "SHOULDER_PITCH_L": KOALA_MB_ARM,
  "SHOULDER_ROLL_L":  KOALA_MB_ARM,
  "SHOULDER_YAW_L":   KOALA_MB_ARM,
  "ELBOW_PITCH_L":    KOALA_MB_ARM,
  "ELBOW_YAW_L":      KOALA_MB_ARM,
  "WRIST_PITCH_L":    KOALA_MB_ARM,
  "WRIST_YAW_L":      KOALA_MB_ARM,
  # Head
  "HEAD_YAW":   KOALA,
  "HEAD_PITCH": KOALA,
}


# ══════════════════════════════════════════════════════════════════════════════
# Derived lookup tables (auto-generated from JOINT_MOTOR)
# ══════════════════════════════════════════════════════════════════════════════

STIFFNESS: dict[str, float] = {j: m.stiffness for j, m in JOINT_MOTOR.items()}
DAMPING: dict[str, float] = {j: m.damping for j, m in JOINT_MOTOR.items()}
EFFORT_LIMIT: dict[str, float] = {j: m.effort_limit for j, m in JOINT_MOTOR.items()}
VELOCITY_LIMIT: dict[str, float] = {j: m.velocity_limit for j, m in JOINT_MOTOR.items()}
ARMATURE: dict[str, float] = {j: m.armature for j, m in JOINT_MOTOR.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Actuator configs (derived from motor specs)
# ══════════════════════════════════════════════════════════════════════════════

##
# MJCF and assets.
##
_HERE = Path(__file__).parent

THEMIS_XML: Path = _HERE / "xmls" / "themis_29dof.xml"
assert THEMIS_XML.exists(), f"Missing THEMIS XML: {THEMIS_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, THEMIS_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(THEMIS_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def _actuator(motor: MotorSpec, *target_patterns: str) -> BuiltinPositionActuatorCfg:
  """Create an actuator config from a motor spec and joint name patterns."""
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_patterns,
    stiffness=motor.stiffness,
    damping=motor.damping,
    effort_limit=motor.effort_limit,
    armature=motor.armature,
  )


HIP_YAW_ROLL_ACTUATOR = _actuator(PANDA_P, "HIP_YAW_.*", "HIP_ROLL_.*")
HIP_PITCH_KNEE_ACTUATOR = _actuator(KODIAK, "HIP_PITCH_.*", "KNEE_PITCH_.*")
ANKLE_ACTUATOR = _actuator(KOALA_MB_ANKLE, "ANKLE_PITCH_.*", "ANKLE_ROLL_.*")
ARM_ACTUATOR = _actuator(
  KOALA_MB_ARM,
  "SHOULDER_PITCH_.*", "SHOULDER_ROLL_.*", "SHOULDER_YAW_.*",
  "ELBOW_PITCH_.*", "ELBOW_YAW_.*", "WRIST_PITCH_.*", "WRIST_YAW_.*",
)
HEAD_ACTUATOR = _actuator(KOALA, "HEAD_YAW", "HEAD_PITCH")


##
# Joint ordering.
# Follows the same convention as the MJCF actuator block in the original XML:
#   right leg → left leg → right arm → left arm → head
##
JOINT_NAMES_EXPR = [
  # Right leg (6)
  "HIP_YAW_R",
  "HIP_ROLL_R",
  "HIP_PITCH_R",
  "KNEE_PITCH_R",
  "ANKLE_PITCH_R",
  "ANKLE_ROLL_R",
  # Left leg (6)
  "HIP_YAW_L",
  "HIP_ROLL_L",
  "HIP_PITCH_L",
  "KNEE_PITCH_L",
  "ANKLE_PITCH_L",
  "ANKLE_ROLL_L",
  # Right arm (7)
  "SHOULDER_PITCH_R",
  "SHOULDER_ROLL_R",
  "SHOULDER_YAW_R",
  "ELBOW_PITCH_R",
  "ELBOW_YAW_R",
  "WRIST_PITCH_R",
  "WRIST_YAW_R",
  # Left arm (7)
  "SHOULDER_PITCH_L",
  "SHOULDER_ROLL_L",
  "SHOULDER_YAW_L",
  "ELBOW_PITCH_L",
  "ELBOW_YAW_L",
  "WRIST_PITCH_L",
  "WRIST_YAW_L",
  # Head (2)
  "HEAD_YAW",
  "HEAD_PITCH",
]

##
# Home keyframe – matches IsaacLab TH02_A7_CFG init_state.
##


# Walking default pose — arms held in a compact swing posture (elbows bent).
HOME_KEYFRAME_WALKING = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.17),
  joint_pos={
    "HIP_YAW_R":   0.00,
    "HIP_YAW_L":   0.00,
    "HIP_ROLL_.*":  0.0,
    "HIP_PITCH_.*": -0.2,
    "KNEE_PITCH_.*": 0.5,
    "ANKLE_PITCH_.*": -0.3,
    "ANKLE_ROLL_.*": 0.0,
    "SHOULDER_PITCH_.*": 0.2*1.0,
    "SHOULDER_ROLL_R":  -0.25*1.0,
    "SHOULDER_ROLL_L":  0.25*1.0,
    "SHOULDER_YAW_R":   0.1,
    "SHOULDER_YAW_L":   -0.1,
    "ELBOW_PITCH_.*": 1.2,
    "ELBOW_YAW_.*":   0.0,
    "WRIST_PITCH_.*": 0.0,
    "WRIST_YAW_.*":   0.0,
    "HEAD_YAW":   0.0,
    "HEAD_PITCH": 0.0,
  },
  joint_vel={".*": 0.0},
)


# Loco-manipulation default pose — arms lowered and extended (elbows straight)
# for pushing.
HOME_KEYFRAME_LOCO_MANIP = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.17),
  joint_pos={
    "HIP_YAW_R":   0.00,
    "HIP_YAW_L":   0.00,
    "HIP_ROLL_.*":  0.0,
    "HIP_PITCH_.*": -0.2,
    "KNEE_PITCH_.*": 0.5,
    "ANKLE_PITCH_.*": -0.3,
    "ANKLE_ROLL_.*": 0.0,
    "SHOULDER_PITCH_.*": 0.1,
    "SHOULDER_ROLL_R":  -0.2*1.0,
    "SHOULDER_ROLL_L":  0.2*1.0,
    "SHOULDER_YAW_R":   0.04,
    "SHOULDER_YAW_L":   -0.04,
    "ELBOW_PITCH_.*": 1.2*0.0,
    "ELBOW_YAW_.*":   0.0,
    "WRIST_PITCH_.*": 0.0,
    "WRIST_YAW_.*":   0.0,
    "HEAD_YAW":   0.0,
    "HEAD_PITCH": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Default (walking) keyframe; kept as the module-level alias.
HOME_KEYFRAME = HOME_KEYFRAME_WALKING


##
# Single-left-foot balance keyframe.
# Left leg: nearly straight, planted as stance foot.
# Right leg: hip flexed up, knee bent, foot hanging in the air.
# Arms: shoulder-rolled out sideways (T-pose style) for balance, elbow extended.
# Base height ~1.05 m: CoM roughly above the left foot sole.
##
BALANCE_LEFT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.15),
  joint_pos={
    # ── Stance leg (left) — nearly straight ──
    "HIP_YAW_L":    -0.03,
    "HIP_ROLL_L":   -0.16,
    "HIP_PITCH_L":  -0.16,   # slight forward lean
    "KNEE_PITCH_L":  0.33,   # slight knee bend
    "ANKLE_PITCH_L": -0.19,  # ankle compensates
    "ANKLE_ROLL_L":  0.14,
    # ── Swing leg (right) — hip flexed, knee bent, foot hanging ──
    "HIP_YAW_R":    0.0,
    "HIP_ROLL_R":   0.0,
    "HIP_PITCH_R":  -0.87,   # hip flexed forward
    "KNEE_PITCH_R":  1.88,   # knee bent so foot hangs clear of ground
    "ANKLE_PITCH_R": -0.66,  # foot roughly horizontal
    "ANKLE_ROLL_R":  0.0,
    # ── Arms out to sides for balance (T-pose, elbows extended) ──
    "SHOULDER_PITCH_R":  0.0,
    "SHOULDER_ROLL_R":  -1.2,  # abduct right arm out
    "SHOULDER_YAW_R":    0.0,
    "ELBOW_PITCH_R":     1.2,  # elbow straight
    "ELBOW_YAW_R":       0.0,
    "WRIST_PITCH_R":     0.0,
    "WRIST_YAW_R":       0.0,
    "SHOULDER_PITCH_L":  0.0,
    "SHOULDER_ROLL_L":   1.2,  # abduct left arm out
    "SHOULDER_YAW_L":    0.0,
    "ELBOW_PITCH_L":     1.2,  # elbow straight
    "ELBOW_YAW_L":       0.0,
    "WRIST_PITCH_L":     0.0,
    "WRIST_YAW_L":       0.0,
    # ── Head neutral ──
    "HEAD_YAW":   0.0,
    "HEAD_PITCH": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Re-enable hand contact geoms after FEET_ONLY_COLLISION disables them.
# FEET_ONLY_COLLISION has disable_other_geoms=True, which zeroes out every
# geom not matching .*_collision — including hand_contact_L/R.  This second
# pass restores those geoms so they can be used for box-push contact sensing.
HAND_CONTACT_COLLISION = CollisionCfg(
  geom_names_expr=("hand_contact_.*",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.0, 0.005, 0.0001),
  disable_other_geoms=False,
)


##
# Final articulation config.
##
THEMIS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HIP_YAW_ROLL_ACTUATOR,
    HIP_PITCH_KNEE_ACTUATOR,
    ANKLE_ACTUATOR,
    ARM_ACTUATOR,
    HEAD_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_themis_robot_cfg(loco_manip: bool = False) -> EntityCfg:
  """Get a fresh THEMIS robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    loco_manip: When ``True``, initialise with the loco-manipulation default
      pose (arms lowered/extended for pushing); otherwise use the walking pose.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME_LOCO_MANIP if loco_manip else HOME_KEYFRAME_WALKING,
    collisions=(FEET_ONLY_COLLISION, HAND_CONTACT_COLLISION),
    spec_fn=get_spec,
    articulation=THEMIS_ARTICULATION,
  )


##
# Action scale table 
##
THEMIS_ACTION_SCALE: dict[str, float] = {}
for _a in THEMIS_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    THEMIS_ACTION_SCALE[_n] = 0.25 * _e / _s

# manual set to 0.5 for ankles to allow more aggressive foot placement during walking and balance tasks; see the comment in the KOALA_MB_ANKLE motor spec for rationale.
for _n in ("ANKLE_PITCH_R", "ANKLE_ROLL_R", "ANKLE_PITCH_L", "ANKLE_ROLL_L"):
  THEMIS_ACTION_SCALE[_n] = 0.5


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_themis_robot_cfg())
  viewer.launch(robot.spec.compile())
