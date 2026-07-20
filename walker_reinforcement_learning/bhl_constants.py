"""Berkeley Humanoid Lite (BHL) constants.

Actuators (from the BHL motor spec sheet):
  - 6512 (MAD 6C12): hip roll/yaw/pitch + knee  -> gains DERIVED from measured rotor inertia
  - 5010 (MAD 5010):  ankle pitch/roll          -> gains from an ESTIMATED rotor inertia (see below)

List of variables ESTIMATED
Line 32: ROTOR_INERTIA_5010 = 2.9e-5 # kg*m^2
    When found we can derive the following: ARMATURE_5010 = J·15² → STIFFNESS_5010 = ARMATURE·ω² → DAMPING_5010 = 2ζ·ARMATURE·ω.
Line 34: NATURAL_FREQ = 2.0 * math.pi * 10.0  # 10 Hz, in rad/s
Line 35: DAMPING_RATIO = 2.0
     When found we can derive all 12 gains Kp and Kd for both motor groups
Line 88: friction=(0.6,)
    When derived: friction in FEET_ONLY_COLLISION
"""
import math
from pathlib import Path
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

BHL_XML: Path = Path("/Users/daniel/Desktop/SNU_SRBL/RL/rlbhl.xml") # Change path when used in the windows remote
assert BHL_XML.exists(), f"oops"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(BHL_XML))
GEAR_RATIO = 15.0
ROTOR_INERTIA_6512 = 9.94e-5  # kg*m^2
EFFORT_6512 = 14.89  # Nm
ROTOR_INERTIA_5010 = 2.9e-5  # kg*m^2
EFFORT_5010 = 19.05  # Nm
NATURAL_FREQ = 2.0 * math.pi * 10.0  # 10 Hz, in rad/s
DAMPING_RATIO = 2.0
ARMATURE_6512 = reflected_inertia(ROTOR_INERTIA_6512, GEAR_RATIO)
ARMATURE_5010 = reflected_inertia(ROTOR_INERTIA_5010, GEAR_RATIO)

STIFFNESS_6512 = ARMATURE_6512 * NATURAL_FREQ**2
DAMPING_6512 = 2.0 * DAMPING_RATIO * ARMATURE_6512 * NATURAL_FREQ
STIFFNESS_5010 = ARMATURE_5010 * NATURAL_FREQ**2
DAMPING_5010 = 2.0 * DAMPING_RATIO * ARMATURE_5010 * NATURAL_FREQ

BHL_ACTUATOR_6512 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint",
        ".*_hip_pitch_joint",
        ".*_knee_pitch_joint",
    ),
    stiffness=STIFFNESS_6512,
    damping=DAMPING_6512,
    effort_limit=EFFORT_6512,
    armature=ARMATURE_6512,
)

BHL_ACTUATOR_5010 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ),
    stiffness=STIFFNESS_5010,
    damping=DAMPING_5010,
    effort_limit=EFFORT_5010,
    armature=ARMATURE_5010,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.56),
    joint_pos={
        ".*_hip_pitch_joint": -0.25,
        ".*_knee_pitch_joint": 0.70,
        ".*_ankle_pitch_joint": -0.40,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        ".*_ankle_roll_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

BHL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(BHL_ACTUATOR_6512, BHL_ACTUATOR_5010),
    soft_joint_pos_limit_factor=0.9,
)

def get_bhl_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=BHL_ARTICULATION,
    )

BHL_ACTION_SCALE: dict[str, float] = {}
for a in BHL_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    assert a.effort_limit is not None
    for n in a.target_names_expr:
        BHL_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity
    robot = Entity(get_bhl_robot_cfg())
    viewer.launch(robot.spec.compile())