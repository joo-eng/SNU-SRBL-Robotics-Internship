"""
Bipedal robot standing balance controller using MuJoCo.
Uses Jacobian-based inverse kinematics with PID balance control.

READ THIS FIRST!
Run with (macOS requires mjpython for the interactive viewer, not
plain python) Other operating systems may have diferent ways of executing:
mjpython "/Path_to_file/stand_pid_controller.py"
ALSO edit the path/name of xml right under def main() in line 95.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# Variables:

# Model Configuration
LEGS = ["left", "right"]
JOINT_ORDER = ["hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"]
UNCONTROLLED_JOINTS = {"hip_yaw"}

# Standing Posture Targets
WIDEN = 0.0
KNEE = 0.0
HIP_PITCH_COMP = 0.0
ANKLE_PITCH_COMP = 0.0
STAND_TORSO_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Balance Control (PID Gains)
KP_XY, KD_XY = 60.0, 40.0 # Forward/Back stability
KP_Z, KD_Z = 250.0, 107.0 # Height stability
KP_ROT, KD_ROT = 20.0, 15.0 # Rotation (Proportional and Derivative)
KI_ROT, ROT_I_LIMIT = 15.0, 0.3 # Rotation (Integral)
YAW_DAMPING_FRACTION = 0.2 # Damping

# Posture Control
KP_POSTURE, KD_POSTURE = 0.0, 0.0

def sensor_adr(model, name): # Return the address of a sensor in the model.
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found in model")
    return model.sensor_adr[sid]

def actuator_id(model, name): # Return the ID of an actuator in the model.
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator '{name}' not found in model")
    return aid

def quat_to_roll_pitch(q): # Convert quaternion [w, x, y, z] to roll and pitch angles.
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch

def build_joint_targets(joint_keys): # Build target joint angles for standing posture.
    targets = np.zeros(len(joint_keys))
    for i, (leg, j) in enumerate(joint_keys):
        if j == "hip_roll":
            targets[i] = WIDEN if leg == "left" else -WIDEN
        elif j == "knee_pitch":
            targets[i] = KNEE
        elif j == "hip_pitch":
            targets[i] = HIP_PITCH_COMP
        elif j == "ankle_pitch":
            targets[i] = ANKLE_PITCH_COMP
    return targets


def find_standing_height(model, data, qadr, targets): # Find the CoM height where the robot stands without penetration.
    def penetration(z):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, z]
        data.qpos[3:7] = STAND_TORSO_QUAT
        for i, val in enumerate(targets):
            data.qpos[qadr[i]] = val
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            return None
        return min(data.contact[i].dist for i in range(data.ncon))
    lo, hi = -0.60, -0.15
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = penetration(mid)
        if p is None:
            hi = mid
        else:
            lo = mid
    return lo

def main(): # Initialize the robot model and run the balance control loop.
    model = mujoco.MjModel.from_xml_path("/Path/to/file/name_of_file.xml")
    data = mujoco.MjData(model)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    mtotal = model.body_subtreemass[torso_id]
    g = abs(model.opt.gravity[2])

    joint_keys = [(leg, j) for leg in LEGS for j in JOINT_ORDER]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg_{leg}_{j}_joint") for leg, j in joint_keys]
    leg_dof_idx = [model.jnt_dofadr[jid] for jid in joint_ids]
    qadr = [model.jnt_qposadr[jid] for jid in joint_ids]

    actuator_names = [f"leg_{leg}_{j}_motor" for leg, j in joint_keys]
    pos_sensor_names = [f"leg_{leg}_{j}_pos" for leg, j in joint_keys]
    vel_sensor_names = [f"leg_{leg}_{j}_vel" for leg, j in joint_keys]

    pos_adrs = [sensor_adr(model, n) for n in pos_sensor_names]
    vel_adrs = [sensor_adr(model, n) for n in vel_sensor_names]
    act_ids = [actuator_id(model, n) for n in actuator_names]
    ctrl_ranges = [model.actuator_ctrlrange[i].copy() for i in act_ids]

    quat_adr = sensor_adr(model, "torso_quat")
    gyro_adr = sensor_adr(model, "torso_gyro")

    n_all = len(joint_keys)
    joint_targets = build_joint_targets(joint_keys)

    ctrl_idx = [k for k, (leg, j) in enumerate(joint_keys) if j not in UNCONTROLLED_JOINTS]
    uncontrolled_idx = [k for k in range(n_all) if k not in ctrl_idx]

    leg_dof_idx_ctrl = [leg_dof_idx[k] for k in ctrl_idx]
    n = len(ctrl_idx)

    stand_z = find_standing_height(model, data, qadr, joint_targets)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, stand_z + 0.0005]
    data.qpos[3:7] = STAND_TORSO_QUAT
    for i, val in enumerate(joint_targets):
        data.qpos[qadr[i]] = val
    mujoco.mj_forward(model, data)

    for k in uncontrolled_idx:
        data.ctrl[act_ids[k]] = 0.0

    com0 = data.subtree_com[torso_id].copy()
    target_xy = com0[:2].copy()
    target_z = com0[2]
    com_prev = com0.copy()

    Jcom = np.zeros((3, model.nv))
    Jrot = np.zeros((3, model.nv))
    roll_i = 0.0
    pitch_i = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        dt = model.opt.timestep
        while viewer.is_running():
            mujoco.mj_jacSubtreeCom(model, data, Jcom, torso_id)
            mujoco.mj_jacBody(model, data, None, Jrot, torso_id)
            Jcom_leg = Jcom[:, leg_dof_idx_ctrl]
            Jrot_leg = Jrot[:, leg_dof_idx_ctrl]

            com = data.subtree_com[torso_id]
            com_vel = (com - com_prev) / dt
            com_prev = com.copy()

            quat = data.sensordata[quat_adr : quat_adr + 4]
            roll, pitch = quat_to_roll_pitch(quat)
            gyro_local = data.sensordata[gyro_adr : gyro_adr + 3]
            R = data.xmat[torso_id].reshape(3, 3)
            gyro_world = R @ gyro_local

            roll_i = np.clip(roll_i + (0.0 - roll) * dt, -ROT_I_LIMIT, ROT_I_LIMIT)
            pitch_i = np.clip(pitch_i + (0.0 - pitch) * dt, -ROT_I_LIMIT, ROT_I_LIMIT)

            fx = mtotal * (KP_XY * (target_xy[0] - com[0]) - KD_XY * com_vel[0])
            fy = mtotal * (KP_XY * (target_xy[1] - com[1]) - KD_XY * com_vel[1])
            fz = mtotal * g + mtotal * (KP_Z * (target_z - com[2]) - KD_Z * com_vel[2])
            f_com = np.array([fx, fy, fz])

            mx = KP_ROT * (0.0 - roll) - KD_ROT * gyro_world[0] + KI_ROT * roll_i
            my = KP_ROT * (0.0 - pitch) - KD_ROT * gyro_world[1] + KI_ROT * pitch_i
            mz = -YAW_DAMPING_FRACTION * KD_ROT * gyro_world[2]
            m_des = np.array([mx, my, mz])

            tau_com = Jcom_leg.T @ f_com
            tau_rot = Jrot_leg.T @ m_des

            j_task = np.vstack([Jcom_leg, Jrot_leg])
            j_task_pinv = np.linalg.pinv(j_task)
            nullspace = np.eye(n) - j_task_pinv @ j_task
            jpos = np.array([data.sensordata[pos_adrs[k]] for k in ctrl_idx])
            jvel = np.array([data.sensordata[vel_adrs[k]] for k in ctrl_idx])
            targets_ctrl = joint_targets[ctrl_idx]
            tau_posture_desired = KP_POSTURE * (targets_ctrl - jpos) - KD_POSTURE * jvel
            tau_posture = nullspace @ tau_posture_desired

            tau = tau_com + tau_rot + tau_posture

            tau_ctrl = tau / 15.0

            for k, joint_k in enumerate(ctrl_idx):
                lo, hi = ctrl_ranges[joint_k]
                data.ctrl[act_ids[joint_k]] = np.clip(tau_ctrl[k], lo, hi)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
