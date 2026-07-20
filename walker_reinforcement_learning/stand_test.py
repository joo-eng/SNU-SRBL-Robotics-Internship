"""
Paste:
mjpython "/Users/daniel/Desktop/SNU_SRBL/RL/stand_test.py" "/Users/daniel/Desktop/SNU_SRBL/RL/rlbhl.xml" --view 
(changes depending on setup)
"""
import argparse
import math
import os
import re
import numpy as np
import mujoco

GEAR_RATIO = 15.0
ROTOR_INERTIA_6512 = 9.94e-5
ROTOR_INERTIA_5010 = 2.9e-5  # ESTIMATE
NATURAL_FREQ = 2.0 * math.pi * 10.0
DAMPING_RATIO = 2.0
EFFORT = {"6512": 14.89, "5010": 19.05}
KEYFRAME = {
    r".*_hip_pitch_joint": -0.25,
    r".*_knee_pitch_joint": 0.70,
    r".*_ankle_pitch_joint": -0.40,
    r".*_hip_roll_joint": 0.0,
    r".*_hip_yaw_joint": 0.0,
    r".*_ankle_roll_joint": 0.0,
}

def gains(J):
    arm = J * GEAR_RATIO**2
    return arm, arm * NATURAL_FREQ**2, 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ

_, KP6, KD6 = gains(ROTOR_INERTIA_6512)
_, KP5, KD5 = gains(ROTOR_INERTIA_5010)
KP = {"6512": KP6, "5010": KP5}
KD = {"6512": KD6, "5010": KD5}
group = lambda j: "5010" if "ankle" in j else "6512"

def target(joint):
    for pat, val in KEYFRAME.items():
        if re.fullmatch(pat, joint):
            return val
    return 0.0

def build_model(xml_path):
    spec = mujoco.MjSpec.from_file(xml_path)
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 3]
    light.dir = [0, 0, -1]
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.pos = [0, 0, 0]
    floor.contype = 1
    floor.conaffinity = 1
    floor.condim = 3
    for j in spec.joints:
        if j.name == "torso_free":
            continue
        g = group(j.name)
        a = spec.add_actuator()
        a.name = j.name.replace("_joint", "") + "_pos"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = j.name
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = KP[g]
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[1] = -KP[g]
        a.biasprm[2] = -KD[g]
        a.forcerange = [-EFFORT[g], EFFORT[g]]
    return spec.compile()
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?", default=None, help="path to robot MJCF")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--save", default=None, help="optional PNG output path")
    ap.add_argument(
        "--view",
        action="store_true",
        help="open interactive 3D viewer holding the pose (macOS: run with mjpython)",
    )
    args = ap.parse_args()

    xml_path = args.xml
    if xml_path is None:
        for cand in ("bhl.xml", "rlbhl.xml"):
            if os.path.exists(cand):
                xml_path = cand
                break
    if xml_path is None or not os.path.exists(xml_path):
        raise SystemExit(
            "Robot XML not found. Pass the path explicitly:\n"
            "    python stand_test.py /path/to/rlbhl.xml"
        )

    m = build_model(xml_path)
    d = mujoco.MjData(m)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = [g for g in range(m.ngeom) if g != floor]

    hinges = []
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n == "torso_free":
            continue
        hinges.append((n, int(m.jnt_qposadr[i]), int(m.jnt_dofadr[i]), target(n)))

    def apply_keyframe(base_z):
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qpos[0:3] = [0, 0, base_z]
        d.qpos[3:7] = [1, 0, 0, 0]
        for _, qa, _, tv in hinges:
            d.qpos[qa] = tv
        mujoco.mj_forward(m, d)

    apply_keyframe(1.0)
    low = float(np.min([d.geom_xpos[g, 2] - m.geom_rbound[g] for g in robot_geoms]))
    apply_keyframe(1.0 - low + 0.003)

    for ai in range(m.nu):
        jid = m.actuator_trnid[ai, 0]
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        d.ctrl[ai] = target(jn)

    if args.view:
        import time
        import mujoco.viewer as mjv
 
        print("Opening interactive viewer -- close the window to exit.")
        print("(macOS: this must be run with `mjpython`, not `python`.)")
        print("Open the 'Control' panel (top-right) to see the 12 joint actuators;")
        print("drag a slider to command that joint and watch the leg move.")
        with mjv.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                mujoco.mj_step(m, d)
                v.sync()
                lag = m.opt.timestep - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
        return

    for _ in range(int(args.seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)

    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    up = float(d.xmat[tid].reshape(3, 3)[2, 2])
    lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    contacts = {lf: 0, rf: 0}
    for c in range(d.ncon):
        for gg in (d.contact[c].geom1, d.contact[c].geom2):
            if gg in contacts:
                contacts[gg] += 1

    stands = up > 0.95 and contacts[lf] > 0 and contacts[rf] > 0 and d.qpos[2] > 0.4
    print(f"robot xml:            {xml_path}")
    print(f"gains: hip/knee kp={KP6:.1f} kd={KD6:.2f} | ankle kp={KP5:.1f} kd={KD5:.2f}")
    print(f"settled torso height: {d.qpos[2]:.3f} m")
    print(f"uprightness:          {up:.3f}  (1.0 = vertical)")
    print(f"base speed:           {np.linalg.norm(d.qvel[0:3]):.4f} m/s")
    print(f"foot contacts:        L={contacts[lf]} R={contacts[rf]}")
    print("VERDICT:", "STANDS" if stands else "FALLS / unstable")

    if args.save:
        with mujoco.Renderer(m, 480, 640) as r:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance = 2.0
            cam.elevation = -15
            cam.lookat[:] = [0, 0, 0.3]
            r.update_scene(d, cam)
            import PIL.Image

            PIL.Image.fromarray(r.render()).save(args.save)
            print("saved", args.save)

if __name__ == "__main__":
    main()