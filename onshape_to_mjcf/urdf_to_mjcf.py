"""
urdf_to_mjcf.py -- convert an onshape-to-robot URDF into a MuJoCo MJCF file.
Generated using my beloved Claude <3

WHAT IT CARRIES OVER
  - <asset> mesh list, deduplicated, built from every mesh filename in the URDF
  - body tree (URDF parent/child joints -> nested MJCF <body>)
  - geoms: rpy -> quat, rgba from URDF <material>, visual/collision pairs collapsed
  - inertials: full 3x3 inertia tensor -> diaginertia + principal-axis quat
  - joints: axis and range (from URDF <limit lower/upper>)
  - <sensor>: jointpos/jointvel per actuated joint, plus IMU + CoM tracking

WHAT IT DOES NOT CARRY OVER (by design)
  - <actuator> block. Actuators are defined elsewhere in your pipeline.
  - actuatorfrcrange on each <joint>. URDF only has a uniform effort="10" on
    every joint, which does not reflect the real dual-motor setup. The script
    therefore EMITS NO actuatorfrcrange and instead writes a TODO banner listing
    every actuated joint, so nothing is silently wrong. Fill these in from the
    same place your actuators are defined. See the FRC_RANGE dict below if you
    would rather have the script write them.

USAGE
    python urdf_to_mjcf.py --urdf BHL_rl.urdf --output BHL.xml
    python urdf_to_mjcf.py --urdf BHL_rl.urdf --output BHL.xml --meshdir BHLasset
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# ----------------------------------------------------------------------------
# CONFIG -- edit these for a different robot
# ----------------------------------------------------------------------------

# URDF root link gets renamed to this in the MJCF.
ROOT_BODY_NAME = "torso"

# Free joint added to the root body so the robot can fall/move in sim.
ROOT_FREE_JOINT = "torso_free"

# Site placed at the root body origin. IMU + CoM sensors attach here.
ROOT_SITE_NAME = "torso_center"

# Links to drop entirely (URDF frame-markers that MJCF represents as sites).
SKIP_LINKS = {"imu", "imu_2"}

# Optional: joint-name substring -> actuatorfrcrange magnitude.
# Left empty so the script emits a TODO instead of guessing. If you want the
# script to write them, fill this in, e.g. {"ankle": 15} with DEFAULT_FRC = 14.
FRC_RANGE = {}
DEFAULT_FRC = None


# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------

def rpy_to_quat(r, p, y):
    """URDF fixed-axis roll-pitch-yaw -> MuJoCo (w, x, y, z) quaternion."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def jacobi_eigen(A, iterations=100):
    """Symmetric 3x3 eigendecomposition. Returns (eigenvalues, eigenvectors).

    Pure stdlib -- no numpy dependency. Jacobi rotation is plenty accurate for
    a 3x3 inertia tensor and avoids adding a package to your environment.
    """
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(iterations):
        off = sum(a[i][j] ** 2 for i in range(3) for j in range(3) if i != j)
        if off < 1e-20:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                if abs(a[p][q]) < 1e-18:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(3):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(3):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(3):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    return [a[i][i] for i in range(3)], v


def matrix_to_quat(m):
    """3x3 rotation matrix -> (w, x, y, z)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return (w, x, y, z)


def inertia_to_diag(ixx, ixy, ixz, iyy, iyz, izz):
    """Full inertia tensor -> (diaginertia, quat) in MuJoCo convention.

    MuJoCo wants the inertia expressed in its principal frame: three diagonal
    values plus a quaternion rotating body frame -> principal frame.
    """
    A = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    vals, vecs = jacobi_eigen(A)

    order = sorted(range(3), key=lambda i: -vals[i])
    vals = [vals[i] for i in order]
    R = [[vecs[r][order[c]] for c in range(3)] for r in range(3)]

    # Force a right-handed frame; a reflection is not a valid rotation.
    det = (
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
        - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
        + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
    )
    if det < 0:
        for r in range(3):
            R[r][2] = -R[r][2]

    return vals, matrix_to_quat(R)


def fmt(vals, prec=6):
    """Format a float sequence compactly: 0.0 -> '0', 1.500000 -> '1.5'."""
    out = []
    for v in vals:
        if abs(v) < 1e-12:
            out.append("0")
        else:
            s = f"{v:.{prec}g}"
            out.append(s)
    return " ".join(out)


# ----------------------------------------------------------------------------
# URDF parsing
# ----------------------------------------------------------------------------

def parse_origin(elem):
    """<origin xyz rpy> -> (xyz tuple, rpy tuple). Missing origin = identity."""
    if elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    o = elem.find("origin")
    if o is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(x) for x in o.get("xyz", "0 0 0").split())
    rpy = tuple(float(x) for x in o.get("rpy", "0 0 0").split())
    return xyz, rpy


def mesh_name(filename):
    """'package://assets/foo.stl' -> 'foo'."""
    return Path(filename.split("/")[-1]).stem


def parse_urdf(path):
    """Read the URDF into plain dicts: links, joints, and the mesh list."""
    tree = ET.parse(path)
    root = tree.getroot()

    links, meshes, seen = {}, [], set()

    for link in root.findall("link"):
        name = link.get("name")
        entry = {"name": name, "inertial": None, "geoms": []}

        inert = link.find("inertial")
        if inert is not None:
            xyz, rpy = parse_origin(inert)
            mass_el = inert.find("mass")
            i_el = inert.find("inertia")
            if mass_el is not None and i_el is not None:
                entry["inertial"] = {
                    "pos": xyz,
                    "rpy": rpy,
                    "mass": float(mass_el.get("value")),
                    "i": tuple(
                        float(i_el.get(k))
                        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                    ),
                }

        # Only <visual> is read. URDF from onshape-to-robot duplicates each mesh
        # as a matching <collision>, and the target MJCF collapses them into one
        # visual geom, so reading both would double every geom.
        for vis in link.findall("visual"):
            g = vis.find("geometry")
            if g is None:
                continue
            m = g.find("mesh")
            if m is None:
                continue

            fn = m.get("filename")
            mname = mesh_name(fn)
            if mname not in seen:
                seen.add(mname)
                meshes.append((mname, Path(fn.split("/")[-1]).name))

            xyz, rpy = parse_origin(vis)
            rgba = "0.8 0.8 0.8 1"
            mat = vis.find("material")
            if mat is not None:
                col = mat.find("color")
                if col is not None:
                    rgba = col.get("rgba", rgba)

            entry["geoms"].append({"mesh": mname, "pos": xyz,
                                   "rpy": rpy, "rgba": rgba})

        links[name] = entry

    joints = []
    for j in root.findall("joint"):
        xyz, rpy = parse_origin(j)
        axis_el = j.find("axis")
        axis = (
            tuple(float(x) for x in axis_el.get("xyz", "0 0 1").split())
            if axis_el is not None
            else (0.0, 0.0, 1.0)
        )
        lim = j.find("limit")
        lower = upper = None
        if lim is not None and lim.get("lower") is not None:
            lower = float(lim.get("lower"))
            upper = float(lim.get("upper"))

        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "pos": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        })

    return links, joints, meshes


def find_root(links, joints):
    """The one link that is never a child is the root."""
    children = {j["child"] for j in joints}
    roots = [n for n in links if n not in children]
    if not roots:
        raise ValueError("No root link found -- the URDF tree has a cycle.")
    return roots[0]


# ----------------------------------------------------------------------------
# MJCF emission
# ----------------------------------------------------------------------------

def emit_body(name, links, joints_by_parent, joint_in, indent, actuated, out):
    """Recursively write one <body> and everything below it."""
    pad = "  " * indent
    link = links[name]
    disp = ROOT_BODY_NAME if joint_in is None else name

    if joint_in is None:
        out.append(f'{pad}<body name="{disp}" pos="0 0 0" quat="1 0 0 0">')
    else:
        q = rpy_to_quat(*joint_in["rpy"])
        out.append(
            f'{pad}<body name="{disp}" pos="{fmt(joint_in["pos"])}" '
            f'quat="{fmt(q)}">'
        )

    if link["inertial"]:
        inr = link["inertial"]
        diag, iq = inertia_to_diag(*inr["i"])
        out.append(
            f'{pad}  <inertial pos="{fmt(inr["pos"])}" quat="{fmt(iq)}" '
            f'mass="{fmt([inr["mass"]])}" diaginertia="{fmt(diag)}"/>'
        )

    if joint_in is None:
        out.append(f'{pad}  <joint name="{ROOT_FREE_JOINT}" type="free"/>')
        out.append(f'{pad}  <site name="{ROOT_SITE_NAME}" pos="0 0 0" size="0.01"/>')
    elif joint_in["type"] in ("revolute", "continuous", "prismatic"):
        jt = ' type="slide"' if joint_in["type"] == "prismatic" else ""
        rng = ""
        if joint_in["lower"] is not None:
            rng = f' range="{fmt([joint_in["lower"], joint_in["upper"]])}"'

        frc = ""
        if DEFAULT_FRC is not None:
            mag = DEFAULT_FRC
            for key, val in FRC_RANGE.items():
                if key in joint_in["name"]:
                    mag = val
                    break
            frc = f' actuatorfrcrange="{-mag} {mag}"'

        out.append(
            f'{pad}  <joint name="{joint_in["name"]}"{jt} pos="0 0 0" '
            f'axis="{fmt(joint_in["axis"])}"{rng}{frc}/>'
        )
        actuated.append(joint_in["name"])

    for g in link["geoms"]:
        q = rpy_to_quat(*g["rpy"])
        out.append(
            f'{pad}  <geom pos="{fmt(g["pos"])}" quat="{fmt(q)}" type="mesh" '
            f'rgba="{g["rgba"]}" mesh="{g["mesh"]}"/>'
        )

    for j in joints_by_parent.get(name, []):
        if j["child"] in SKIP_LINKS:
            continue
        emit_body(j["child"], links, joints_by_parent, j,
                  indent + 1, actuated, out)

    out.append(f"{pad}</body>")


def build_sensors(actuated):
    """jointpos/jointvel per actuated joint, then IMU and CoM tracking."""
    out = ["  <sensor>", "    <!--Joint Positions-->"]
    for jn in actuated:
        base = jn[:-6] if jn.endswith("_joint") else jn
        out.append(f'    <jointpos name="{base}_pos" joint="{jn}"/>')
        out.append(f'    <jointvel name="{base}_vel" joint="{jn}"/>')
    out += [
        "    <!--For IMU-->",
        f'    <gyro name="imu_ang_vel" site="{ROOT_SITE_NAME}"/>',
        f'    <velocimeter name="imu_lin_vel" site="{ROOT_SITE_NAME}"/>',
        f'    <accelerometer name="imu_lin_acc" site="{ROOT_SITE_NAME}"/>',
        f'    <framezaxis name="imu_upvector" objtype="body" objname="world" '
        f'reftype="site" refname="{ROOT_SITE_NAME}"/>',
        f'    <subtreeangmom name="root_angmom" body="{ROOT_BODY_NAME}"/>',
        f'    <gyro name="torso_gyro" site="{ROOT_SITE_NAME}"/>',
        f'    <accelerometer name="torso_accel" site="{ROOT_SITE_NAME}"/>',
        "    <!--CoM tracking-->",
        f'    <framepos name="torso_pos" objtype="body" objname="{ROOT_BODY_NAME}"/>',
        f'    <framequat name="torso_quat" objtype="body" objname="{ROOT_BODY_NAME}"/>',
        f'    <framelinvel name="torso_linvel" objtype="body" objname="{ROOT_BODY_NAME}"/>',
        f'    <frameangvel name="torso_angvel" objtype="body" objname="{ROOT_BODY_NAME}"/>',
        "  </sensor>",
    ]
    return out


def convert(urdf_path, out_path, meshdir, model_name):
    links, joints, meshes = parse_urdf(urdf_path)

    joints_by_parent = {}
    for j in joints:
        joints_by_parent.setdefault(j["parent"], []).append(j)

    root_link = find_root(links, joints)
    actuated, body_lines = [], []
    emit_body(root_link, links, joints_by_parent, None, 2, actuated, body_lines)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- {model_name} -- generated by urdf_to_mjcf.py from "
        f"{Path(urdf_path).name} -->",
        "<!--",
        "  NOT INCLUDED IN THIS FILE, ADD SEPARATELY:",
        "",
        "  1. <actuator> block. Defined elsewhere in the pipeline.",
        "",
        "  2. actuatorfrcrange on each <joint> below. The source URDF carries a",
        "     uniform effort=\"10\" on every joint, which does not reflect the",
        "     real per-motor torque limits, so nothing was emitted rather than",
        "     writing a value that is wrong. Add a range to each of these:",
    ]
    for jn in actuated:
        lines.append(f"       - {jn}")
    lines += [
        "",
        "  3. Gains (kp/kd). URDF has no equivalent field.",
        "-->",
        f'<mujoco model="{model_name}">',
        f'  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>',
        "",
        "  <default>",
        "    <!-- Visual-only unless a geom explicitly re-enables contact. -->",
        '    <geom contype="0" conaffinity="0"/>',
        "  </default>",
        "",
        "  <asset>",
    ]
    for mname, fname in meshes:
        lines.append(
            f'    <mesh name="{mname}" content_type="model/stl" file="{fname}"/>'
        )
    lines += ["  </asset>", "", "  <worldbody>"]
    lines += body_lines
    lines += ["  </worldbody>"]
    lines += build_sensors(actuated)
    lines.append("</mujoco>")

    Path(out_path).write_text("\n".join(lines) + "\n")
    return len(meshes), len(actuated), len(links)


def main():
    ap = argparse.ArgumentParser(
        description="Convert an onshape-to-robot URDF into a MuJoCo MJCF file."
    )
    ap.add_argument("--urdf", required=True, help="Path to the input .urdf")
    ap.add_argument("--output", required=True, help="Path for the output .xml")
    ap.add_argument("--meshdir", default="BHLasset",
                    help="meshdir attribute for <compiler> (default: BHLasset)")
    ap.add_argument("--model-name", default="mjcf",
                    help="MuJoCo model name (default: mjcf)")
    args = ap.parse_args()

    if not Path(args.urdf).exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    nm, na, nl = convert(args.urdf, args.output, args.meshdir, args.model_name)

    print(f"Wrote {args.output}")
    print(f"  {nl} links -> bodies, {nm} meshes, {na} actuated joints")
    print(f"  Reminder: actuatorfrcrange is NOT set on any joint. "
          f"See the header comment in {args.output}.")


if __name__ == "__main__":
    main()
