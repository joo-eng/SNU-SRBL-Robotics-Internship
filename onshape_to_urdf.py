"""
Manual:

OnShape Model Requirements (Be sure to ONLY contain these files, to prevent junk in the .urdf):

Body:
- Aluminum Extrusions
- Batteries
- Power Relay
- Emergency Stopper
- Battery Holder
- Conventor
- Diodes

Legs:
- Ankle Mounts
- Feet Mounts
- Housing
- Motor Shells
- Rear Structure
- Stopper

config.json:
Check if you have a file named config.json and paste this into the file:
{
  "output_filename": "(put file name here)",
  "document_id": "(the id to OnShape file)",
  "output_format": "urdf",
  "assembly_name": "Assembly"
}

Edits: Edit the following lines to adjust this script.
Line 46: Paste in access key
Line 47: Paste in secret key (do not share secret key to others)
Line 57: Edit path to config.json

Sensors are customizable to ones liking
"""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ["ONSHAPE_API"] = "https://cad.onshape.com"
os.environ["ONSHAPE_ACCESS_KEY"] = "Access Key" # Customize Access Key
os.environ["ONSHAPE_SECRET_KEY"] = "Secret Key" # Customize Secret Key

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate URDF file from onshape CAD project.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file.",
        default="/path/to/config.json", # Customize Path to config.json
    )
    args = parser.parse_args()

    config_file_path = Path(args.config)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file {config_file_path} does not exist!")

    robot_name = json.load(open(config_file_path))["output_filename"]
    mjcf_dir = config_file_path.parent
    robot_dir = mjcf_dir.parent
    scad_dir = robot_dir / "scad"

    if scad_dir.exists():
        assets_dir = mjcf_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        for file in scad_dir.iterdir():
            shutil.copy(file, assets_dir / file.name)

    shutil.copy(config_file_path, robot_dir / "config.json")
    os.system(f"onshape-to-robot {robot_dir}")

    if (mjcf_dir / "assets" / "merged").exists():
        shutil.copytree(
            mjcf_dir / "assets" / "merged",
            robot_dir / "meshes",
            dirs_exist_ok=True,
        )

    if (mjcf_dir / "assets").exists():
        shutil.rmtree(mjcf_dir / "assets")

    with open(mjcf_dir / f"{robot_name}.xml", "r") as file:
        content = file.read()

    content = content.replace("assets/merged/", "../meshes/")

    content = content.replace("</actuator>", """</actuator>

  <sensor>
    <jointpos name="leg_left_hip_roll_pos"        joint="leg_left_hip_roll_joint"/>
    <jointpos name="leg_left_hip_yaw_pos"         joint="leg_left_hip_yaw_joint"/>
    <jointpos name="leg_left_hip_pitch_pos"       joint="leg_left_hip_pitch_joint"/>
    <jointpos name="leg_left_knee_pitch_pos"      joint="leg_left_knee_pitch_joint"/>
    <jointpos name="leg_left_ankle_pitch_pos"     joint="leg_left_ankle_pitch_joint"/>
    <jointpos name="leg_left_ankle_roll_pos"      joint="leg_left_ankle_roll_joint"/>
    <jointpos name="leg_right_hip_roll_pos"       joint="leg_right_hip_roll_joint"/>
    <jointpos name="leg_right_hip_yaw_pos"        joint="leg_right_hip_yaw_joint"/>
    <jointpos name="leg_right_hip_pitch_pos"      joint="leg_right_hip_pitch_joint"/>
    <jointpos name="leg_right_knee_pitch_pos"     joint="leg_right_knee_pitch_joint"/>
    <jointpos name="leg_right_ankle_pitch_pos"    joint="leg_right_ankle_pitch_joint"/>
    <jointpos name="leg_right_ankle_roll_pos"     joint="leg_right_ankle_roll_joint"/>

    <jointvel name="leg_left_hip_roll_vel"        joint="leg_left_hip_roll_joint"/>
    <jointvel name="leg_left_hip_yaw_vel"         joint="leg_left_hip_yaw_joint"/>
    <jointvel name="leg_left_hip_pitch_vel"       joint="leg_left_hip_pitch_joint"/>
    <jointvel name="leg_left_knee_pitch_vel"      joint="leg_left_knee_pitch_joint"/>
    <jointvel name="leg_left_ankle_pitch_vel"     joint="leg_left_ankle_pitch_joint"/>
    <jointvel name="leg_left_ankle_roll_vel"      joint="leg_left_ankle_roll_joint"/>
    <jointvel name="leg_right_hip_roll_vel"       joint="leg_right_hip_roll_joint"/>
    <jointvel name="leg_right_hip_yaw_vel"        joint="leg_right_hip_yaw_joint"/>
    <jointvel name="leg_right_hip_pitch_vel"      joint="leg_right_hip_pitch_joint"/>
    <jointvel name="leg_right_knee_pitch_vel"     joint="leg_right_knee_pitch_joint"/>
    <jointvel name="leg_right_ankle_pitch_vel"    joint="leg_right_ankle_pitch_joint"/>
    <jointvel name="leg_right_ankle_roll_vel"     joint="leg_right_ankle_roll_joint"/>

    <jointactuatorfrc name="leg_left_hip_roll_torque"         joint="leg_left_hip_roll_joint"/>
    <jointactuatorfrc name="leg_left_hip_yaw_torque"          joint="leg_left_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_left_hip_pitch_torque"        joint="leg_left_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_left_knee_pitch_torque"       joint="leg_left_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_pitch_torque"      joint="leg_left_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_roll_torque"       joint="leg_left_ankle_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_roll_torque"        joint="leg_right_hip_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_yaw_torque"         joint="leg_right_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_right_hip_pitch_torque"       joint="leg_right_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_right_knee_pitch_torque"      joint="leg_right_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_pitch_torque"     joint="leg_right_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_roll_torque"      joint="leg_right_ankle_roll_joint"/>
                              
    <gyro name="imu_ang_vel" site="torso_center"/>
    <velocimeter name="imu_lin_vel" site="torso_center"/>
    <accelerometer name="imu_lin_acc" site="torso_center"/>
    <framezaxis name="imu_upvector" objtype="body" objname="world" reftype="site" refname="torso_center"/>
    <subtreeangmom name="root_angmom" body="torso"/>
    <gyro name="torso_gyro" site="torso_center"/>
    <accelerometer name="torso_accel" site="torso_center"/>
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framelinvel name="torso_linvel" objtype="body" objname="torso"/>
    <frameangvel name="torso_angvel" objtype="body" objname="torso"/>  
  </sensor>""")

    with open(mjcf_dir / f"{robot_name}.xml", "w") as file:
        file.write(content)