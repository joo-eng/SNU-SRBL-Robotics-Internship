# Use viewer.launch() for macOS compatibility. Modify to viewer.launch_passive() if needed.
import mujoco
import mujoco.viewer
import time
model = mujoco.MjModel.from_xml_path("/Path/to/file/name_of_file.xml")
data = mujoco.MjData(model)
with mujoco.viewer.launch(model, data) as viewer:
    t = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
        t += model.opt.timestep