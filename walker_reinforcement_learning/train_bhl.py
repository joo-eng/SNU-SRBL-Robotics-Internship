import bhl_register
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()

"""
Personal notes for checking RL progress:
Climbing = good:
Mean reward 35.3 49.97 yes
Mean episode length 777.32 903 check
Episode_Reward/track_linear_velocity 0.405 Variance too high to check
Episode_Reward/upright 0.9529 0.80 Variance too high to check

Dropping = good:
Episode_Termination/fell_over 0.5 Variance too high to check
Metrics/twist/error_vel_xy and error_vel_yaw 0.9984 1.8839 0.6 1.6 check
Mean action std 0.73 0.56 check
"""