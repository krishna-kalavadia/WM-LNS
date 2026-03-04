# NOTE: environment code is from https://github.com/kwonathan/franka-kitchen-pybullet

import math

# Simulation
control_dt = 1. / 240.

# Franka kitchen environment
baseStartPosition = [0.0, 0.0, 1.25]
baseStartOrientationE = [0.0, 0.0, 0.0]
jointStartPositions = [0.0, -0.3, -0.35, -2.25, 0.0, 1.8675, 0.0, 0.04, 0.04]

kitchenStartPosition = [0.6, 0.1, 0.0]
kettleStartPosition = [0.625, -0.15, 1.25]
# kettleStartPosition = [0.3, -0.55, 1.25]
kitchenStartOrientationE = [0.0, 0.0, -math.pi / 2]
kitchenGlobalScaling = 0.75

microwaveStartPosition = [0.58125, 0.6625, 1.2]

startPosition = [-0.1, 0.7, 1.3]
goalPosition = [0.3, -0.55, 1.3]

# Camera
cameraDistance = 1.3
cameraYaw = -60.0
cameraPitch = -30.0
cameraTargetPosition = [0.5, 0.0, 1.5]
