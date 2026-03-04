# NOTE: environment code is from https://github.com/kwonathan/franka-kitchen-pybullet

import pybullet as p
import time
import wm_lns.manipulator_experiments.config as config
from wm_lns.manipulator_experiments.kitchen_assets.load_franka_kitchen import loadFrankaKitchen, updateFrankaKitchen

class Environment:

    def __init__(self):
        self.value = None

    def load(self):
        p.resetDebugVisualizerCamera(cameraDistance=config.cameraDistance,
                                     cameraYaw=config.cameraYaw,
                                     cameraPitch=config.cameraPitch,
                                     cameraTargetPosition=config.cameraTargetPosition)
        kitchen, kettle = loadFrankaKitchen()
        self.value = kitchen
        self.kitchen, self.kettle = kitchen, kettle
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # If you want to add a disk under the goal position as a visual marker
        # disk = self.add_physical_goal_disk(config.goalPosition)
        # disk = self.add_physical_goal_disk(config.startPosition)

        print("Start Location: ", config.startPosition)
        print("Goal Location: ", config.goalPosition)

        # Run sim so things settle
        for _ in range(200):
            p.stepSimulation()
            time.sleep(config.control_dt)

        

    def update(self):
        updateFrankaKitchen(self.value)
        p.stepSimulation()
        time.sleep(config.control_dt)

    def add_physical_goal_disk(self, goal_pos, radius=0.04, thickness=0.02,
                            color=[0.5, 0.5, 0.5, 1.0], mass=0.2):
        x, y, z = goal_pos
        
        # Eh start a bit above so it can settle under gravity
        start_z = z + 0.2
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=thickness)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=thickness, rgbaColor=color)
        disk = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, start_z - thickness / 2.0],
            baseOrientation=[0, 0, 0, 1],
        )
        p.changeDynamics(disk, -1, lateralFriction=1.0, linearDamping=0.1, angularDamping=0.1)
        return disk

