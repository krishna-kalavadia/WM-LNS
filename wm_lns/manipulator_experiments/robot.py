# NOTE: environment code is from https://github.com/kwonathan/franka-kitchen-pybullet

import pybullet as p
import wm_lns.manipulator_experiments.config as config

class Robot:
    def __init__(self):
        self.baseStartPosition     = config.baseStartPosition
        self.baseStartOrientationQ = p.getQuaternionFromEuler(config.baseStartOrientationE)
        self.jointStartPositions   = config.jointStartPositions

        # load the Panda URDF with a fixed base
        self.id = p.loadURDF(
            "franka_panda/panda.urdf",
            self.baseStartPosition,
            self.baseStartOrientationQ,
            useFixedBase=True,
            flags=(
                p.URDF_USE_SELF_COLLISION
                | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            ),
        )

        # Rest joints
        i = 0
        for j in range(p.getNumJoints(self.id)):
            jointType = p.getJointInfo(self.id, j)[2]
            if jointType in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                p.resetJointState(self.id, j, self.jointStartPositions[i])
                i += 1

        # Add cylinder to support Panda arm       
        base_pos, _ = p.getBasePositionAndOrientation(self.id)
        z_base = base_pos[2]
        support_radius = 0.12  
        shift_back     = 0.04  
        support_height   = z_base - 0.05
        support_center_z = support_height / 2.0
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=support_radius,
            height=support_height
        )
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=support_radius,
            length=support_height,
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )

        cyl_x = base_pos[0] - shift_back
        cyl_y = base_pos[1]
        cyl_z = support_center_z

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[cyl_x, cyl_y, cyl_z],
            baseOrientation=[0, 0, 0, 1]
        )
