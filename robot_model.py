import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# Pinocchio-based kinematics and dynamics engine.
# From the simulator we only take measurements (base pose and twist, joint
# positions and velocities); everything else (feet, com, torso poses, jacobians,
# mass matrix, ...) is computed here by running pinocchio on those measurements.
class RobotModel:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.mass = sum(inertia.mass for inertia in self.model.inertias)

        # frames of interest
        self.frames = {name: self.model.getFrameId(name) for name in ['l_sole', 'r_sole', 'torso', 'body']}

        # name of the joint actuating each velocity index (base indices get 'root_joint')
        self.dof_names = [''] * self.nv
        for jid in range(1, self.model.njoints):
            for k in range(self.model.joints[jid].nv):
                self.dof_names[self.model.idx_vs[jid] + k] = self.model.names[jid]

        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.nv)

    # update the model from the measurements coming from the simulator.
    # base twist is given in the world frame (as an estimator would provide it);
    # pinocchio expects the free-flyer twist in the local base frame.
    def set_measurement(self, base_position, base_orientation, base_lin_velocity, base_ang_velocity, joint_positions, joint_velocities):
        R_wb = R.from_rotvec(base_orientation).as_matrix()

        self.q[0:3] = base_position
        self.q[3:7] = R.from_matrix(R_wb).as_quat() # xyzw
        self.q[7:]  = joint_positions

        self.v[0:3] = R_wb.T @ base_lin_velocity
        self.v[3:6] = R_wb.T @ base_ang_velocity
        self.v[6:]  = joint_velocities

        # run kinematics and dynamics algorithms once for all downstream getters
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, self.q, self.v)
        pin.crba(self.model, self.data, self.q)
        pin.nonLinearEffects(self.model, self.data, self.q, self.v)
        pin.centerOfMass(self.model, self.data, self.q, self.v)
        pin.jacobianCenterOfMass(self.model, self.data, self.q)
        pin.dccrba(self.model, self.data, self.q, self.v)

    # ----- kinematics (world frame) -----
    def get_com_position(self):
        return self.data.com[0].copy()

    def get_com_velocity(self):
        return self.data.vcom[0].copy()

    # pose as [rotation vector, position], matching the rest of the code
    def get_frame_pose(self, name):
        placement = self.data.oMf[self.frames[name]]
        return np.hstack((R.from_matrix(placement.rotation).as_rotvec(), placement.translation))

    def get_frame_orientation(self, name):
        return R.from_matrix(self.data.oMf[self.frames[name]].rotation).as_rotvec()

    # spatial velocity as [angular, linear] in the world frame
    def get_frame_spatial_velocity(self, name):
        velocity = pin.getFrameVelocity(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
        return np.hstack((velocity.angular, velocity.linear))

    def get_frame_angular_velocity(self, name):
        return pin.getFrameVelocity(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED).angular.copy()

    # ----- jacobians ([angular, linear] rows, world frame, to match dart) -----
    def get_frame_jacobian(self, name):
        J = pin.getFrameJacobian(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
        return np.vstack((J[3:6], J[0:3]))

    def get_angular_jacobian(self, name):
        return pin.getFrameJacobian(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)[3:6].copy()

    def get_com_jacobian(self):
        return self.data.Jcom.copy()

    def get_frame_jacobian_deriv(self, name):
        dJ = pin.getFrameJacobianTimeVariation(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
        return np.vstack((dJ[3:6], dJ[0:3]))

    def get_angular_jacobian_deriv(self, name):
        return pin.getFrameJacobianTimeVariation(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)[3:6].copy()

    def get_com_jacobian_deriv(self):
        return self.data.dAg[0:3, :] / self.mass

    # ----- dynamics -----
    def get_mass_matrix(self):
        M = self.data.M
        return np.triu(M) + np.triu(M, 1).T # crba only fills the upper triangle

    def get_coriolis_and_gravity_forces(self):
        return self.data.nle.copy()
