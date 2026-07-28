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

    # ----- poses and velocities ([angular, linear] rows, world frame) -----
    # name is a frame name or 'com'; part optionally selects 'ang' or 'pos'
    # pose is [rotation vector, position], velocity is [angular, linear]
    def get_pose(self, name, part=None):
        if name == 'com':
            x = self.data.com[0] # a point only has a position (pos)
        else:
            placement = self.data.oMf[self.frames[name]]
            x = np.hstack((R.from_matrix(placement.rotation).as_rotvec(), placement.translation))
            if   part == 'ang': x = x[0:3]
            elif part == 'pos': x = x[3:6]
        return x.copy()

    def get_velocity(self, name, part=None):
        if name == 'com':
            x = self.data.vcom[0]
        else:
            velocity = pin.getFrameVelocity(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
            x = np.hstack((velocity.angular, velocity.linear))
            if   part == 'ang': x = x[0:3]
            elif part == 'pos': x = x[3:6]
        return x.copy()

    # ----- jacobians ([angular, linear] rows, world frame, to match dart) -----
    # name is a frame name or 'com'; part optionally selects 'ang' or 'pos' rows
    def get_jacobian(self, name, part=None):
        if name == 'com':
            J = self.data.Jcom # the com only has a linear (pos) jacobian
        else:
            # pinocchio returns [linear, angular]; reorder to [angular, linear] to match dart
            J = pin.getFrameJacobian(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
            J = np.vstack((J[3:6], J[0:3]))
            if   part == 'ang': J = J[0:3]
            elif part == 'pos': J = J[3:6]
        return J.copy()

    def get_jacobian_deriv(self, name, part=None):
        if name == 'com':
            J = self.data.dAg[0:3, :] / self.mass
        else:
            J = pin.getFrameJacobianTimeVariation(self.model, self.data, self.frames[name], pin.LOCAL_WORLD_ALIGNED)
            J = np.vstack((J[3:6], J[0:3]))
            if   part == 'ang': J = J[0:3]
            elif part == 'pos': J = J[3:6]
        return J.copy()

    # ----- dynamics -----
    def get_mass_matrix(self):
        M = self.data.M
        return np.triu(M) + np.triu(M, 1).T # crba only fills the upper triangle

    def get_coriolis_and_gravity_forces(self):
        return self.data.nle.copy()
