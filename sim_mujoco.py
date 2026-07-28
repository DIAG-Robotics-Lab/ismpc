import numpy as np
import mujoco
import mujoco.viewer
import copy
from scipy.spatial.transform import Rotation as R
from utils import *
import os
import ismpc
import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from robot_model import RobotModel
from logger import Logger

# the mujoco model (mujoco/scene.xml + hrp4.xml + assets) is generated from the
# urdf by convert_to_mjcf.py; re-run that tool if the urdf changes.

class Hrp4Controller:
    def __init__(self):
        self.time = 0
        self.zmp = np.zeros(3) # last valid zmp measurement, held while airborne
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'ss_duration': 70,
            'ds_duration': 30,
            'world_time_step': 0.01,
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 100,
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

        # pinocchio model: computes all kinematics/dynamics terms from the measurements
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.robot_model = RobotModel(os.path.join(current_dir, "urdf", "hrp4.urdf"))
        self.params['dof'] = self.robot_model.nv

        # mujoco simulator model (actuators ordered like the pinocchio joints)
        self.joint_order = self.robot_model.dof_names[6:]
        self.model = mujoco.MjModel.from_xml_path(os.path.join(current_dir, "mujoco", "scene.xml"))
        self.data = mujoco.MjData(self.model)

        # addresses of the measured joints in mujoco order -> pinocchio order
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_order]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'body')

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        self.data.qpos[3:7] = [1., 0., 0., 0.] # base orientation (quaternion, wxyz)
        for joint_name, value in initial_configuration.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = value * np.pi / 180.

        # position the robot on the ground
        mujoco.mj_forward(self.model, self.data)
        self.update_robot_model()
        lsole_pos = self.robot_model.get_pose('l_sole', 'pos')
        rsole_pos = self.robot_model.get_pose('r_sole', 'pos')
        self.data.qpos[0] = - (lsole_pos[0] + rsole_pos[0]) / 2.
        self.data.qpos[1] = - (lsole_pos[1] + rsole_pos[1]) / 2.
        self.data.qpos[2] = - (lsole_pos[2] + rsole_pos[2]) / 2.
        mujoco.mj_forward(self.model, self.data)

        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]

        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.robot_model, redundant_dofs)

        # initialize footstep planner
        reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10
        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )

        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial,
            self.footstep_planner,
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial,
            self.footstep_planner,
            self.params
            )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        d = np.zeros(9)
        d[7] = - self.params['world_time_step'] * self.params['g']
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R_kf = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1], \
                      self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
        self.kf = filter.KalmanFilter(block_diag(A, A, A), \
                                      block_diag(B, B, B), \
                                      d, \
                                      block_diag(H, H, H), \
                                      block_diag(Q, Q, Q), \
                                      block_diag(R_kf, R_kf, R_kf), \
                                      block_diag(P, P, P), \
                                      x)

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)

    def control(self):
        # create current and desired states
        self.current = self.retrieve_state()

        # update kalman filter
        u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        self.kf.predict(u)
        x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
                                            self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
                                            self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))

        # update current state using kalman filter output
        self.current['com']['pos'][0] = x_flt[0]
        self.current['com']['vel'][0] = x_flt[1]
        self.current['zmp']['pos'][0] = x_flt[2]
        self.current['com']['pos'][1] = x_flt[3]
        self.current['com']['vel'][1] = x_flt[4]
        self.current['zmp']['pos'][1] = x_flt[5]
        self.current['com']['pos'][2] = x_flt[6]
        self.current['com']['vel'][2] = x_flt[7]
        self.current['zmp']['pos'][2] = x_flt[8]

        # get references using mpc
        lip_state, contact = self.mpc.solve(self.current, self.time)

        self.desired['com']['pos'] = lip_state['com']['pos']
        self.desired['com']['vel'] = lip_state['com']['vel']
        self.desired['com']['acc'] = lip_state['com']['acc']
        self.desired['zmp']['pos'] = lip_state['zmp']['pos']
        self.desired['zmp']['vel'] = lip_state['zmp']['vel']

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # set torso and base orientation references to the average of the feet
        # (feet vectors are [linear, angular]; the orientation part is [3:6])
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][3:6] + self.desired['rfoot'][key][3:6]) / 2.

        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact)

        # set torque commands (actuators are ordered like the pinocchio joints)
        self.data.ctrl[:] = commands

        # log and plot
        self.logger.log_data(self.current, self.desired)
        #self.logger.update_plot(self.time)

        self.time += 1

    def update_robot_model(self):
        # measurements taken from the simulator
        d = self.data
        base_position    = d.qpos[0:3]
        base_orientation = R.from_quat(d.qpos[[4, 5, 6, 3]]).as_rotvec() # mujoco quaternion is wxyz
        R_wb = R.from_rotvec(base_orientation).as_matrix()

        # mujoco free joint: linear velocity is in the world frame, angular in the local frame
        base_lin_velocity = d.qvel[0:3]
        base_ang_velocity = R_wb @ d.qvel[3:6]

        # everything else is computed by pinocchio from these measurements
        self.robot_model.set_measurement(
            base_position     = base_position,
            base_orientation  = base_orientation,
            base_lin_velocity = base_lin_velocity,
            base_ang_velocity = base_ang_velocity,
            joint_positions   = d.qpos[self.qadr],
            joint_velocities  = d.qvel[self.vadr])

    def measure_zmp(self):
        # contact forces are a measurement (force sensors at the feet); the zmp
        # is derived from them together with the com height
        m, d = self.model, self.data

        # ground reaction forces on the robot, in the world frame
        force = np.zeros(3)
        grfs = []
        f_contact = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f_contact)
            # mj_contactForce is expressed in the contact frame (rows are the axes
            # in world) and acts on the robot (geom2); rotate it to the world frame
            grf = contact.frame.reshape(3, 3).T @ f_contact[0:3]
            grfs.append((contact.pos.copy(), grf))
            force += grf

        if force[2] <= 0.1: # threshold for when we lose contact
            return self.zmp.copy() # hold the last valid measurement

        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = self.robot_model.get_pose('com')[2] - force[2] / (self.robot_model.mass * self.params['g'] / self.params['h'])
        for point, grf in grfs:
            if grf[2] <= 0.1: continue
            zmp[0] += (point[0] * grf[2] / force[2] + (zmp[2] - point[2]) * grf[0] / force[2])
            zmp[1] += (point[1] * grf[2] / force[2] + (zmp[2] - point[2]) * grf[1] / force[2])

        # remember it so we can hold it if we lose contact
        self.zmp = zmp.copy()
        return zmp

    def retrieve_state(self):
        # update the pinocchio model with the current measurements
        self.update_robot_model()

        # base pose and velocity are measurements (absolute localization estimator)
        base_orientation = R.from_quat(self.data.qpos[[4, 5, 6, 3]]).as_rotvec()
        base_angular_velocity = R.from_rotvec(base_orientation).as_matrix() @ self.data.qvel[3:6]

        # com and torso pose (orientation and position) from pinocchio
        com_position = self.robot_model.get_pose('com')
        torso_orientation = self.robot_model.get_pose('torso', 'ang')

        # feet poses (orientation and position) from pinocchio
        left_foot_pose  = self.robot_model.get_pose('l_sole')
        right_foot_pose = self.robot_model.get_pose('r_sole')

        # velocities from pinocchio
        com_velocity = self.robot_model.get_velocity('com')
        torso_angular_velocity = self.robot_model.get_velocity('torso', 'ang')
        l_foot_spatial_velocity = self.robot_model.get_velocity('l_sole')
        r_foot_spatial_velocity = self.robot_model.get_velocity('r_sole')

        # zmp measured from the contact forces
        zmp = self.measure_zmp()

        # generalized position: [base position, base orientation (rotvec), joint positions]
        joint_position = np.concatenate((self.data.qpos[0:3], base_orientation, self.data.qpos[self.qadr]))

        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': joint_position,
                      'vel': self.robot_model.v.copy(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }

if __name__ == "__main__":
    node = Hrp4Controller()

    with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
        viewer.cam.lookat = [1., 0., 0.5]
        viewer.cam.distance = 4.
        while viewer.is_running():
            node.control()
            mujoco.mj_step(node.model, node.data)
            viewer.sync()
