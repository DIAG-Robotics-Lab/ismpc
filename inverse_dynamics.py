import numpy as np
from utils import *

class InverseDynamics:
    def __init__(self, robot, redundant_dofs, foot_size=0.1, µ=0.5):
        self.robot = robot
        self.dofs = self.robot.nv
        self.d = foot_size / 2.
        self.µ = µ

        # define sizes for QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims

        self.n_eq_constraints = self.dofs
        self.n_ineq_constraints = 8 * self.num_contacts

        # initialize QP solver
        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            if self.robot.dof_names[i] in redundant_dofs:
                self.joint_selection[i, i] = 1

    def get_joint_torques(self, desired, current, contact):
        contact_l = contact == 'lfoot'  or contact == 'ds'
        contact_r = contact == 'rfoot' or contact == 'ds'

        # weights and gains
        tasks = ['lfoot', 'rfoot', 'com', 'torso', 'base', 'joints']
        weights   = {'lfoot':  1., 'rfoot':  1., 'com':  1., 'torso': 1., 'base': 1., 'joints': 1.e-2}
        pos_gains = {'lfoot': 10., 'rfoot': 10., 'com':  5., 'torso': 1., 'base': 1., 'joints': 10.  }
        vel_gains = {'lfoot': 10., 'rfoot': 10., 'com': 10., 'torso': 2., 'base': 2., 'joints': 1.   }

        # jacobians
        J = {'lfoot' : self.robot.get_jacobian('l_sole'),
             'rfoot' : self.robot.get_jacobian('r_sole'),
             'com'   : self.robot.get_jacobian('com'),
             'torso' : self.robot.get_jacobian('torso', 'ang'),
             'base'  : self.robot.get_jacobian('body', 'ang'),
             'joints': self.joint_selection}

        # jacobians derivatives
        Jdot = {'lfoot' : self.robot.get_jacobian_deriv('l_sole'),
                'rfoot' : self.robot.get_jacobian_deriv('r_sole'),
                'com'   : self.robot.get_jacobian_deriv('com'),
                'torso' : self.robot.get_jacobian_deriv('torso', 'ang'),
                'base'  : self.robot.get_jacobian_deriv('body', 'ang'),
                'joints': np.zeros((self.dofs, self.dofs))}

        # feedforward terms
        ff = {'lfoot' : desired['lfoot']['acc'],
              'rfoot' : desired['rfoot']['acc'],
              'com'   : desired['com']['acc'],
              'torso' : desired['torso']['acc'],
              'base'  : desired['base']['acc'],
              'joints': desired['joint']['acc']}

        # error vectors
        pos_error = {'lfoot' : pose_difference(desired['lfoot']['pos'] , current['lfoot']['pos'] ),
                     'rfoot' : pose_difference(desired['rfoot']['pos'], current['rfoot']['pos']),
                     'com'   : desired['com']['pos'] - current['com']['pos'],
                     'torso' : rotation_vector_difference(desired['torso']['pos'], current['torso']['pos']),
                     'base'  : rotation_vector_difference(desired['base']['pos'] , current['base']['pos'] ),
                     'joints': desired['joint']['pos'] - current['joint']['pos']}

        # velocity error vectors
        vel_error = {'lfoot' : desired['lfoot']['vel'] - current['lfoot']['vel'],
                     'rfoot' : desired['rfoot']['vel'] - current['rfoot']['vel'],
                     'com'   : desired['com']['vel']   - current['com']['vel'],
                     'torso' : desired['torso']['vel'] - current['torso']['vel'],
                     'base'  : desired['base']['vel']  - current['base']['vel'],
                     'joints': desired['joint']['vel'] - current['joint']['vel']}

        # cost function
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            H_task =   weights[task] * J[task].T @ J[task]
            F_task = - weights[task] * J[task].T @ (ff[task]
                                                    + vel_gains[task] * vel_error[task]
                                                    + pos_gains[task] * pos_error[task]
                                                    - Jdot[task] @ current['joint']['vel'])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # regularization term for contact forces
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * 1e-6

        # dynamics constraints: M * q_ddot + C - J_c^T * f_c = tau
        inertia_matrix = self.robot.get_mass_matrix()
        actuation_matrix = block_diag(np.zeros((6, 6)), np.eye(self.dofs - 6))
        contact_jacobian = np.vstack((contact_l * J['lfoot'], contact_r * J['rfoot']))
        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - contact_jacobian.T))
        b_eq = - self.robot.get_coriolis_and_gravity_forces()

        # inequality constraints
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        # contact wrench is [force, moment]: cop bounds on the moment, friction on the force
        A = np.array([[0, 0, -self.d,  1,  0, 0],
                      [0, 0, -self.d, -1,  0, 0],
                      [0, 0, -self.d,  0,  1, 0],
                      [0, 0, -self.d,  0, -1, 0],
                      [ 1,  0, -self.µ, 0, 0, 0],
                      [-1,  0, -self.µ, 0, 0, 0],
                      [ 0,  1, -self.µ, 0, 0, 0],
                      [ 0, -1, -self.µ, 0, 0, 0]])
        A_ineq[0:self.n_ineq_constraints, f_c_indices] = block_diag(A, A)

        # solve the QP, compute torques and return them
        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        tau = solution[tau_indices]
        return tau[6:]