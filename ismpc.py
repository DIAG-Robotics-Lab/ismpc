import numpy as np
import casadi as cs
import time
import matplotlib.pyplot as plt

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.N = params['N']
    N = self.N
    self.delta = params['world_time_step']
    delta = self.delta
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.step_height = params['step_height']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.footstep_plan = self.footstep_planner.footstep_plan
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    # lip model matrices
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # dynamics
    self.f = lambda x, u: cs.vertcat(
      self.A_lip @ x[:3] + self.B_lip @ u[0],
      self.A_lip @ x[3:] + self.B_lip @ u[1]
    )

    # optimization problem
    self.opt = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}
    self.opt.solver("osqp", p_opts, s_opts)

    self.U = self.opt.variable(2, N)
    self.X = self.opt.variable(6, N + 1)

    self.x0_param = self.opt.parameter(6)
    self.zmp_x_mid_param = self.opt.parameter(N)
    self.zmp_y_mid_param = self.opt.parameter(N)

    for i in range(N):
      self.opt.subject_to(self.X[:, i + 1] == self.X[:, i] + delta * self.f(self.X[:, i], self.U[:, i]))

    cost = cs.sumsqr(self.U[0, :]) + cs.sumsqr(self.U[1, :]) + \
           100 * cs.sumsqr(self.X[2, 1:].T - self.zmp_x_mid_param) + \
           100 * cs.sumsqr(self.X[5, 1:].T - self.zmp_y_mid_param)

    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + self.foot_size / 2.)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - self.foot_size / 2.)

    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # stability constraint with periodic tail
    self.opt.subject_to(self.X[1, 0] + self.eta**3 * (self.X[0, 0] - self.X[2, 0]) == \
                        self.X[1, N] + self.eta**3 * (self.X[0, N] - self.X[2, N]))
    self.opt.subject_to(self.X[4, 0] + self.eta**3 * (self.X[3, 0] - self.X[5, 0]) == \
                        self.X[4, N] + self.eta**3 * (self.X[3, N] - self.X[5, N]))

    self.opt.minimize(cost)

    self.x = np.zeros(6)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def solve(self, current, t):
    self.x = np.array([current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
                       current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1]])
    
    mc_x, mc_y = self.generate_moving_constraint(t)

    # solve optimization problem
    self.opt.set_value(self.x0_param, self.x)
    self.opt.set_value(self.zmp_x_mid_param, mc_x)
    self.opt.set_value(self.zmp_y_mid_param, mc_y)

    sol = self.opt.solve()
    self.x = sol.value(self.X[:,1])
    self.u = sol.value(self.U[:,0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.h])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], 0.])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], 0.])
    self.lip_state['zmp']['vel'] = np.hstack((self.u, 0.))
    self.lip_state['com']['acc'] = np.hstack((self.eta**2 * (self.lip_state['com']['pos'][:2] - self.lip_state['zmp']['pos'][:2]), 0.))

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact += self.footstep_planner.footstep_plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact

  def generate_moving_constraint_at_time(self, time):
    step_index = self.footstep_planner.get_step_index_at_time(time)
    time_in_step = time - self.footstep_planner.get_start_time(step_index)
    phase = self.footstep_planner.get_phase_at_time(time)
    single_support_duration = self.footstep_plan[step_index]['ss_duration']
    double_support_duration = self.footstep_plan[step_index]['ds_duration']

    if phase == 'ss':
      return self.footstep_plan[step_index]['pos']

    # linear interpolation for x and y coordinates of the foot positions during double support
    if step_index == 0: start_pos = (self.initial['lsole']['pos'][3:] + self.initial['rsole']['pos'][3:]) / 2.
    else:               start_pos = np.array(self.footstep_plan[step_index]['pos'])
    target_pos = np.array(self.footstep_plan[step_index + 1]['pos'])
    
    moving_constraint = start_pos + (target_pos - start_pos) * ((time_in_step - single_support_duration) / double_support_duration)
    return moving_constraint
  
  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lsole']['pos'][3] + self.initial['rsole']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lsole']['pos'][4] + self.initial['rsole']['pos'][4]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_plan[j]['ds_duration']
      fs_current_pos = self.footstep_plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y