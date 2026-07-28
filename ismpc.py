import numpy as np
import scipy.sparse as sp
import piqp
import time

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    # parameters
    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.g = params['g']
    self.foot_size = params['foot_size']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function

    # lip model matrices (state per axis is [com, com_vel, zmp], control is the zmp velocity)
    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    # cost weights: sum_k ||u_k||^2 + w_zmp * sum_k ||zmp_k - zmp_mid_k||^2
    w_u, w_zmp = 1., 100.

    # the qp is a linear-quadratic optimal control problem, solved with the
    # structure-exploiting sparse solver in piqp. variables are stacked stage
    # wise: z = [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
    self.nx, self.nu, self.stage = 9, 3, 12
    N = self.N
    self.nz = self.stage * N + self.nx
    self.neq = self.nx * N + self.nx + 3 # dynamics + initial state + stability
    xi = lambda i: self.stage * i        # index of x_i in z
    ui = lambda i: self.stage * i + self.nx # index of u_i in z

    # discretized dynamics x_{i+1} = Ad x_i + Bd u_i + d
    Abar = np.zeros((9, 9))
    for a in range(3): Abar[3*a:3*a+3, 3*a:3*a+3] = self.A_lip
    Ad = np.eye(9) + self.delta * Abar
    Bd = np.zeros((9, 3))
    for a in range(3): Bd[3*a+2, a] = self.delta
    d = np.zeros(9); d[7] = - self.delta * self.g # gravity term on the vertical com

    # cost hessian (diagonal): w_zmp on the zmp states (stages 1..N), w_u on the controls
    Pdiag = np.zeros(self.nz)
    self.zmp_ids = [np.array([xi(i) + 3*a + 2 for i in range(1, N+1)]) for a in range(3)]
    for a in range(3): Pdiag[self.zmp_ids[a]] = 2. * w_zmp
    for i in range(N): Pdiag[ui(i):ui(i)+3] = 2. * w_u
    self.P = sp.diags(Pdiag).tocsc()
    self.w_zmp = w_zmp

    # equality constraints A z = b (built once, only b changes with the state)
    rows, cols, vals = [], [], []
    def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
    for i in range(N): # dynamics: x_{i+1} - Ad x_i - Bd u_i = d
      for k in range(9):
        r = 9*i + k
        add(r, xi(i+1) + k, 1.)
        for j in range(9):
          if Ad[k, j]: add(r, xi(i) + j, - Ad[k, j])
        for j in range(3):
          if Bd[k, j]: add(r, ui(i) + j, - Bd[k, j])
    for k in range(9): add(9*N + k, xi(0) + k, 1.) # initial state: x_0 = x0
    for a in range(3): # stability: x_1[axis] + eta (x_0 - zmp_0) == same at stage N
      r = 9*N + 9 + a
      add(r, xi(0) + 3*a+1,  1.); add(r, xi(0) + 3*a,  self.eta); add(r, xi(0) + 3*a+2, - self.eta)
      add(r, xi(N) + 3*a+1, -1.); add(r, xi(N) + 3*a, - self.eta); add(r, xi(N) + 3*a+2,   self.eta)
    self.A = sp.csc_matrix((vals, (rows, cols)), shape=(self.neq, self.nz))
    self.init_slice = slice(9*N, 9*N + 9) # rows of b holding the initial state

    self.b = np.zeros(self.neq)
    for i in range(N): self.b[9*i:9*i+9] = d
    self.c = np.zeros(self.nz)
    self.x_l = np.full(self.nz, -np.inf) # only the zmp states are bounded (moving zmp box)
    self.x_u = np.full(self.nz,  np.inf)

    # set up the solver with the initial values (updated in place at each solve)
    self.xi, self.ui = xi, ui
    self.update_qp(np.zeros(9), *self.generate_moving_constraint(0))
    self.solver = piqp.SparseSolver()
    # the generic sparse ldlt backend benchmarks faster than sparse_multistage
    # here (small per-stage blocks, and the stability term couples stage 0 to N)
    self.solver.settings.kkt_solver = piqp.KKTSolver.sparse_ldlt
    self.solver.settings.eps_abs = 1e-6
    self.solver.settings.verbose = False
    self.solver.setup(self.P, self.c, self.A, self.b, None, None, None, self.x_l, self.x_u)

    # state
    self.x = np.zeros(9)
    self.solve_time = 0. # time spent in the last qp solve
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}

  def update_qp(self, x0, mc_x, mc_y, mc_z):
    mc = (mc_x, mc_y, mc_z)
    self.b[self.init_slice] = x0
    for a in range(3):
      self.c[self.zmp_ids[a]]   = - 2. * self.w_zmp * mc[a]
      self.x_l[self.zmp_ids[a]] = mc[a] - self.foot_size / 2.
      self.x_u[self.zmp_ids[a]] = mc[a] + self.foot_size / 2.

  def solve(self, current, t):
    x0 = np.array([current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
                   current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1],
                   current['com']['pos'][2], current['com']['vel'][2], current['zmp']['pos'][2]])

    self.update_qp(x0, *self.generate_moving_constraint(t))

    t_solve = time.perf_counter()
    self.solver.update(c=self.c, b=self.b, x_l=self.x_l, x_u=self.x_u)
    self.solver.solve()
    self.solve_time = time.perf_counter() - t_solve

    z = self.solver.result.x
    self.x = z[self.xi(1):self.xi(1) + 9] # state at the first stage
    self.u = z[self.ui(0):self.ui(0) + 3] # control at the first stage

    # create output LIP state
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.x[6]])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], self.x[7]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
    self.lip_state['zmp']['vel'] = self.u
    self.lip_state['com']['acc'] = self.eta**2 * (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']) + np.hstack([0, 0, - self.g])

    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact

  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][0] + self.initial['rfoot']['pos'][0]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][1] + self.initial['rfoot']['pos'][1]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y, np.zeros(self.N)
