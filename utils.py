import proxsuite
import numpy as np
import math

# rotation vector of the relative rotation R_b^-1 * R_a, computed directly with
# quaternions (much faster than a general rotation library in the control loop)
def rotation_vector_difference(rotvec_a, rotvec_b):
    def to_quat(v):
        theta = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        s = 0.5 if theta < 1e-8 else math.sin(0.5 * theta) / theta
        return math.cos(0.5 * theta), s * v[0], s * v[1], s * v[2]

    aw, ax, ay, az = to_quat(rotvec_a)
    bw, bx, by, bz = to_quat(rotvec_b)

    # quaternion product conj(b) * a
    w = bw * aw + bx * ax + by * ay + bz * az
    x = bw * ax - bx * aw - by * az + bz * ay
    y = bw * ay + bx * az - by * aw - bz * ax
    z = bw * az - bx * ay + by * ax - bz * aw

    if w < 0.: w, x, y, z = -w, -x, -y, -z # take the shortest rotation
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-8: return np.array([2. * x, 2. * y, 2. * z])
    k = 2. * math.atan2(n, w) / n
    return np.array([k * x, k * y, k * z])

def pose_difference(pose_a, pose_b):
    pos_diff = pose_a[:3] - pose_b[:3]
    rot_diff = rotation_vector_difference(pose_a[3:], pose_b[3:])
    return np.hstack((pos_diff, rot_diff))

def block_diag(*arrays):
    arrays = [np.atleast_2d(a) if np.isscalar(a) else np.atleast_2d(a) for a in arrays]

    rows = sum(arr.shape[0] for arr in arrays)
    cols = sum(arr.shape[1] for arr in arrays)
    block_matrix = np.zeros((rows, cols), dtype=arrays[0].dtype)

    current_row = 0
    current_col = 0

    for arr in arrays:
        r, c = arr.shape
        block_matrix[current_row:current_row + r, current_col:current_col + c] = arr
        current_row += r
        current_col += c

    return block_matrix

# solves a constrained QP with proxqp (proxsuite):
#   min 0.5 x' H x + F' x   s.t.  A_eq x == b_eq,  A_ineq x <= b_ineq
class QPSolver:
    def __init__(self, n_vars, n_eq_constraints=0, n_ineq_constraints=0):
        self.n_vars = n_vars
        self.n_eq_constraints = n_eq_constraints
        self.n_ineq_constraints = n_ineq_constraints

        self.qp = proxsuite.proxqp.dense.QP(n_vars, n_eq_constraints, n_ineq_constraints)
        self.qp.settings.eps_abs = 1e-6
        self.qp.settings.verbose = False
        # warm start each solve from the previous solution (fast in a control loop)
        self.qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT

        # proxqp uses l <= A_ineq x <= u; we only have the upper bound
        self.lower = - 1e20 * np.ones(n_ineq_constraints) if n_ineq_constraints > 0 else None
        self.initialized = False

    def set_values(self, H, F, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        A = A_eq   if self.n_eq_constraints   > 0 else None
        b = b_eq   if self.n_eq_constraints   > 0 else None
        C = A_ineq if self.n_ineq_constraints > 0 else None
        u = b_ineq if self.n_ineq_constraints > 0 else None
        # first call sets up the workspace, subsequent calls just update the values
        if not self.initialized:
            self.qp.init(H, F, A, b, C, self.lower, u)
            self.initialized = True
        else:
            self.qp.update(H=H, g=F, A=A, b=b, C=C, l=self.lower, u=u)

    def solve(self):
        self.qp.solve()
        return np.array(self.qp.results.x).flatten()