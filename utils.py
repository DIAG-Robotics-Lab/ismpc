import proxsuite
from scipy.spatial.transform import Rotation as R
import numpy as np

def rotation_vector_difference(rotvec_a, rotvec_b):
    R_a = R.from_rotvec(rotvec_a)
    R_b = R.from_rotvec(rotvec_b)
    R_diff = R_b.inv() * R_a
    return R_diff.as_rotvec()

def pose_difference(pose_a, pose_b):
    pos_diff = pose_a[:3] - pose_b[:3]
    rot_diff = rotation_vector_difference(pose_a[3:], pose_b[3:])
    return np.hstack((pos_diff, rot_diff))

# converts a rotation matrix to a rotation vector
def get_rotvec(rot_matrix):
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_rotvec()

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