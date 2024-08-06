import numpy as np
import cvxpy as cp
from algos.basic import Basic_Learning_Algo

class Homophily_Learning_Algo(Basic_Learning_Algo):

    def __init__(self, network, actions, theta1, theta2, beta, max_iters):
        super().__init__(network, actions, theta1, theta2, beta)
        self.n = len(network.nodes())
        self.k = len(actions[0])
        self.G = None
        self.B = None
        self.L = None
        self.max_iters = max_iters

    def calc_obj_function(self, L_t, G_t, B_t):
        I = np.eye(self.n)
        first_term = np.linalg.norm(((I - self.beta * G_t) @ self.actions - B_t), 'fro') ** 2
        second_term = self.theta1 * np.linalg.norm(G_t, 'fro') ** 2
        third_term = self.theta2 * np.trace(B_t.T @ L_t @ B_t)
        return first_term + second_term + third_term

    def calc_laplacian_from_G(self, G_t):
        return np.diag(np.sum(G_t, axis=1)) - G_t

    def create_opt_problem(self, B_t):
        I = np.eye(self.n)
        G = cp.Variable((self.n, self.n))
        self.G = G
        L = cp.Variable((self.n, self.n))
        self.L = L

        symmetry_constraints = [G == G.T]
        diagonal_constraints = [G[i, i] == 0 for i in range(self.n)]
        sparsity_constraint = [cp.norm1(G) <= self.n]
        laplacian_constraint = [L == cp.diag(cp.sum(G, axis=1))]

        constraints = symmetry_constraints + diagonal_constraints + sparsity_constraint + laplacian_constraint

        obj = cp.Minimize(
            cp.norm(((I - self.beta * G) @ self.actions - B_t) , 'fro') ** 2 
            + self.theta1 *  cp.norm(G, 'fro') ** 2
            + self.theta2 * cp.trace(B_t.T @ L @ B_t)
        )

        problem = cp.Problem(obj, constraints)

        return problem

    def run_algo(self):

        print("Running learning algo with homophilous marginal benefits")

        B_t = np.random.randn(self.n, self.k)
        obj_val = 0

        t = 1
        delta = 1
        while (delta > (10 ** -4)) or (t < self.max_iters):
            
            problem_t = self.create_opt_problem(B_t)
            problem_t.solve(solver=cp.ECOS, verbose=False)

            G_t = self.G.value
            L_t = self.L.value

            I = np.eye(self.n)
            B_t = np.linalg.inv(I + self.theta2 * L_t) @ (I - self.beta * G_t) @ self.actions
            self.B = B_t

            curr_obj_val = self.calc_obj_function(L_t, G_t, B_t)

            delta = np.abs(curr_obj_val - obj_val)
            obj_val = curr_obj_val

            t += 1

        opt_G = self.G.value
        opt_B = self.B

        # print(f"The learned graph is {opt_G}")
        # print(f"The learned marginal benefits are {opt_B}")

        return opt_G, opt_B
