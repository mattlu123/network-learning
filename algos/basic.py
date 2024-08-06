import numpy as np
import cvxpy as cp

class Basic_Learning_Algo():

    def __init__(self, network, actions, theta1, theta2, beta):
        self.network = network
        self.actions = actions
        self.theta1 = theta1
        self.theta2 = theta2
        self.beta = beta
        self.n = len(network.nodes())
        self.k = len(actions[0])
        self.G = None
        self.B = None


    def calc_frobenius_norm(self, matrix):
        res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res += np.abs(matrix[i][j]) ** 2

        return np.sqrt(res)


    def calc_frobenius_norm_v2(self, matrix):
        conj_transpose = np.conjugate(matrix.T)
        return np.sqrt(np.trace(matrix * conj_transpose))
    
    def calc_individual_lin_quad_payoffs(self, model, i, alpha, a, beta):

        individual_effect = alpha[i] * a[i] - 0.5 * a[i]
        network_effect = 0
        for j in range(1, len(model) + 1):
            network_effect += beta * a[i] * model[i][j]

        res = individual_effect + network_effect

        return res


    def create_opt_problem(self):
        I = np.eye(self.n)
        G = cp.Variable((self.n, self.n), pos=True)
        self.G = G
        B = cp.Variable((self.n, self.k))
        self.B = B

        symmetry_constraints = [G == G.T]
        diagonal_constraints = [G[i, i] == 0 for i in range(self.n)]
        sparsity_constraint = [cp.norm1(G) <= self.n]

        constraints = symmetry_constraints + diagonal_constraints + sparsity_constraint

        obj = cp.Minimize(
            cp.norm(((I - self.beta * G) @ self.actions - B) , 'fro') ** 2 
            + self.theta1 *  cp.norm(G, 'fro') ** 2
            + self.theta2 * cp.norm(B, 'fro') ** 2
        )

        problem = cp.Problem(obj, constraints)

        return problem


    def run_algo(self):

        print("Running basic learning algo")

        problem = self.create_opt_problem()
        
        problem.solve(solver=cp.ECOS)

        opt_G = self.G.value
        opt_B = self.B.value

        # print(f"The learned graph is {opt_G}")
        # print(f"The learned marginal benefits are {opt_B}")

        return opt_G, opt_B
