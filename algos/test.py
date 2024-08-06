import numpy as np
import cvxpy as cp
import copy
from algos.basic import Basic_Learning_Algo

class Test(Basic_Learning_Algo):
    
    def __init__(self, network, actions, theta, beta, T, k, max_iters):
        super().__init__(network, actions, theta, None, beta)
        self.n = len(network.nodes())
        self.T = T
        self.k = k
        self.G = None
        self.A = None
        self.Gamma = None
        self.max_iters = max_iters

    def generate_joint_distribution_matrix(self, m):
        values = np.random.rand(m)
        values /= np.sum(values)
        joint_distribution_matrix = np.outer(values, values)
        
        return joint_distribution_matrix

    def generate_information_matrix(self, P, m):
        Gamma = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Gamma[i, j] = P[i, j] / np.sum(P[:, i])

        return Gamma
    
    def convert_to_simple_graph(self, threshold, G):
        G[G < threshold] = 0
        G[G >= threshold] = 1

        return G

    def initialize_variables(self):
        A_0 = np.random.normal(loc=0, scale=1, size=(self.T, self.k))
        joint_distribution_matrix = self.generate_joint_distribution_matrix(self.T)
        Gamma_0 = self.generate_information_matrix(joint_distribution_matrix, self.T)

        return A_0, Gamma_0
    
    def create_opt_problem_G(self, X, Gamma_k, A_k):

        #variables
        G = cp.Variable((self.n, self.n), nonneg=True) # nxn
        self.G = G

        # constraints
        symmetry_constraint = [G == G.T]
        diagonal_constraint = [G[i, i] == 0 for i in range(self.n)]
        graph_constraint = [G @ np.ones((self.n, 1)) == np.ones((self.n, 1))]

        constraints = symmetry_constraint + diagonal_constraint + graph_constraint
        
        # obj function
        expr_1 = cp.norm((np.eye(self.T * self.n) - cp.kron(self.beta * Gamma_k, G)) @ X - np.kron(A_k, np.ones(self.n).reshape(1, -1).T), 'fro') ** 2
        expr_2 = self.theta1 *  cp.norm(G, 'fro') ** 2
        obj = cp.Minimize(expr_1 + expr_2)

        problem = cp.Problem(obj, constraints)
        # print("G curvature: ", expr_1.curvature, expr_2.curvature)

        return problem
    
    def create_opt_problem_A(self, X, G_k, Gamma_k):

        # variables
        A = cp.Variable((self.T, self.k))
        self.A = A

        # obj function
        expr_1 = cp.norm((np.eye(self.T * self.n) - np.kron(self.beta * Gamma_k, G_k)) @ X - cp.kron(A, np.ones(self.n).reshape(1, -1).T), 'fro') ** 2
        expr_2 = self.theta1 *  cp.norm(G_k, 'fro') ** 2
        obj = cp.Minimize(expr_1 + expr_2)

        problem = cp.Problem(obj, [])
        #print("b curvature: ", expr_1.curvature, expr_2.curvature)

        return problem

    def create_opt_problem_Gamma(self, X, G_k, A_k):
        
        # variables
        Gamma = cp.Variable((self.T, self.T), nonneg=True)
        self.Gamma = Gamma

        # constraints        
        gamma_constraint = [np.ones((self.T, 1)).T @ Gamma == np.ones((self.T, 1))]

        # obj function
        expr_1 = cp.norm((np.eye(self.T * self.n) - cp.kron(self.beta * Gamma, G_k)) @ X - np.kron(A_k, np.ones(self.n).reshape(1,-1).T), 'fro') ** 2
        expr_2 = self.theta1 *  cp.norm(G_k, 'fro') ** 2
        obj = cp.Minimize(expr_1 + expr_2)

        problem = cp.Problem(obj, gamma_constraint)
        #print("Gamma curvature: ", expr_1.curvature, expr_2.curvature)

        return problem
    
    def calc_objective_function(self, X, G, Gamma, A):
        first_term = np.linalg.norm((np.eye(self.T * self.n) - np.kron(self.beta * Gamma, G)) @ X - np.kron(A, np.ones(self.n).reshape(1, -1).T), 'fro') ** 2
        second_term = self.theta1 * np.linalg.norm(G, 'fro') ** 2
        
        return first_term + second_term

    def run_algo(self):

        print("Running incomplete info learning algo")

        # initialize vars
        A_0, Gamma_0 = self.initialize_variables()
        self.A = A_0
        self.Gamma = Gamma_0

        # print("b_0: ", b_0)
        # print("Gamma_0: ", Gamma_0)

        # tracker vars
        t = 1
        delta = 1
        obj_val = 0
        loss_values = []

        while ((delta > 10 ** -4) and (t < self.max_iters)):
            A_k = None
            Gamma_k = None

            if t != 1:
                A_k = copy.deepcopy(self.A.value)
                Gamma_k = copy.deepcopy(self.Gamma.value)
            else:
                A_k = self.A
                Gamma_k = self.Gamma
            
            # solve for G
            problem_G_k = self.create_opt_problem_G(self.actions, Gamma_k, A_k)
            problem_G_k.solve(solver='ECOS', verbose=False)
            G_k = copy.deepcopy(self.G.value)
            np.fill_diagonal(G_k, 0)

            # solve for A
            problem_A_k = self.create_opt_problem_A(self.actions, G_k, Gamma_k)
            problem_A_k.solve(solver='ECOS', verbose=False)
            A_k = copy.deepcopy(self.A.value)

            # solve for Gamma
            problem_Gamma_k = self.create_opt_problem_Gamma(self.actions, G_k, A_k)
            problem_Gamma_k.solve(solver='ECOS', verbose=False)
            Gamma_k = copy.deepcopy(self.Gamma.value)

            # check terminating condition and update trackers
            curr_obj_val = self.calc_objective_function(self.actions, G_k, Gamma_k, A_k)

            delta = np.abs(curr_obj_val - obj_val)
            obj_val = curr_obj_val
            loss_values.append(obj_val)
            t += 1

            # print("# of iterations:" , t)
            # print("G: ", self.G.value)
            # print("b: ", self.b.value)
            # print("Gamma: ", self.Gamma.value)

        opt_G = self.G.value
        opt_A = self.A.value
        opt_Gamma = self.Gamma.value

        # print(f"The learned graph is {opt_G}")
        # print(f"The learned marginal benefits are {opt_A}")
        # print(f"The learned information matrix is {opt_Gamma}")

        return opt_G, opt_A, opt_Gamma, loss_values