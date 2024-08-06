import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from algos.basic import Basic_Learning_Algo
from algos.homophily import Homophily_Learning_Algo
#from algos.incomplete import Incomplete_Info_Learning_Algo
from algos.test import Test
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
n = 8
k = 5
beta = 0.9
theta1 = 1
theta2 = 1

def cal_a(dim, beta, b, A):
    try:
        return np.linalg.inv(np.identity(dim[0]) - beta * A).dot(b)
    except:
        return np.linalg.pinv(np.identity(dim[0]) - beta * A).dot(b)

def homo_graph_series(n, Dim, beta, graph):
    L = nx.normalized_laplacian_matrix(graph).todense()
    A = nx.adjacency_matrix(graph).todense()
    L = np.linalg.pinv(L)
    b_com = np.random.multivariate_normal(mean = np.zeros(n), cov = L.tolist(), size = Dim).reshape(n, Dim) #+ noise_flag * np.random.normal(0, a_noise_level, (n, Dim))
    spectral_radius = np.max([np.abs(i.real) for i in np.linalg.eigvals(A)])
    a_com = cal_a(A.shape, beta * 1.0 / spectral_radius, b_com, A)
    return a_com, b_com, A

def compare_graphs(threshold, baseline, inferred):
    inferred[inferred < threshold] = 0
    inferred[inferred >= threshold] = 1

    plt.figure(figsize=(8, 4)).subplots_adjust(hspace=0.6)

    # plt.subplot(1, 2, 1)
    # plt.title("Baseline")
    # baseline_graph = nx.from_numpy_array(baseline)
    # nx.draw(baseline_graph, pos= nx.kamada_kawai_layout(baseline_graph), with_labels=True)

    # plt.subplot(1, 2, 2)
    # plt.title("Inferred")
    # inferred_graph = nx.from_numpy_array(inferred)
    # nx.draw(inferred_graph, pos= nx.kamada_kawai_layout(inferred_graph), with_labels=True)

    # plt.show()

    return roc_auc_score(baseline.flatten(), inferred.flatten())

def sim_n_times_basic(num_trials, n):
    graph_res = np.zeros(num_trials)
    benefits_res = np.zeros(num_trials)

    for i in range(0, num_trials):
        erdos = nx.erdos_renyi_graph(n, 0.2, seed=None, directed=False)
        actions, benefits, G = homo_graph_series(n, k, beta, erdos)
        basic = Basic_Learning_Algo(erdos, actions, theta1, theta2, beta)
        opt_G, opt_B = basic.run_algo()

        graph_res[i] = compare_graphs(0.0001, G, opt_G)
        benefits_res[i] = r2_score(benefits, opt_B)

    print(f"Basic Algo over {num_trials} num trials")
    print(f"Graph inference average AUC: {np.mean(graph_res)}")
    print(f"Graph inference AUC range: [{np.min(graph_res)}, {np.max(graph_res)}]")
    print(f"Benefits inference average R^2: {np.mean(benefits_res)}")
    print(f"Benefits inference R^2 range: [{np.min(benefits_res)}, {np.max(benefits_res)}]")

    return graph_res, benefits_res

#sim_n_times_basic(30, n)

def sim_n_times_homophily(num_trials, n, max_iters):
    graph_res = np.zeros(num_trials)
    benefits_res = np.zeros(num_trials)

    for i in range(0, num_trials):
        erdos = nx.erdos_renyi_graph(n, 0.2, seed=None, directed=False)
        actions, benefits, G = homo_graph_series(n, k, beta, erdos)
        homophily = Homophily_Learning_Algo(erdos, actions, theta1, theta2, beta, max_iters)
        opt_G, opt_B = homophily.run_algo()

        graph_res[i] = compare_graphs(0.0001, G, opt_G)
        benefits_res[i] = r2_score(benefits, opt_B)

    print(f"Homophily Algo over {num_trials} num trials")
    print(f"Graph inference average AUC: {np.mean(graph_res)}")
    print(f"Graph inference AUC range: [{np.min(graph_res)}, {np.max(graph_res)}]")
    print(f"Benefits inference average AUC: {np.mean(benefits_res)}")
    print(f"Benefits inference AUC range: [{np.min(benefits_res)}, {np.max(benefits_res)}]")

    return graph_res, benefits_res

#sim_n_times_homophily(1, n, 500)



#######################
### INCOMPLETE INFO ###
#######################

# params
n = 20 # num players
m = 2 # states of the world
k = 10 # num games
t = 2 # types
theta = 1

# joint distribution for private signals (symmetric)
def generate_joint_distribution_matrix(m):
    
    values = np.random.rand(m)
    values /= np.sum(values)
    joint_distribution_matrix = np.outer(values, values)
    
    return joint_distribution_matrix

# information matrix based on joint distribution matrix
def generate_information_matrix(P, m):
    
    Gamma = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            Gamma[i, j] = P[i, j] / np.sum(P[:, i])

    return Gamma

# marginal benefits prob matrix
def generate_benefits_distribution(T):
    P = np.random.rand(T, T)
    
    for t in range(T):
        for i in range(T):
            if i != t:
                P[t, i] = np.random.uniform(0, P[t, t] - 0.01)
    
    P /= P.sum()
    
    return P

# calculate actions
def cal_actions_incomplete(T, beta, B, A, Gamma):

    try:
        return np.linalg.inv(np.identity(A.shape[0] * T) - beta * np.kron((Gamma), A)) @ B
    except:
        return np.linalg.pinv(np.identity(A.shape[0] * T) - beta * np.kron((Gamma), A)) @ B
    
# marginal benefits generation for 2 states of the world
def cal_B_incomplete(n, Dim, L, T, benefits_prob, joint_prob):

    # generate n marginal benefits according to bivariate normal
    samples = np.random.multivariate_normal(mean = np.zeros(n), cov = L.tolist(), size = Dim).reshape(n, Dim) #+ noise_flag * np.random.normal(0, a_noise_level, (n, 1))
    samples = samples[:T, :]
    samples = np.sort(samples, axis=0)[::-1]
    B_com = np.zeros((T, Dim))

    for i in range(Dim):
        for j in range(T):
            prob_s_j = np.sum(joint_prob[:, j])
            expectation = 0
            for l in range(T):
                expectation += samples[l, i] * (benefits_prob[j, l] / prob_s_j)
            B_com[j, i] = expectation

    B_hat = np.kron(B_com, np.ones(n).reshape(1, -1).T)

    return B_hat, B_com

def homo_graph_series_incomplete(n, T, Dim, beta, graph, Gamma, joint_prob):
    L = nx.normalized_laplacian_matrix(graph).todense()
    A = nx.adjacency_matrix(graph).todense()
    L = np.linalg.pinv(L)
    benefits_probs = generate_benefits_distribution(T)
    B_hat, B_com = cal_B_incomplete(n, Dim, L, T, benefits_probs, joint_prob)
    spectral_radius = np.max([np.abs(i.real) for i in np.linalg.eigvals(A)])
    a_com = cal_actions_incomplete(t, beta * 1.0 / spectral_radius, B_hat, A, Gamma)
    beta = beta * 1.0 / spectral_radius
    return a_com, B_com, beta, A

def test(model):
    try:
        return model.run_algo()
    except:
        return test(model)

# plots heat map of adjacency matrix
def plot_heat_map(graph, title):

    plt.figure(figsize=(8, 6))
    sns.heatmap(graph, fmt=".2f", cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.show()

def plot_loss_function(loss_values):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, marker='o')
    plt.title('Loss Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def log_loss(G, G_opt):
    G = np.array(G)
    G_opt = np.array(G_opt)

    epsilon = 1e-15
    G_opt = np.clip(G_opt, epsilon, 1 - epsilon)
    log_loss_val =  -np.mean(G * np.log(G_opt) + (1 - G) * np.log(1 - G_opt))

    return log_loss_val

# simulates an incomplete information game num_trials times
def sim_n_times_incomplete(num_trials, n, theta, beta, max_iters):
    graph_res_auc = np.zeros(num_trials)
    graph_res_rmse = np.zeros(num_trials)
    graph_res_mae = np.zeros(num_trials)
    graph_res_log_loss = np.zeros(num_trials)
    benefits_res = np.zeros(num_trials)
    Gamma_res = np.zeros(num_trials)

    for i in range(0, num_trials):
        # erdos = nx.erdos_renyi_graph(n, 0.2, seed=None, directed=False)
        clique = nx.complete_graph(3)
        power_law = nx.barabasi_albert_graph(n, 3, seed=None, initial_graph=clique)
        P = generate_joint_distribution_matrix(2)
        Gamma = generate_information_matrix(P, 2)
        actions, benefits, beta_, G = homo_graph_series_incomplete(n, t, k, beta, power_law, Gamma, P)
        benefits = benefits.reshape(1,-1).T

        # print("Benefits: ", benefits)
        # print("Actions: ", actions.shape)
        # print("Gamma: ", Gamma)

        incomplete = Test(power_law, actions, theta, beta, 2, k, max_iters)
        opt_G, opt_A, opt_Gamma, loss_values = test(incomplete)
        # opt_G, opt_A, opt_Gamma, loss_values = incomplete.run_algo()

        graph_res_auc[i] = roc_auc_score(G.flatten(), opt_G.flatten())
        graph_res_mae[i] = mean_absolute_error(G, opt_G)
        graph_res_rmse[i] = np.sqrt(mean_squared_error(G, opt_G))
        graph_res_log_loss[i] = log_loss(G, opt_G)
        benefits_res[i] = r2_score(benefits.flatten(), opt_A.flatten())
        Gamma_res[i] = r2_score(Gamma.flatten(), opt_Gamma.flatten())
        
        # visualization
        # plot_loss_function(loss_values)
        # plot_heat_map(G, "True Graph")
        # plot_heat_map(opt_G, "Inferred Graph")

    print(f"Incomplete Algo over {num_trials} num trials")
    print(f"Graph inference average AUC: {np.mean(graph_res_auc)}")
    print(f"Graph inference AUC range: [{np.min(graph_res_auc)}, {np.max(graph_res_auc)}]")
    print(f"Graph inference average MAE: {np.mean(graph_res_mae)}")
    print(f"Graph inference MAE range: [{np.min(graph_res_auc)}, {np.max(graph_res_mae)}]")
    print(f"Graph inference average RMSE: {np.mean(graph_res_rmse)}")
    print(f"Graph inference RMSE range: [{np.min(graph_res_rmse)}, {np.max(graph_res_rmse)}]")
    print(f"Graph inference average log loss: {np.mean(graph_res_log_loss)}")
    print(f"Graph inference log loss range: [{np.min(graph_res_log_loss)}, {np.max(graph_res_log_loss)}]")
    print(f"Benefits inference average r2: {np.mean(benefits_res)}")
    print(f"Benefits inference r2 range: [{np.min(benefits_res)}, {np.max(benefits_res)}]")
    print(f"Gamma inference average r2: {np.mean(Gamma_res)}")
    print(f"Gamma inference r2 range: [{np.min(Gamma_res)}, {np.max(Gamma_res)}]")

    return graph_res_auc, graph_res_mae, graph_res_rmse, graph_res_log_loss, benefits_res, Gamma_res

# simulations + plots
betas = [0.5] #betas = [0.1, 0.3, 0.5, 0.7, 0.9]
num_trials = 10
res_auc = np.zeros(len(betas))
res_mae = np.zeros(len(betas))
res_rmse = np.zeros(len(betas))
res_log_loss = np.zeros(len(betas))

for i, b in enumerate(betas):
    print()
    print("beta: ", b)
    auc, mae, rmse, log_loss, benefits, Gamma = sim_n_times_incomplete(num_trials, n, theta, b, 2000)
    res_auc[i] = np.mean(auc)
    res_mae[i] = np.mean(mae)
    res_rmse[i] = np.mean(rmse)
    res_log_loss[i] = np.mean(log_loss)

plt.figure(figsize=(8, 4)).subplots_adjust(hspace=0.6)

plt.subplot(1, 3, 1)
plt.plot(betas, res_mae)
plt.xlabel("beta")
plt.ylabel("MAE")
plt.title("beta vs. MAE")

plt.subplot(1, 3, 2)
plt.plot(betas, res_rmse)
plt.xlabel("beta")
plt.ylabel("RMSE")
plt.title("beta vs. RMSE")

plt.subplot(1, 3, 3)
plt.plot(betas, res_auc)
plt.xlabel("beta")
plt.ylabel("AUC")
plt.title("beta vs. AUC")
