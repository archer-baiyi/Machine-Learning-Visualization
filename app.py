from flask import Flask, render_template, jsonify, request
import numpy as np
from scipy.stats import multivariate_normal
import copy

app = Flask(__name__)

# ==========================================
# Global History States
# ==========================================
gmm_state_history = []
gmm_current_index = 0

kmeans_state_history = []
kmeans_current_index = 0

# ==========================================
# Page Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gmm')
def gmm_page():
    return render_template('gmm.html')

@app.route('/kmeans')
def kmeans_page():
    return render_template('kmeans.html')

# ==========================================
# 1. GMM API Routes
# ==========================================
@app.route('/api/gmm/init', methods=['POST'])
def init_gmm_state():
    global gmm_state_history, gmm_current_index
    data = request.get_json()
    X = np.array(data.get('X', []))
    K = int(data.get('K', 2))
    if len(X) < K: return jsonify({"error": "Points must be >= K."}), 400

    N, D = X.shape
    np.random.seed()
    initial_indices = np.random.choice(N, K, replace=False)
    means = X[initial_indices].copy()
    data_variance = np.var(X, axis=0)
    covs = [np.diag(data_variance) for _ in range(K)]
    
    initial_state = {
        "X": X.tolist(), "N": N, "K": K, "D": D,
        "means": means.tolist(),
        "covs": [c.tolist() for c in covs],
        "weights": (np.ones(K) / K).tolist(),
        "resps": (np.ones((N, K)) / K).tolist(),
        "step_count": 0, "iteration": 0, "current_action": "Initialization"
    }
    gmm_state_history = [initial_state]
    gmm_current_index = 0
    return jsonify(gmm_state_history[gmm_current_index])

def compute_gmm_next_step(current_state):
    next_state = copy.deepcopy(current_state)
    X = np.array(next_state["X"])
    K, N, D = next_state["K"], next_state["N"], next_state["D"]
    means = np.array(next_state["means"])
    covs = [np.array(c) for c in next_state["covs"]]
    weights = np.array(next_state["weights"])
    resps = np.array(next_state["resps"])
    
    if next_state["step_count"] % 2 == 0:
        for k in range(K):
            resps[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covs[k])
        resps_sum = resps.sum(axis=1, keepdims=True)
        resps_sum[resps_sum == 0] = 1e-10 
        resps /= resps_sum
        next_state["current_action"] = "E-Step: Update Probabilities"
    else:
        Nk = resps.sum(axis=0)
        Nk[Nk == 0] = 1e-10
        weights = Nk / N
        for k in range(K):
            means[k] = np.average(X, axis=0, weights=resps[:, k])
            diff = X - means[k]
            covs[k] = np.dot((resps[:, k:k+1] * diff).T, diff) / Nk[k] + np.eye(D) * 1e-4
        next_state["current_action"] = "M-Step: Update Parameters"
        next_state["iteration"] += 1

    next_state["step_count"] += 1
    next_state["means"] = means.tolist()
    next_state["covs"] = [c.tolist() for c in covs]
    next_state["weights"] = weights.tolist()
    next_state["resps"] = resps.tolist()
    return next_state

@app.route('/api/gmm/action', methods=['POST'])
def handle_gmm_action():
    global gmm_state_history, gmm_current_index
    payload = request.get_json()
    action = payload.get('action')
    n = int(payload.get('n', 10))
    
    if action == 'prev_step': gmm_current_index = max(0, gmm_current_index - 1)
    elif action == 'prev_iter': gmm_current_index = max(0, gmm_current_index - 2)
    elif action == 'next_step':
        if gmm_current_index == len(gmm_state_history) - 1: gmm_state_history.append(compute_gmm_next_step(gmm_state_history[gmm_current_index]))
        gmm_current_index += 1
    elif action == 'next_iter':
        for _ in range(2): 
            if gmm_current_index == len(gmm_state_history) - 1: gmm_state_history.append(compute_gmm_next_step(gmm_state_history[gmm_current_index]))
            gmm_current_index += 1
    elif action == 'next_n_iter':
        for _ in range(n * 2):
            if gmm_current_index == len(gmm_state_history) - 1: gmm_state_history.append(compute_gmm_next_step(gmm_state_history[gmm_current_index]))
            gmm_current_index += 1

    return jsonify(gmm_state_history[gmm_current_index])

# ==========================================
# 2. K-Means API Routes
# ==========================================
@app.route('/api/kmeans/init', methods=['POST'])
def init_kmeans_state():
    global kmeans_state_history, kmeans_current_index
    data = request.get_json()
    X = np.array(data.get('X', []))
    K = int(data.get('K', 2))
    if len(X) < K: return jsonify({"error": "Points must be >= K."}), 400

    N, D = X.shape
    np.random.seed()
    # Randomly pick K points as initial centroids
    initial_indices = np.random.choice(N, K, replace=False)
    centroids = X[initial_indices].copy()
    
    initial_state = {
        "X": X.tolist(), "N": N, "K": K,
        "centroids": centroids.tolist(),
        "labels": [-1] * N, # -1 means unassigned
        "step_count": 0, "iteration": 0, "current_action": "Initialization"
    }
    kmeans_state_history = [initial_state]
    kmeans_current_index = 0
    return jsonify(kmeans_state_history[kmeans_current_index])

def compute_kmeans_next_step(current_state):
    next_state = copy.deepcopy(current_state)
    X = np.array(next_state["X"])
    K, N = next_state["K"], next_state["N"]
    centroids = np.array(next_state["centroids"])
    labels = np.array(next_state["labels"])
    
    if next_state["step_count"] % 2 == 0:
        # ---- Step 1: Assignment (Expectation) ----
        for i in range(N):
            distances = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = np.argmin(distances)
        next_state["current_action"] = "Assign Points to Nearest Centroid"
    else:
        # ---- Step 2: Update (Maximization) ----
        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            # If a cluster is empty, centroid stays in place
        next_state["current_action"] = "Move Centroids to Cluster Means"
        next_state["iteration"] += 1

    next_state["step_count"] += 1
    next_state["centroids"] = centroids.tolist()
    next_state["labels"] = labels.tolist()
    return next_state

@app.route('/api/kmeans/action', methods=['POST'])
def handle_kmeans_action():
    global kmeans_state_history, kmeans_current_index
    payload = request.get_json()
    action = payload.get('action')
    n = int(payload.get('n', 10))
    
    if action == 'prev_step': kmeans_current_index = max(0, kmeans_current_index - 1)
    elif action == 'prev_iter': kmeans_current_index = max(0, kmeans_current_index - 2)
    elif action == 'next_step':
        if kmeans_current_index == len(kmeans_state_history) - 1: kmeans_state_history.append(compute_kmeans_next_step(kmeans_state_history[kmeans_current_index]))
        kmeans_current_index += 1
    elif action == 'next_iter':
        for _ in range(2): 
            if kmeans_current_index == len(kmeans_state_history) - 1: kmeans_state_history.append(compute_kmeans_next_step(kmeans_state_history[kmeans_current_index]))
            kmeans_current_index += 1
    elif action == 'next_n_iter':
        for _ in range(n * 2):
            if kmeans_current_index == len(kmeans_state_history) - 1: kmeans_state_history.append(compute_kmeans_next_step(kmeans_state_history[kmeans_current_index]))
            kmeans_current_index += 1

    return jsonify(kmeans_state_history[kmeans_current_index])

if __name__ == '__main__':
    app.run(debug=True, port=5000)