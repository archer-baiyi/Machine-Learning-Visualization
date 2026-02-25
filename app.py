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

svm_state_history = []
svm_current_index = 0

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

@app.route('/svm')
def svm_page():
    return render_template('svm.html')

# ==========================================
# 1. GMM API
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
        "means": means.tolist(), "covs": [c.tolist() for c in covs],
        "weights": (np.ones(K) / K).tolist(), "resps": (np.ones((N, K)) / K).tolist(),
        "step_count": 0, "iteration": 0, "current_action": "Initialization"
    }
    gmm_state_history = [initial_state]
    gmm_current_index = 0
    return jsonify(gmm_state_history[gmm_current_index])

def compute_gmm_next_step(current_state):
    next_state = copy.deepcopy(current_state)
    X, K, N, D = np.array(next_state["X"]), next_state["K"], next_state["N"], next_state["D"]
    means, covs = np.array(next_state["means"]), [np.array(c) for c in next_state["covs"]]
    weights, resps = np.array(next_state["weights"]), np.array(next_state["resps"])
    
    if next_state["step_count"] % 2 == 0:
        for k in range(K): resps[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covs[k])
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
    next_state["means"], next_state["covs"] = means.tolist(), [c.tolist() for c in covs]
    next_state["weights"], next_state["resps"] = weights.tolist(), resps.tolist()
    return next_state

@app.route('/api/gmm/action', methods=['POST'])
def handle_gmm_action():
    global gmm_state_history, gmm_current_index
    payload = request.get_json()
    action, n = payload.get('action'), int(payload.get('n', 10))
    if action == 'prev_step': gmm_current_index = max(0, gmm_current_index - 1)
    elif action == 'prev_iter': gmm_current_index = max(0, gmm_current_index - 2)
    elif action in ['next_step', 'next_iter', 'next_n_iter']:
        steps = 1 if action == 'next_step' else (2 if action == 'next_iter' else n * 2)
        for _ in range(steps):
            if gmm_current_index == len(gmm_state_history) - 1: gmm_state_history.append(compute_gmm_next_step(gmm_state_history[gmm_current_index]))
            gmm_current_index += 1
    return jsonify(gmm_state_history[gmm_current_index])

# ==========================================
# 2. K-Means API
# ==========================================
@app.route('/api/kmeans/init', methods=['POST'])
def init_kmeans_state():
    global kmeans_state_history, kmeans_current_index
    data = request.get_json()
    X = np.array(data.get('X', []))
    K = int(data.get('K', 2))
    if len(X) < K: return jsonify({"error": "Points must be >= K."}), 400
    np.random.seed()
    centroids = X[np.random.choice(X.shape[0], K, replace=False)].copy()
    initial_state = {
        "X": X.tolist(), "N": X.shape[0], "K": K, "centroids": centroids.tolist(),
        "labels": [-1] * X.shape[0], "step_count": 0, "iteration": 0, "current_action": "Initialization"
    }
    kmeans_state_history, kmeans_current_index = [initial_state], 0
    return jsonify(kmeans_state_history[kmeans_current_index])

def compute_kmeans_next_step(current_state):
    next_state = copy.deepcopy(current_state)
    X, K, N = np.array(next_state["X"]), next_state["K"], next_state["N"]
    centroids, labels = np.array(next_state["centroids"]), np.array(next_state["labels"])
    if next_state["step_count"] % 2 == 0:
        for i in range(N): labels[i] = np.argmin(np.linalg.norm(centroids - X[i], axis=1))
        next_state["current_action"] = "Assign Points to Nearest Centroid"
    else:
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0: centroids[k] = np.mean(pts, axis=0)
        next_state["current_action"] = "Move Centroids to Cluster Means"
        next_state["iteration"] += 1
    next_state["step_count"] += 1
    next_state["centroids"], next_state["labels"] = centroids.tolist(), labels.tolist()
    return next_state

@app.route('/api/kmeans/action', methods=['POST'])
def handle_kmeans_action():
    global kmeans_state_history, kmeans_current_index
    payload = request.get_json()
    action, n = payload.get('action'), int(payload.get('n', 10))
    if action == 'prev_step': kmeans_current_index = max(0, kmeans_current_index - 1)
    elif action == 'prev_iter': kmeans_current_index = max(0, kmeans_current_index - 2)
    elif action in ['next_step', 'next_iter', 'next_n_iter']:
        steps = 1 if action == 'next_step' else (2 if action == 'next_iter' else n * 2)
        for _ in range(steps):
            if kmeans_current_index == len(kmeans_state_history) - 1: kmeans_state_history.append(compute_kmeans_next_step(kmeans_state_history[kmeans_current_index]))
            kmeans_current_index += 1
    return jsonify(kmeans_state_history[kmeans_current_index])

# ==========================================
# 3. SVM API (Linear SVM via Gradient Descent)
# ==========================================
@app.route('/api/svm/init', methods=['POST'])
def init_svm_state():
    global svm_state_history, svm_current_index
    data = request.get_json()
    X = np.array(data.get('X', []))
    y = np.array(data.get('y', [])) # Tags: +1 or -1
    C = float(data.get('C', 1.0))
    
    if len(X) < 2 or len(np.unique(y)) < 2:
        return jsonify({"error": "Need at least two points and both classes (+1 and -1) to train SVM."}), 400

    np.random.seed()
    # Initialize weights randomly near zero
    w = np.random.randn(2) * 0.1
    b = 0.0
    
    initial_state = {
        "X": X.tolist(), "y": y.tolist(), "C": C,
        "w": w.tolist(), "b": b,
        "step_count": 0, "iteration": 0, "current_action": "Initialization (Random Hyperplane)"
    }
    svm_state_history = [initial_state]
    svm_current_index = 0
    return jsonify(svm_state_history[svm_current_index])

def compute_svm_next_step(current_state):
    next_state = copy.deepcopy(current_state)
    X = np.array(next_state["X"])
    y = np.array(next_state["y"])
    w = np.array(next_state["w"])
    b = next_state["b"]
    C = next_state["C"]
    
    lr = 0.05 # Fixed learning rate for visual stability
    
    # Calculate margin: y_i * (w^T x_i + b)
    margins = y * (np.dot(X, w) + b)
    # Points inside the margin or misclassified (margin < 1)
    misclassified = margins < 1
    
    # Gradients for Hinge Loss
    # L = 1/2 ||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
    grad_w = w - C * np.dot(y[misclassified], X[misclassified])
    grad_b = -C * np.sum(y[misclassified])
    
    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    
    next_state["w"] = w_new.tolist()
    next_state["b"] = b_new
    next_state["step_count"] += 1
    next_state["iteration"] += 1
    next_state["current_action"] = f"Gradient Step (Support Vectors: {np.sum(misclassified)})"
    
    return next_state

@app.route('/api/svm/action', methods=['POST'])
def handle_svm_action():
    global svm_state_history, svm_current_index
    payload = request.get_json()
    action = payload.get('action')
    n = int(payload.get('n', 10))
    
    if action == 'prev_step' or action == 'prev_iter':
        svm_current_index = max(0, svm_current_index - 1)
    elif action == 'next_step' or action == 'next_iter':
        if svm_current_index == len(svm_state_history) - 1:
            svm_state_history.append(compute_svm_next_step(svm_state_history[svm_current_index]))
        svm_current_index += 1
    elif action == 'next_n_iter':
        for _ in range(n):
            if svm_current_index == len(svm_state_history) - 1:
                svm_state_history.append(compute_svm_next_step(svm_state_history[svm_current_index]))
            svm_current_index += 1

    return jsonify(svm_state_history[svm_current_index])

if __name__ == '__main__':
    app.run(debug=True, port=5000)