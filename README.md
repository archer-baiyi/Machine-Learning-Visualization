# Machine Learning Algorithm Visualizations

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-success.svg)](https://machine-learning-virtualization.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-black.svg)](https://flask.palletsprojects.com/)

An interactive, web-based educational dashboard designed to visualize the inner workings of fundamental Machine Learning algorithms step-by-step. 

**🚀 Live Demo:** [https://machine-learning-virtualization.onrender.com](https://machine-learning-virtualization.onrender.com)

> ⚠️ **Note:** The backend is hosted on Render's free tier. It goes to sleep after 15 minutes of inactivity, so **the initial load may take 1-2 minutes**. Please be patient! Once awake, all operations are instantaneous.

---

## 💡 About The Project

Understanding how Machine Learning algorithms converge can be difficult through static equations alone. This project bridges the gap by providing a highly interactive visualization tool. It separates heavy mathematical computations (handled by a Python backend) from the visual rendering (handled by HTML5 Canvas on the frontend).

### 📊 Currently Supported Algorithms:

#### Classification (Supervised Learning)
1. **Linear SVM (Support Vector Machine):**
   - Visualizes the maximal margin hyperplane separating two classes using **Gradient Descent**.
   - Dynamically highlights **Support Vectors** (points inside the margin or misclassified) during the optimization process.
   - Allows real-time adjustment of the **C (Penalty)** parameter to observe the shift between hard-margin and soft-margin behavior.

2. **Linear Classification with Basis Functions:**
   - A static, high-DPI visual gallery demonstrating the core intuition behind the **Kernel Trick**.
   - Showcases how complex, non-linearly separable datasets (e.g., Concentric Circles, Checkerboards, Sinusoidal waves, Parabolas) become linearly separable by applying mathematical feature transformations.
   - Features beautifully rendered LaTeX equations explaining the exact spatial mapping.

#### Clustering (Unsupervised Learning)
3. **Gaussian Mixture Model (GMM) via EM Algorithm:**
   - Visualizes **Soft Assignments**.
   - Displays Expectation (E-Step) and Maximization (M-Step) separately.
   - Dynamically renders covariance ellipses showing distribution shapes, stretching, and rotation.
   
4. **K-Means Clustering:**
   - Visualizes **Hard Assignments** using exact Voronoi decision boundaries.
   - Displays assignment steps (assigning points to nearest centroids) and update steps (moving centroids to cluster means).

---

## ✨ Key Features

- **Interactive Step-by-Step Playback:** Move forward and backward through exact algorithm steps (e.g., jump between E-Steps and M-Steps, or gradient steps) to see exactly how parameters are updated.
- **Time Travel:** A history state management system allows you to undo steps and revisit previous iterations smoothly.
- **Fast Forward:** Skip the tedious micro-adjustments and jump ahead by $N$ iterations instantly.
- **Custom Data Generation:** - **Auto Generate:** Create multi-cluster or linearly separable data with random variances and covariances based on specific parameters.
  - **Mouse Click:** Manually drop labeled or unlabeled data points directly onto the canvas to test edge cases.
- **Responsive High-DPI Canvas:** Clean, grid-based coordinate system for accurate spatial representations, automatically scaled for Retina displays.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, NumPy, SciPy (for strict mathematical matrix computations, gradients, and PDF calculations).
- **Frontend:** Vanilla JavaScript, HTML5 Canvas (for lightweight, zero-dependency, real-time rendering), CSS3, MathJax (for LaTeX rendering).

---

## 💻 Local Installation & Usage

If you want to run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/archer-baiyi/Machine-Learning-Virtualization.git
   cd Machine-Learning-Virtualization
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install Flask numpy scipy
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   Navigate to `http://127.0.0.1:5000`