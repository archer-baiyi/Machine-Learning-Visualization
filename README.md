# Machine Learning Algorithm Visualizations

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-success.svg)](https://machine-learning-virtualization.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-black.svg)](https://flask.palletsprojects.com/)

An interactive, web-based educational dashboard designed to visualize the inner workings of fundamental Machine Learning algorithms step-by-step. 

**🚀 Live Demo:** [https://machine-learning-virtualization.onrender.com](https://machine-learning-virtualization.onrender.com)

---

## 💡 About The Project

Understanding how Machine Learning algorithms converge can be difficult through static equations alone. This project bridges the gap by providing a highly interactive visualization tool. It separates heavy mathematical computations (handled by a Python backend) from the visual rendering (handled by HTML5 Canvas on the frontend).

### 📊 Currently Supported Algorithms:

1. **Gaussian Mixture Model (GMM) via EM Algorithm:** - Visualizes **Soft Assignments**.
   - Displays Expectation (E-Step) and Maximization (M-Step) separately.
   - Dynamically renders covariance ellipses showing distribution shapes, stretching, and rotation.
   
2. **K-Means Clustering:**
   - Visualizes **Hard Assignments**.
   - Displays assignment steps (assigning points to nearest centroids) and update steps (moving centroids to cluster means).

*(More algorithms, such as Support Vector Machines, are currently in development).*

---

## ✨ Key Features

- **Interactive Step-by-Step Playback:** Move forward and backward through exact algorithm steps (e.g., jump between E-Steps and M-Steps) to see exactly how parameters are updated.
- **Time Travel:** A history state management system allows you to undo steps and revisit previous iterations smoothly.
- **Fast Forward:** Skip the tedious micro-adjustments and jump ahead by $N$ iterations instantly.
- **Custom Data Generation:** - **Auto Generate:** Create multi-cluster data with random variances and covariances based on specific parameters.
  - **Mouse Click:** Manually drop data points or clusters directly onto the canvas to test edge cases.
- **Responsive Canvas:** Clean, grid-based coordinate system for accurate spatial representations.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, NumPy, SciPy (for strict mathematical matrix computations and PDF calculations).
- **Frontend:** Vanilla JavaScript, HTML5 Canvas (for lightweight, zero-dependency, real-time rendering), CSS3.

---

## 💻 Local Installation & Usage

If you want to run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/archer-baiyi/Machine-Learning-Virtualization.git](https://github.com/archer-baiyi/Machine-Learning-Virtualization.git)
   cd Machine-Learning-Virtualization
   ```

2. Create a virtual environment (Optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install Flask numpy scipy
```

4. Run the application:

```bash
python app.py
```

5. Open your browser:
```bash
Navigate to http://127.0.0.1:5000
```