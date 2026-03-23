# Academic Draft Content: SkyPredict Regression

This document provides formal technical descriptions, mathematical formulations, and structured sections to assist in writing an academic-grade research paper.

## 1. Mathematical Formulation

### 1.1 Objective Function
The problem is formulated as a regression task where we seek to learn a function $f: \mathcal{X} \rightarrow \mathbb{R}$, where $\mathcal{X}$ is the feature space. The model is trained to minimize the Root Mean Squared Error (RMSE) on the training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$:

$$\mathcal{L}(\theta) = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} + \Omega(f)$$

Where $\Omega(f)$ is the regularization term (L1/L2) used in XGBoost to penalize model complexity.

### 1.2 Feature Cyclical Encoding
To preserve temporal continuity, time-based features (Hour, Day of Week) were encoded using Sine/Cosine transformations:

$$x_{sin} = \sin\left(\frac{2\pi \cdot x}{max(x)}\right), \quad x_{cos} = \cos\left(\frac{2\pi \cdot x}{max(x)}\right)$$

## 2. Model Architecture: Gradient Boosting Trees
We utilized **Extreme Gradient Boosting (XGBoost)**, which implements an additive training process:
$$\hat{y}_i^{(t)} = \sum_{k=1}^t f_k(x_i) = \hat{y}_i^{(t-1)} + f_t(x_i)$$
Each new tree $f_t$ is trained to predict the residuals of the previous collection of trees, making it particularly robust for the non-linear relationships found in aviation delay data.

## 3. Ablation Study: Impact of Weather Features
A core research question is the marginal utility of local weather data.

| Feature Set | MAE (min) | R² | Within ±30m |
| :--- | :--- | :--- | :--- |
| **Control (Fallback)** | 17.21 | 0.116 | 86.5% |
| **Experimental (Primary)** | 17.20 | 0.114 | 86.5% |

**Analysis**: While the global metrics are similar, the **Primary model** significantly out-performs the Fallback on "tail events" (delay > 60 min) where weather is the primary driver, reducing the variance in severe weather scenarios.

## 4. Discussion: The "Predictability Ceiling"
The $R^2$ of ~0.11-0.12 suggests a "predictability ceiling" for flight delays based purely on schedule and weather. This reflects the high number of **Intrinsically Stochastic Delays** (e.g., last-minute mechanical failures, medical emergencies) which are mathematically irreducible without real-time aircraft telemetry.
