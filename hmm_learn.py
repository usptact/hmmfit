import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Import your generator
from hmm_generate import generate_hmm_series

# --- Define true parameters ---
init_probs = np.array([0.2, 0.8])
transition_matrix = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])
means = np.array([0.0, 1.0])
stds = np.array([0.9, 0.9])

# --- Generate synthetic data ---
n_samples = 1000
states, observations = generate_hmm_series(
    n_steps=n_samples,
    init_probs=init_probs,
    transition_matrix=transition_matrix,
    means=means,
    stds=stds,
    random_state=42
)

side_observations = states * 2 + np.random.normal(0, 0.3, size=len(states))

X = np.column_stack([observations, side_observations])

# Reshape for hmmlearn (expects 2D: (n_samples, n_features))
#X = observations.reshape(-1, 1)

# --- Fit Gaussian HMM ---
n_states = len(init_probs)
model = hmm.GaussianHMM(
    n_components=n_states,
    covariance_type="diag",
    n_iter=200,
    random_state=42
)
model.fit(X)

# --- Extract recovered parameters ---
print("Recovered start probabilities:\n", model.startprob_)
print("\nRecovered transition matrix:\n", model.transmat_)
print("\nRecovered means:\n", model.means_.flatten())
print("\nRecovered variances:\n", model.covars_.flatten())

# --- Decode hidden states ---
hidden_states = model.predict(X)

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Subplot 1: side information
axes[0].plot(side_observations, label="Side information", color="purple", alpha=0.7)
axes[0].legend()
axes[0].set_ylabel("Side feature")

# Subplot 2: main observation with states
axes[1].plot(observations, label="Observations", alpha=0.6)
axes[1].plot(states, label="True states", alpha=0.8)
axes[1].plot(hidden_states, label="Recovered states", linestyle="--", alpha=0.8)
axes[1].legend()
axes[1].set_ylabel("Main observation")
axes[1].set_xlabel("Timestep")

plt.tight_layout()
plt.show()
