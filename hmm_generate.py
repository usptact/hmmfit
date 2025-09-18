import numpy as np


def generate_hmm_series(n_steps, init_probs, transition_matrix, means, stds, random_state=None):
    """
    Generate a time series from an HMM with Gaussian emissions.

    Parameters
    ----------
    n_steps : int
        Length of the time series to generate.
    init_probs : array-like, shape (n_states,)
        Initial state distribution.
    transition_matrix : array-like, shape (n_states, n_states)
        State transition probability matrix.
    means : array-like, shape (n_states,)
        Mean of Gaussian emission for each state.
    stds : array-like, shape (n_states,)
        Std deviation of Gaussian emission for each state.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    states : np.ndarray, shape (n_steps,)
        Sequence of hidden states.
    observations : np.ndarray, shape (n_steps,)
        Generated observations.
    """
    rng = np.random.default_rng(random_state)
    n_states = len(init_probs)

    # Sample initial state
    states = np.zeros(n_steps, dtype=int)
    observations = np.zeros(n_steps)
    states[0] = rng.choice(n_states, p=init_probs)
    observations[0] = rng.normal(means[states[0]], stds[states[0]])

    # Generate subsequent states and observations
    for t in range(1, n_steps):
        states[t] = rng.choice(n_states, p=transition_matrix[states[t - 1]])
        observations[t] = rng.normal(means[states[t]], stds[states[t]])

    return states, observations


# Example usage: 3-state HMM
if __name__ == "__main__":
    init_probs = [0.5, 0.3, 0.2]  # starting state distribution
    transition_matrix = [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ]
    means = [0.0, 0.0, 0.0]  # emission means per state
    stds = [1.0, 1.0, 1.0]  # emission std devs per state

    states, obs = generate_hmm_series(
        n_steps=50,
        init_probs=init_probs,
        transition_matrix=transition_matrix,
        means=means,
        stds=stds,
        random_state=43
    )

    print("States:", states)
    print("Observations:", obs)
