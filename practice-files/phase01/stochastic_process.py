'''
Building stochastic processes from scratch and implementing it(random walk, Markov Chain, Metropolis hastings, langevin dynamics),
and forward diffusion process, I build them from scratch and apply on real example to see how they work, it is given in example
exercies down in this file.  
'''
import random
import math
import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)
    
    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states
    
    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
    



def random_walk(steps=100):
    curr_pos = 0.0
    # pos_track = []

    for i in range(steps):
        flip = random.random()
        if flip > 0.5:
            curr_pos += 1.0
            # pos_track.append(curr_pos)
        else:
            curr_pos -= 1.0
            # pos_track.append(curr_pos)
    return curr_pos


## Exercise 1: Simulate 1000 random walks of 10000 steps. Plot the distribution of final 
# positions. Verify it is approximately Gaussian with mean 0 and standard deviation sqrt(10000) = 100.

rand_walks = 1000

finalpositions = []

for i in range(rand_walks):
    positions = random_walk(10000)
    finalpositions.append(positions)


mean_mu = sum(finalpositions) / rand_walks
var_pos = sum([(xi - mean_mu) ** 2 for xi in finalpositions]) / rand_walks
std = math.sqrt(var_pos)


print(mean_mu)  # it came out to be 
print(std)


def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions

final_pos = []
for i in range(rand_walks):
    positions = random_walk_1d(10000)
    final_pos.append(positions[-1])

mu = sum(final_pos) / rand_walks
var = sum([(xi - mu) ** 2 for xi in final_pos]) / rand_walks
sigma = math.sqrt(var)
print(mu, sigma)


## Exercise 2: Build a text generator using a Markov chain. Train on a small corpus: for each word, count transitions 
# to the next word. Build the transition matrix. Generate new sentences by sampling from the chain.

corpus = ['the', 'king', 'of', 'the', 'country', 'Italia', 'is', 'at','tower',
         'heat', 'in', 'summer', 'is', 'unbearable',
          'tourist', 'takes', 'train', 'to', 'florence', 'from', 'station',
          'women', 'kids', 'play', 'in', 'beach']

unique_words = sorted(list(set(corpus)))
word_to_idx = {word:i for i, word in enumerate(unique_words)}
idx_to_word = {i:word for i, word in enumerate(unique_words)}
n_states = len(unique_words)
print(word_to_idx)

# count transition
# initialize a matrix of zeros

transition_counts = np.zeros((n_states, n_states))

# count how often a wordA is followed by word B
for i in range(len(corpus) - 1):
    curr_word = corpus[i]
    next_word = corpus[i+1]
    curr_idx = word_to_idx[curr_word]
    next_idx = word_to_idx[next_word]
    transition_counts[curr_idx, next_idx] += 1

# convert counts ot probablities(raw normalization)

for i in range(n_states):
    row_sum = transition_counts[i].sum()
    if row_sum > 0:
        transition_counts[i] = transition_counts[i] / row_sum
    else:
        transition_counts[i] = np.ones(n_states) / n_states  # handle rows with 0 transition

mc = MarkovChain(transition_matrix=transition_counts, state_names=unique_words)

# generate 10 words starting from "the"

start_idx = word_to_idx["the"]
generate_indices = mc.simulate(start_state=start_idx, n_steps=10, seed=42)


# convert indices back to words

generate_sentence =  " ".join([idx_to_word[i] for i in generate_indices])
print(generate_sentence)

## Exercise 3: Implement simulated annealing using Metropolis-Hastings. Start at high temperature (accept almost everything) 
# and gradually cool down (accept only improvements). Use it to find the minimum of a function with many local minima.

def metropolis_hastings_sim_annealing(func, x0, n_samples, T0=100.0, cooling_rate = 0.995, proposal_std = 0.5, seed=None):
    rng = np.random.RandomState(42)
    x = np.array(x0, dtype=float)

    best_x = x.copy()
    best_cost = func(x)

    samples = [x.copy()]
    T = T0
    accepted = 0

    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std

        current_cost = func(x)
        cost_proposed = func(x_proposed)

        # here we want to minimize
        energy_del = current_cost - cost_proposed  # our ratio

        if energy_del > 0 or np.log(rng.rand()) < (energy_del/ T):
            x = x_proposed
            accepted += 1

            # keeping track of absolute best global min.
            if cost_proposed < best_cost:
                best_x = x_proposed.copy()
                best_cost = cost_proposed

        samples.append(x.copy())
        T *= cooling_rate 
    acceptance_rate = accepted / (n_samples - 1)
    return best_x, best_cost, np.array(samples), acceptance_rate

def min_func(x):
    return x**2 + 10 * np.cos(2 * np.pi * x)


start_pos = [5.0]
best_x, best_cost, path, acceptance = metropolis_hastings_sim_annealing(min_func, x0=start_pos, n_samples=10000)

print(f"Starting position: {start_pos}")
print(f'Found global minimum at x= {best_x[0]:.4f}')
print(f"Function value at minimum: {best_cost}")
print(f"acceptance rate is {acceptance}")


## Exercise 4: Compare Langevin dynamics at different temperatures. Sample from a double-well potential U(x) = (x^2 - 1)^2. 
# At low temperature, samples cluster in one well. At high temperature, they spread across both. Find the critical 
# temperature where the chain mixes between wells.


def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(42)
    x = np.array(x0, dtype=float)
    trajectory = []

    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise

        # print(c_loss, loss)
        trajectory.append(x.copy())
    return np.array(trajectory)

def U(x):
    return (x**2 - 1)**2

temps = [0.05, 0.1, 0.5, 1.0, 2.0, 10.0]

def grad_U(x):
    return 2 * (x**2 - 1) * (2 * x)

final_results = []
start = [-1.0]

for t in temps:
    traj = langevin_dynamics(grad_U, x0=start, dt=0.01, temperature=t, n_steps=10000)
    time_spent_in_right_well = np.sum(traj > 0) / len(traj) * 100
    print(f"Temp : {t:<5} | Min X: {traj.min():.2f} | Max X: {traj.max():.2f} | time spent the right well: {time_spent_in_right_well:.1f}%")


## Exercise 5: Implement the forward diffusion process. Start with a 1D signal (e.g., a sine wave). Add noise progressively over 100 
# steps with a linear noise schedule. Show how the signal degrades to pure noise. Then implement a simple denoiser that 
# reverses the process (even a naive one that just subtracts the estimated noise).

fs = 32  # let's take a signal of freq 120
t = np.linspace(0, 1, fs)
orignal_signal = np.sin(2 * np.pi * 5 * t)  # a 5 hz sine wave

n_steps = 100
beta = np.linspace(0.001, 0.02, n_steps)

forward_history = [orignal_signal.copy()]

# forward process; adding noise
current_signal = orignal_signal.copy()

for t in range(n_steps):
    noise = np.random.randn(fs)
    current_signal = np.sqrt(1 - beta[t]) * current_signal + np.sqrt(beta[t]) * noise
    forward_history.append(current_signal.copy())

print(f"Forwarded process done added linear noise beta_t over 100 steps and final shape: {current_signal.shape}")

 
# Reverse Process:
# we start wtih highest noise at the end and try to move back from their.
reverse_hist = [forward_history[-1].copy()]
denoised_signal = forward_history[-1].copy()

for t in reversed(range(n_steps)):
    # re-generating the same noise vector for reconstructing testing
    np.random.seed(t)
    estimated_noise = np.random.randn(fs)

    # algebraically solving for previous step:
    denoised_signal = (denoised_signal - np.sqrt(beta[t]) * estimated_noise) / np.sqrt(1 - beta[t])
    reverse_hist.append(denoised_signal)

print(f"Reversed process done: reconstructed signal matches original: {np.allclose(orignal_signal, denoised_signal)}")
print(orignal_signal )
print(forward_history)