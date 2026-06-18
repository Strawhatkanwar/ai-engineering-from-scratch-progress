'''
Here i will create sampling function from scratch and apply to sample from estimated distribution as we do in nueral networks.
So Here build various statistical sampling methods(Rejetion, importance, MC, MCMC, Gibbs) and LLM decoding stategies like
temperature, top-k, top-p. 
'''

import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()  # getting percentile between 0 and 1
    return -math.log(u) / lam

lam = 3
samples_expo = [sample_exponential_inverse_cdf(lam) for _ in range(10000)]
mean_sam = sum(samples_expo) / len(samples_expo)
print(f"checking if mean of sample equals 1/lambda :{mean_sam:.2f} {1/lam:.2f}")


## Rejection sampling

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
    
a = 0.5
b = 2.0
#underlying normal parameters
mu = 0
sigma = 1

# target pdf: Truncated normal formula
# Total area kept under the regular normal curve between a and b
area_kept = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)

def target_pdf(x):
    
    if x < a or x > b:
        return 0.0
    return stats.norm.pdf(x, mu, sigma) / area_kept

def proposal_sam():
    return random.uniform(a ,b)

def proposal_pdf(x):
    return 1.0 / (b - a)

max_target = target_pdf(a)
M = max_target / proposal_pdf(a)

samples = [rejection_sample(target_pdf, proposal_sam, proposal_pdf, M) for _ in range(5000)]

plt.hist(samples, bins=30, density=True, alpha=0.6, color='gold', edgecolor='darkgoldenrod')
x_axis = np.linspace(a, b, 200)
y_axis = [target_pdf(val) for val in x_axis]
plt.plot(x_axis, y_axis, color='darkorange', linewidth=2.5, label='True Truncated PDF')

plt.title(f'Rejection Sampling Success!\nTruncated Normal between {a} and {b}')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.savefig()

## importance sampling
def importance_sampling(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n

def f(x):
    return x ** 2  # estimate E[X**2]

mu = 2.0
sigma = 1.5

# defining a wide proposal
a = mu -5 * sigma
b = mu + 5 * sigma

def normal_distribution_target(x):
    return stats.norm.pdf(x, mu, sigma)

def uniform_prposal_pdf(x):
    return 1.0 / (b - a)

def proposal_sample():
    return sample_uniform(a, b)

n_samples = 5000

estimated = importance_sampling(f, normal_distribution_target, uniform_prposal_pdf, proposal_sample, n_samples)
print(estimated)
true = (mu**2) + (sigma**2)
#print(true)  # the estimated 7.37 is close to original 6.25

## monte carlo estimation of pi

def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n



### Exercieses

## exercise 1: Implement inverse CDF sampling for the Cauchy distribution. The CDF is F(x) = 0.5 + arctan(x)/pi. 
# Generate 10,000 samples and plot the histogram against the true PDF. Notice the heavy tails (extreme values far from center).

def sample_cauchy(x):
    return 0.5 + math.atan(x) / math.pi

def true_pdf_cauchy(x):
    return 1.0/ (math.pi * (1 + x**2))

def inverse_cdf_cauchy(p, x0=0 , gamma=1):
    return x0 + gamma * math.tan(math.pi * (p - 0.5))

n_samples = 10000

# generate 10000 flat unknown random percentiles (probabiliteis between 0 and 1)
u_percentiles = [random.random() for _ in range(n_samples)]

# passing those percentiles through our Inverse CDF to get actual samples
samples = [inverse_cdf_cauchy(p) for p in u_percentiles]

plot_range = (-5, 5)
plt.hist(samples, bins=100, range=plot_range, density=True, alpha=0.6, color='r', edgecolor='orange')

# drawing the true_pdf line over the top

x_axis = np.linspace(plot_range[0], plot_range[1], 500)
y_axis = [true_pdf_cauchy(val) for val in x_axis]

plt.plot(x_axis, y_axis, color='g', linewidth=2.5, label="True cauchy pdf")

plt.title("The wild cauchy distribution\n Generated via our inverse cdf function")
plt.xlabel("X")
plt.ylabel("Density")
plt.legend()
plt.savefig("cauchy.png")
print("figure save to {os.getcwd()} named as cauchy.png")

## Exercise2: Use rejection sampling to generate samples from a Beta(2, 5) distribution using a 
# Uniform(0, 1) proposal. Plot the accepted samples against the true Beta PDF. What is the theoretical acceptance rate?

plt.clf()
def target_pdf(x):
    if x < 0 or x > 1: 
        return 0.0
    return stats.beta.pdf(x, 2, 5)

def proposal_sample_beta():
    return random.random()

def proposal_pdf_beta(x):
    return 1.0

M = target_pdf(0.2) / proposal_pdf(0.2)  # the peak of curve is our M bound.

accepted_sam = [rejection_sample(target_pdf, proposal_sample_beta, proposal_pdf_beta, M) for _ in range(10000)]

plt.hist(accepted_sam, density=True, bins=40, range=(0, 1), alpha=0.6, color='coral', edgecolor="chocolate")

#drawing a true smooth beta line over the top between 0 and 1. 
x_axis = np.linspace(0, 1, 500)
# print(x_axis)
y_axis = [target_pdf(val) for val in x_axis]

plt.plot(x_axis, y_axis, color="orange", linewidth=2.5, label="true beta(2, 5) PDf")
plt.title("Rejection sammpling on Organic Beta Curve\nfrom a flat uniform proposal")
plt.xlabel("X")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.legend()
plt.savefig("Rejectionsampling.png")


## Exercise 3: Estimate the integral of sin(x) from 0 to pi using Monte Carlo with 1,000, 10,000, and 100,000 samples. 
# Compare the error at each level. Verify that the error scales as O(1/sqrt(N)).

# estimate I = sinx dx over domain D(0, pi), i have sample x1, x2, ... xn from D. so estimation would be (volume of D / N) * sum(g(xi))

sample = [1000, 10000, 100000]
# first approach.
true_val = 2.0
def est_sin(n):
    total_estimate = []
    for i in n:
        total_height = 0
        for j in range(i):
            x = random.uniform(0, math.pi)
            total_height += math.sin(x)
        # Formula (width of Domain /n) * sum of heights
        estimate = (math.pi / i) * total_height
        total_estimate.append(estimate)
    return  total_estimate

# fir_result = est_sin(sample)

# sin(x) intergral under the bound 0 and pi is -cos(x) and input the bounds gives area 2. so we have to hit darts in area restrictive
# under y <= sin(x) and y can be betwee 0 and 1 and x can be between 0 and pi. 

def mc_sin(n):
    sam_total = []
    for i in n:
        sam_hits = 0

        for j in range(i):
            x = random.uniform(0, math.pi)
            y = random.random()  # height of bounding box
            restrict = math.sin(x)
            ## restriction
            if y <= restrict:
                sam_hits += 1  # let's count the hits that meet our condition.
        sam_integral = math.pi * (sam_hits / i)
        sam_total.append(sam_integral)
    return sam_total

mean_mc_result = est_sin(sample)
dart_results = mc_sin(sample)

print("---------Approach 1 restults----------------------------------------")
for idx, N in enumerate(sample):
    est = mean_mc_result[idx]
    error = abs(est - true_val)
    print(f"N= {N:7d} | Estimate: {est:5f} | abs error: {error:5f}")

print("------------darts results --------------------------------------------")

for idx, N in enumerate(sample):
    est = dart_results[idx]
    error = abs(est - true_val)
    print(f"N = {N:7d} | Estimate: {est:.5f} | Abs Error: {error:.5f}")

# the O(1/N^0.5) rule states as we increase the the sample size by 10 our absolute error tends to decrease by N**0.5. althoug
# it doesn't actually do it here that is because of the random ness of generated sample, maybe it hit the point in first 1k samples.

## Exercise4: Implement Metropolis-Hastings to sample from a 2D distribution p(x, y) proportional to 
# exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8*y) / 2). Plot the samples and the chain trajectory. 
# Experiment with different proposal standard deviations.
# so here proposal distribution will be gaussian (0, sigma^2) and xnew = x_current + guassian(0, sigma^2)


def log_target_pdf(x, y):
    return -(x**2 * y**2 + x**2 + y**2 - 8*x - 8*y) / 2

def metropolis_hastings_2d(n_samples, sigma, start_pos=(0.0, 0.0)):
    current_x, current_y = start_pos
    current_log_p = log_target_pdf(current_x, current_y)

    trajectory = [(current_x, current_y)]
    accepted_count = 0

    for _ in range(n_samples):

        # Proposing a new 2d step using the proposal standard deviation(sigma)
        proposed_x = current_x + random.gauss(0, sigma)
        proposed_y = current_y + random.gauss(0, sigma)

        # lot probabilitty at proposed position
        proposed_log_p = log_target_pdf(proposed_x, proposed_y)

        # caculating acceptance probability ration in log-space: log(p_new / p_old) = log(p_new) - log(p_old)
        log_acceptance_ratio = proposed_log_p - current_log_p

        if math.log(random.random()) < log_acceptance_ratio:
            current_x, current_y = proposed_x, proposed_y
            current_log_p = proposed_log_p

            accepted_count += 1
        trajectory.append((current_x, current_y))
    acceptance_rate = accepted_count / n_samples
    return trajectory, acceptance_rate

n_samples = 1000
test_sigma = [0.1, 0.5, 2.0, 5.0]

for sigma in test_sigma:
    chain, rate = metropolis_hastings_2d(n_samples, sigma)
    print(f"sigma ={sigma:<4} | acceptance rate: {rate * 100:.1f}% | final Pos: ({chain[-1][0]:.2f}, {chain[-1][1]:.2f})")

    #chain, rate = metropolis_hastings_2d(n_samples, 0.5)
    chain_np = np.array(chain)

    x_grid = np.linspace(-1, 6, 200)
    y_grid = np.linspace(-1, 6, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    # Evaluate log-target function across the grid mesh
    Z = np.exp(-(X**2 * Y**2 + X**2 + Y**2 - 8*X - 8*Y) / 2)

    plt.figure(figsize=(8, 6))

    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(label="Probability Density")

    # 2. Plot the trajectory path of the walk

    plt.plot(chain_np[:, 0], chain_np[:, 1], color='red', linewidth=0.8, alpha=0.7, label='MH Chain Path')
    plt.scatter(chain_np[0, 0], chain_np[0, 1], color='white', edgecolor='black', s=100, zorder=5, label='Start (0,0)')
    plt.scatter(chain_np[-1, 0], chain_np[-1, 1], color='cyan', edgecolor='black', s=100, zorder=5, label='Final Position')

    plt.title(f'Metropolis-Hastings Trajectory ($\sigma$ = {sigma})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f"mcmcplot_sigma_{sigma}.png")
    plt.close()
# sigma at 0.5 gives the best result of acceptance of around 56% of samples with final position at (3.38, -0.12)
# which seems to be a low for this function.

# chain, rate = metropolis_hastings_2d(n_samples, 0.5)
# print(chain)

## Exercise 5: Build a complete text generation demo: given a vocabulary of 10 words with logits, generate sequences of 20 tokens 
# using (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Compare the diversity of outputs across 5 runs.

vocab = ["you", "work", "people", "good", "India", "help", "hard", "loving", "heart", "smile"]
logits = [1.0, 9.3, 2.0, 4.8, 8.0, 1.1, 7.0, 2.8, 3.0, 11.0]
tokens_seq = 20



def softmax(logits):
    logits_shifted = [z - max(logits) for z in logits]
    exps = [math.exp(z) for z in logits_shifted]
    total = sum(exps)
    return [e / total for e in exps]

def sample_from_probs(probs):
    u = random.random()
    cumsum = 0.0
    for idx, p in enumerate(probs):
        cumsum += p
        if u <= cumsum:
            return idx
        
    return len(probs) - 1


def greedy(logits):
    return logits.index(max(logits))

def temperature_sam(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)

def sample_top_k(logits, k):
    # logits in descending
    indexed_logits = list(enumerate(logits)) # [(0, 1.0), (1, 2.3),....]
    indexed_logits.sort(key=lambda item: item[1], reverse=True)
    
    # keeping top-k
    top_k = indexed_logits[:k]
    # renormalizing
    top_k_logits = [item[1] for item in top_k]
    top_k_probs = softmax(top_k_logits)

    selected_idx = sample_from_probs(top_k_probs)
    return top_k[selected_idx][0]
    
def sample_top_p(logits, p):
    probs = softmax(logits)
    indexed_probs = list(enumerate(probs))  # [(0, prob1), (1, prob2), ....]

    cumulative_p = 0.0
    selected = []

    # keep adding the highest probability words until we cross threshold p
    for item in indexed_probs:
        selected.append(item)
        cumulative_p +=item[1]
        if cumulative_p >= p:
            break
    # renormalize the probabilities to our selected pool
    pool_probs = [item[1] for item in selected]
    pool_total = sum(pool_probs)
    pool_probs_normalized = [prob / pool_total for prob in pool_probs]
    
    selected_pool_idx = sample_from_probs(pool_probs_normalized)
    return selected[selected_pool_idx][0]

def generate_sequences(strategy_f, **kwargs):
    sequence = []
    for _ in range(20):
        idx = strategy_f(logits, **kwargs) if kwargs else strategy_f(logits)
        sequence.append(vocab[idx])
    return " ".join(sequence)

print("--- (A) GREEDY SEQUENCE ---")
print(generate_sequences(greedy))

print("\n--- (B) TEMPERATURE = 0.7 ---")
print(generate_sequences(temperature_sam, temperature=0.7))

print("\n--- (C) TOP-K = 3 ---")
print(generate_sequences(sample_top_k, k=3))

print("\n--- (D) TOP-P = 0.9 ---")
print(generate_sequences(sample_top_p, p=0.9))




