'''
Here i implement PMF's and PDF's from scratch for different distribution(bernoulli, categorical,
uniform and normal distribution))
Compute expected value, variance and use CLT to explain guassians.
Also implement softmax, cross entropy from scratch and use it in practise 
'''

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(king | Face card) = {p_king_given_face}")


## Pmf and Pdf from scratch

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0/ (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

### expected value and variance

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6

mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)

print(f"Die: E[X] = {mu:.4f}, Var = {var:.4f}, SD = {var**0.5:.4f}")

### step 4: sampling from disribtutions 

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples
    
def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1) * math.cos(2 * math.pi * u2))
        samples.append(mu + sigma * z)
    return samples
    

## softmax and log probabilies.

def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

    
def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]

def joint_to_marginals(joint):
    rows = len(joint)
    cols = len(joint[0])
    marginal_x = [sum(joint[i][j] for j in range(cols)) for i in range(rows)]
    marginal_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
    return marginal_x, marginal_y


def check_independence(joint, marginal_x, marginal_y, tol=1e-9):
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if abs(joint[i][j] - marginal_x[i] * marginal_y[j]) > tol:
                return False
    return True


## central limit theoram

def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages 

print("-"*25 + "exercises" + "-"*25)


## exericise1: Implement inverse transform sampling for the exponential distribution. 
# Verify by sampling 10,000 values and comparing the histogram to the true PDF.

def expo_samples(lam, n_samples=10000):
    sam = []
    for i in range(n_samples):
        X = -math.log(random.random())/lam
        sam.append(X)
    return sam
lam = 3
samples = expo_samples(lam, 10000)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
x_axis = np.linspace(0, max(samples), 1000)
true_pdf = [lam*math.exp(-lam * x) for x in x_axis]
plt.plot(x_axis, true_pdf, 'r-', linewidth=2, label='True PDF formula')
plt.xlabel("value of x")
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

## exercise 2: Build a joint distribution table for two loaded dice. 
# Compute the marginal distributions and check whether the dice are independent.

die_1 = [0.1, 0.1, 0.1, 0.1, 0.2, 0.4]  # load towards the end
die_2 = [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]  # loaded toward 1

independent_tab = [[p1 * p2  for p2 in die_2] for p1 in die_1]
#print(independent_tab)
mx1, mx2 = joint_to_marginals(independent_tab)
#print(mx1, mx2)
print(check_independence(independent_tab, mx1, mx2))

## exercise 3: Compute the cross-entropy loss for a 5-class classifier that outputs 
# logits [2.0, 0.5, -1.0, 3.0, 0.1] when the correct class is index 3. 
# Then verify your answer with PyTorch's nn.CrossEntropyLoss.

logits = [2.0, 0.5, -1.0, 3.0, 0.1]
target_index = 3
loss = cross_entropy_loss(logits, target_index)
print(f"my loss is: {loss:.4f}")

logits = [2.0, 0.5, -1.0, 3.0, 0.1]
logits = torch.tensor([logits])
target_tensor = torch.tensor([target_index])
criterion = torch.nn.CrossEntropyLoss()
loss_torch = criterion(logits, target_tensor)
print(loss, loss_torch)

## exercise 4: Write a function that takes a list of log probabilities and returns the most likely sequence, the total 
# log probability, and the equivalent raw probability. Test it with a sentence of 50 words where each word has probability 0.01.

def log_seq(log_probs):
    total_log_probability = sum(log_probs)

    # converting back to raw probability
    try:
        raw_prob = math.exp(total_log_probability)
    except OverflowError:
        raw_prob = 0.0
    
    return total_log_probability, raw_prob


# testing with 50 word sequence
word_raw_prob = 0.01  # each word has raw probability of 0.1
word_log_prob = math.log(word_raw_prob)
sentence_log_probs = [word_log_prob] * 50

total_log, raw_prob = log_seq(sentence_log_probs)

print(f"Single word log probability: {word_log_prob:.4f}")
print(f"Total sentence log probability: {total_log:.4f}")
print(f"Equivalent raw probability: {raw_prob}")  # extremely less .


