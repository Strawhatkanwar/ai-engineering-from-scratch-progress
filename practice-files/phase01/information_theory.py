'''
This python script is used for applying the information theory concepts and learn from scratch.
Here i did Entropy, cross_entropy, kl_divergence, mutual information and applied them in real ML examples 
Here i learned why crossentrpy loss works for classification tasks and why it's a good choice.
to learn the concepts in detail.
'''

import math
import random
import numpy as np

def information_content(p, base=2):
    if p <= 0 or p >= 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy: {entropy(fair_coin):.4f}bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f}bits")
print(f"fair die entropy: {entropy(fair_die):.4f}bits")

## Cross-Entropy and KL Divergence

def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
        total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

# examples

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"entropy for true dist: {entropy(true_dist):.4f}bits")
print(f"Cross entropy good model: {cross_entropy(true_dist, good_model):.4f}bits")
print(f"Cross entropy bad model: {cross_entropy(true_dist, bad_model):.4f}bits")
print(f"KL Div. good model: {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL Div. bad model: {kl_divergence(true_dist, bad_model):.4f}bits")


## Cross-entropy As classification loss

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:          {logits}")
print(f"Softmax:         {[f'{p:.4f}' for p in probs]}")
print(f"True class:      {true_class}")
print(f"Loss:            {loss:.4f}nats")
print(f"Perplexity:      {math.exp(loss):.2f}")


## Cross entropy equal negative log-likelihood
random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

# zip feeds row by row.

print(f" Cross-entropy loss: {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.6f}")


## Mutual information

def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0

    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)

    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")


## same stuff using Numpy

def np_entropy(p):
    np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])

print(f"Entropy: {np_entropy(true):.4f}nats")
print(f"Cross-ent: {np_cross_entropy(true, pred):.4f}nats")
print(f"KL div:    {np_kl_divergence(true, pred):.4f}nats")


# this we build here from scratch is what torch.nn.CrossEntrpyLoss() does internally

print("-"*25 + "EXERCISES" + "-"*25)

## Exercise1:Compute the entropy of the English alphabet assuming uniform distribution (26 letters). 
# Then estimate it using actual letter frequencies. Which is higher and why? 

p_eng_alpha = [1/26] * 26
entropy_english = entropy(p_eng_alpha)
print(f"English alphabet entropy assuming uniform distribution is {entropy_english}")

# actuall letter frequencies
english_frequencies = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609, # A-H
    0.0697, 0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193, # I-P
    0.0009, 0.0599, 0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015, # Q-X
    0.0197, 0.0007                                                 # Y-Z
]
total_prob = sum(english_frequencies)
actual_probs = [p / total_prob for p in english_frequencies]
entropy_freq = entropy(actual_probs)
print(f"Estimated as per actuall letter frequencies is: {entropy_freq}")

# Unifrom distribution entropy is higher. Because uncertainty is more and element of surprise is 
# more. Actuall frequencies intorduce predictability. like lettter E occurs 13% of the time. It's 
# individual information criterion will be very low. Even though other rarer element like Q has 
# more information content individually but it happens so rarely.

## Exercise 2: A model outputs logits [5.0, 2.0, 0.5] for a sample with true class 1. Compute the cross-entropy loss by hand, 
# then verify with your cross_entropy_loss function. What logits would give zero loss?

# with hand it came to be 3.05 and same with my function with good precision. 

true_class = 1
logits = [5.0, 2.0 , 0.5]
#logits = [-500.0, 2.0, -1000.00]

ce_loss = cross_entropy_loss(true_class, logits)
print(ce_loss)

## for loss to be zero the logits at index 0 and 1 should tend towards infinitely negatively larger than our true class. that will
## make the exponential term 0 for them and we will see 0 loss. like with logits as[-500.0, 2.0, -500.0]

## Exercise 3: Show that KL divergence is not symmetric. Pick two distributions P and Q and compute D_KL(P || Q) and D_KL(Q || P). 
# Explain why they differ.

true_dist = true_dist = [0.44, 0.33, 0.23]
model = [0.2, 0.45, 0.35]

kl_p_q = kl_divergence(p=true_dist, q=model)
kl_q_p = kl_divergence(p=model, q=true_dist)

print(f"kl_p_q={kl_p_q} kl_q_p={kl_q_p}")
print(f"Checking if kl_divergence hold symmetricity: {kl_p_q == kl_q_p}")

## Exercise 4: Build a function that computes perplexity for a sequence of token predictions. 
# Given a list of (true_token_index, predicted_logits) pairs, return the perplexity of the sequence.
def perplexity(a):
    n_tokens = len(a)
    c_e = 0.0
    for index, logits in a:
        c_e += cross_entropy_loss(index, logits)
    av_loss = c_e/n_tokens
    return math.exp(av_loss)

a = [(1, [0.1, 2.4, 0.8]), (0, [3.9, 0.4, 1.8]), (2, [0.19, 1.4, 8.8])]

for index, logits in a:
    prob_dist = softmax(logits)
    target_prob = prob_dist[index]
    loss = cross_entropy_loss(index, logits)
    print(f"Target Prob: {target_prob:.4f} | Loss: {loss:.4f} nats")

print(f"the perplexity for the sequence of tokes of predicts of a is: {perplexity(a)}")