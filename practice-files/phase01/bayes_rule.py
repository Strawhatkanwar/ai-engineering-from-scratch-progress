"""
Here i will practice and apply bayes rule and related algorithm from scratch
"""
import math
from collections import defaultdict

# for validation our scratch logic with inbuild scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report



def bayes_rule(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

result = bayes_rule(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive) = {result:4f}")

## Naive Bayes classifier from scratch

class NaiveBayes:

    '''
    This class demostrates full pipeline for Naive Bayes: from tokenisation, probability estimation
    with laplacian smoothing, log-space prediction.
    '''

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()
        self.message_length = defaultdict(lambda: defaultdict(int))

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()

            #-----added for exercise 3-----------------------------
            if len(words) <= 4:
                self.message_length["short"][label] += 1
            else:
                self.message_length["long"][label] += 1

            #------------added for exerise 3 finish-----------------
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)
            
    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")

        current_length_feature = "short" if len(words) <= 4 else "long"  # for exercise 3

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)

            # for exercise 3 : calculate and fold in the message length feature probability: log(p(length_feature | class))
            # we also used laplace smoothing here as well. There are 2 lengths cateogory(short/ long),
            # so the denominator adds: self.smoothing * 2
            length_count = self.message_length[current_length_feature][cls]
            # print(f"length count for class {cls} and length feature type {current_length_feature} is {length_count}")
            total_class_docs = self.class_counts[cls]
            # print(f"total class docs for class {cls} is  {total_class_docs}")
            length_prob = (length_count + self.smoothing) / (total_class_docs + self.smoothing * 2)
            # print(f"lenght probability for class {cls} is {length_prob}")
            score += math.log(length_prob)

            # Fold in individual word probabilities : log(p(word | class))
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class


### let train on spam data:

train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_message = [
    "free money waiting for you",
    "meeting rescheduled to Friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_message:
    print(f" '{msg}' --> {classifier.predict(msg)}")


## let's inspect the learned probabilities

def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse = True)
    for word, prob in sorted_words[:n]:
        print(f"   {word}: {prob:.4f}")

print("\nTop spam words:")
show_top_words(classifier, "spam")
print("\nTop ham words:")
show_top_words(classifier, "ham")


## now using scikit-learn production-ready naive Bayes implementation:

# the algorithm here is same. CountVectorizer handles tokenisation and vocabulary building. MultinomialNB handles smoothing 
# and log-probabilities internally. Our from-scratch version does same in 40 lines of code.

vectorizor = CountVectorizer()
X_train = vectorizor.fit_transform(train_docs)
clf = MultinomialNB()
clf.fit(X_train, train_labels)

X_test = vectorizor.transform(test_message)
predictions = clf.predict(X_test)
for msg, pred in zip(test_message, predictions):
    print(f" '{msg}' --> {pred}")

print("-"* 25 + "Exercieses" + "-" * 25)

## Exercise 1: Multiple tests. A patient tests positive twice on independent tests (both 99% accurate, 
# disease prevalence 1 in 10,000). What is P(sick) after both tests? Use the posterior from the first 
# test as the prior for the second.

p_sick_1 = bayes_rule(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
p_sick_2 = bayes_rule(prior=p_sick_1, likelihood=0.99, false_positive_rate=0.01)
print(p_sick_1)
print(p_sick_2)

## Exercise 2: Smoothing impact. Run the spam classifier with smoothing values of 0.01, 0.1, 1.0, and 10.0. 
# How do the top word probabilities change? What happens with smoothing=0 and a word that appears only in ham?

nb_1 = NaiveBayes(0.01)
nb_2 = NaiveBayes(0.1)
nb_3 = NaiveBayes()
nb_4 = NaiveBayes(10.0)
nb_5 = NaiveBayes(0.0)

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]


nb_1.train(train_docs, train_labels)
nb_2.train(train_docs, train_labels)
nb_3.train(train_docs, train_labels)
nb_4.train(train_docs, train_labels)
nb_5.train(train_docs, train_labels)

test_message = [
    "free money waiting for you",
    "meeting rescheduled to Friday",
    "you won a free prize",
    "please review the attached report",
]


classifiers = [nb_1, nb_2, nb_3, nb_4, nb_5]

for clf in classifiers:
    print(f"testing smoothing = {clf.smoothing}")
    print("Top Spam Words:")
    show_top_words(clf, "spam", n = 3)
    print("top Ham words")
    show_top_words(clf, "ham", n = 3)

    # making prediction
    print("\nPredictions")
    for msg in test_message:
        try:
            pred = clf.predict(msg)
            print(f"  '{msg}' --> {pred}")
        except ValueError as e:
            print(f"  '{msg}' --> Crashed ({e})")

## The probability of both spam and ham words drop with more laplacian smoothing, best was the 0.01 and gradual decrease with 
# 10x increase in smoothing, denominator terms explodes faster so lowering the probability. with smoothing 0, any word in test
# data that wasn't seen in training for that specific class gets a probaility of exactly 0.0. so with smoothing 0.0 our code crashes
# becaues mat.log(0.0) is math error.



## exercise 3: Add features. Extend the NaiveBayes class to also use message length (short/long) as a feature 
# alongside word counts. Estimate P(short|spam) and P(short|ham) from the training data and fold it into the prediction score.
test_message = [
    "free money waiting for you",
    "meeting rescheduled to Friday",
    "you won a free prize",
    "please review the attached report",
]

nb = NaiveBayes(0.01)
nb.train(train_docs, train_labels)

# predictions with short/ long as a feature inside folded in. 

for msg in test_message:
    pred = nb.predict(msg)
    print(f"'{msg}' --> Predicted Class: {pred}")

## Exercise 4: MAP by hand. Given observed data (7 heads in 10 coin flips), 
# compute the MAP estimate of the bias using a Beta(2,2) prior. Compare it to the MLE estimate (7/10).

# i did it pen and paper and found MLE to be 0.7 and MAP using formula alpha_post = 2 + 7 = 9 and Beta_post = 2 + (10-7) = 5
# so our updated belief distribution is Beta[9, 5] and posterior is 2/3 = 0.667. it pushes our estimate more toward 0.5 because
# we start with uniform prior.