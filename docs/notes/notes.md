---
 title: "Notes: Probablisic Graphing Models: Bayes and Hidden Markov Models"
 author: Ramy Rashad
 numbersections: true
 geometry: margin=1.00in

---

# Probability

In this lesson, we will review a few key concepts in probability including:

- Discrete distributions
- Joint probabilities
- Conditional probabilities

The basics are simple. Probabilities are numbers between 0 and 1 that represent
the likelihood of an event occurring. The probability of an event that is
certain to occur is 1. The probability of an event that is certain not to occur
is 0. The probability of an event that has a 50% chance of occurring is 0.5.
The probability of all events that can occur is 1.

## Discrete Distributions

A discrete distribution is a probability distribution that can take on only
discrete values. For example, a coin toss can only result in two possible
outcomes: heads or tails. The probability of heads is 0.5 and the probability
of tails is 0.5. The probability distribution of a coin toss is a discrete
distribution.

When the events are independent, the probability of multiple events occurring
is the product of their probabilities.

## Joint Prababilities

The joint probability of two events is the probability that both events occur.
The joint probability of two events is the product of their probabilities and
is written as:

$P(X,Y) = P(X)P(Y)$

This is the probability of both X AND Y occurring.

### Example: Independence

The probability that three consecutive (independent) coin tosses results in
heads is:

$0.5 \cdot 0.5 \cdot 0.5 = 0.125$.

The probability of there being at least three heads in four coin tosses:

Look at all the possibilities:

- HHHH
- HHHT
- HHTH
- HTHH
- THHH

There are five possibilities. The probability of getting each exact combination
is $1/16$, which is $0.5\cdot 0.5 \cdot 0.5 \cdot 0.5$. You then multiply the
probabilities of all the combinations to get the total probability, thus, the
probability of there being at least three heads in four coin tosses is $5 \cdot
1/16 = 0.3125$.

### Example: Dependent Events

Suppose you have a bias coin, where the probabilities of the second toss
somehow depend on the outcome of the first toss.

Given:
- $P(X_1=H) = 0.5$
- $P(X_2=H|X_1=H) = 0.9$ (a biased coin)
- $P(X_1=T) = 0.5$
- $P(X_2=T|X_1=T) = 0.8$ (a biased coin)

What is the probability of getting two heads in a row?

$P(X_1=H) \cdot P(X_2=H|X_1=H) = 0.5 \cdot 0.9 = 0.45$

What is the probability of getting heads on the second toss?

$P(X_2=H) = P(X_2=H|X_1=H) \cdot P(X_1=H) + P(X_2=H|X_1=T) \cdot P(X_1=T) = 0.9\cdot 0.5 + 0.2\cdot 0.5 = 0.55$

Note that $P(X_2=H|X_1=T) = 1 - P(X_2=T|X_1=T) = 1-0.8=0.2$

### Lessons

Total probability:

$P(Y) = \sum_{X} P(Y|X)P(X)$

$P(\neg X|Y) = 1 - P(X|Y)$

Note: you can never negate your conditional variable Y and assume that the
probabilities add up to one.

### Example: Cancer Test

Let's look at the probability of having cancer and a positive test result.
Let's say

Given:

- $P(C) = 0.01$
- $P(\neg C) = 0.99$
- $P(T|C) = 0.9$
- $P(F|C) = 0.1$
- $P(T|\neg C) = 0.2$
- $P(F|\neg C) = 0.8$

Then:

- $P(T,C) = P(T|C)P(C) = 0.9 * 0.01 = 0.009$
- $P(F,C) = P(F|C)P(C) = 0.1 * 0.01 = 0.001$
- $P(T,\neg C) = 0.198$
- $P(F,\neg C) = 0.002$


## Bayes Rule

Bayes rule is really just the statement of conditional probability, where B is
the evidence (we know about it), and A is what we care about (the hypothesis):

$P(A|B) = \frac{P(A,B)}{P(B)}$

The numerator comes from the joint probability of A and B. The denominator is
the total probability of B.

The numerator can be expanded to give the typical form of Bayes rule:

$P(A|B) = \frac{P(B|A)\cdot P(A)}{P(B)}$

The denominator can be expanded to give the total probability of B:

$P(B) = \sum_{A} P(B|A)\cdot P(A)$

In words, Bayes rule says that the probability of A given B is equal to the
probability of B given A times the probability of A divided by the probability
of B.

$Posterior = \frac{Likelihood \cdot Prior}{Marginal Likelihood}$

![Bayes Rule](./figs/bayes.png)

\newpage 

## Naive Bayes

Naive Bayes is a simple classifier that uses Bayes rule to classify data. It is
naive because it assumes that all features are independent. This is a naive
assumption, but it works well in practice.

Naive Bayes is a supervised machine learning algorithm that can be trained to
classify data into multi-class categories. In the heart of the Naive Bayes
algorithm is the probabilistic model that computes the conditional
probabilities of the input features and assigns the probability distributions
to each of the possible classes.

In this lesson, we will:

- Review the conditional probability and Bayes Rule
- Learn how the Naive Bayes algorithm works

Great Video on Bayes Theorem:
[https://www.youtube.com/watch?v=pQgO1KF90yU&t=304s](https://www.youtube.com/watch?v=pQgO1KF90yU&t=304s)

Bayes Theorem basically says that because you know something about the
situation, you can remove possibilities from your universe. For example, in the
video, there is someone that passes by quickly wearing a red sweater. Since we
know they are wearing a red sweater, we can remove all the people that are not
wearing a red sweater from our universe of possibilities. But when we do that
the total probability of our universe no longer sums to 1. So we have to
normalize the probabilities of the remaining possibilities to sum to 1. This is
what Bayes Theorem does.


![Bayes Thm](./figs/bayes01.png)
![Bayes Thm](./figs/bayes02.png)

### False Positives

False positives are when you predict that something is true when it is not. For
example, if you predict that someone has cancer when they do not, that is a
false positive. Suppose you have a test that is 99% accurate. That means that
if you have cancer, the test will be positive 99% of the time. But if you do
not have cancer, the test will be positive 1% of the time. So if you test
positive, what is the probability that you have cancer? It is not 99%. It is
actually 50%. This is because the probability of having cancer is very low. So
even though the test is 99% accurate, the probability of having cancer is so
low that the probability of a false positive is actually higher than the
probability of a true positive.

It really comes down to a comparison between the accuracy of the test and the
prevalence of the disease in the population. If the prevalence of the disease
is low, then the accuracy of the test has to be very high in order to have a
low false positive rate.

In the following example, the accuracy of the test is 99%, but the probability
of being sick is only 1 in every 10,000 people, or 0.01%. So the probability of
a false positive is actually higher than the probability of a true positive.
The chances of being sick in this case, even though you tested positive with a
test that is 99%, is less than 1%.

![False Positives 1](./figs/false_positives01.png)
![False Positives 2](./figs/false_positives02.png)

### The Naive Bayes Algorithm

We want to know the probability of a categorical class given some input values.
For example, we want to know the probability that an email is spam (or valid)
given the words in the email.

$P(spam|'easy', 'money', 'cheap') \propto P('easy'|spam) \cdot P('money'|spam) \cdot P('cheap'|spam) \cdot P(spam)$
$P(valid|'easy', 'money', 'cheap') \propto P('easy'|spam) \cdot P('money'|not spam) \cdot P('cheap'|valid) \cdot P(valid)$

The fact that we are multiplying the probabilities together is the naive part
of the algorithm. It assumes that the input values are independent. For
example, you cannot assume that the probility of it being 'hot' and 'cold'
outside at the same time are independent (the probability of that would be
zero). But the assumption works well in practice.

The Naive Bayes algorithm is as follows:

- Calculate the prior probability of each class (spam vs valid)
- Calculate the conditional probability of each class given each input value 
- Multiply the conditional probabilities of each input value together, for each
  class
- Multiply the prior probability of each class by the conditional probability
  of each input value for that class
- Normalize the probabilities for each class so that they sum to 1 (divide each
  class probability by the sum of the class probabilities)
- The class with the highest probability is the prediction
- Return the prediction

### Classification vs Regression

Classification is when you are trying to predict a categorical class. For
example, you might be trying to predict whether an email is spam or valid.
Regression is when you are trying to predict a continuous value. For example,
you might be trying to predict the price of a house given its size and
location.

![Supervised Learning](./figs/supervised.png)
