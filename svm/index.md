---
layout: layout
title: Support Vector Machines
---
# Introduction

An SVM is similar to a perceptron, in that it separates a dataset with two classes using a linear boundary. However, the SVM aims to find the linear boundary that is located as far as possible from the points in the dataset. We also cover the kernel method, which is useful when used in conjunction with an SVM, and it can help classify datasets using highly nonlinear boundaries. An SVM classifier uses two parallel lines instead of one line. The goal of the SVM is twofold; it tries to classify the data correctly and also tries to space the lines as much as possible. 

# Hyperplanes

The equation 

$$
\beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p = 0 \tag{1}
$$

defines a $p$-dimensional hyperplane in the sense that if a point $X = (X_1, X_2, \ldots, X_p)^T$ in $p$-dimensional space satisfies Eq. 1, then $X$ lies on the hyperplane.

Now, suppose that $X$ does not satisfy Eq. 1; rather,
$$
\beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p > 0,
$$
then this tells us that $X$ lies to one side of the hyperplane. On the other hand, if
$$
\beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p < 0,
$$
then $X$ lies on the other side of the hyperplane. So we can think of the hyperplane as dividing $p$-dimensional space into two halves. One can easily determine on which side of the hyperplane a point lies by simply calculating the sign of the left-hand side of Eq. 1.

# The Maximal Margin Classifier

In general, if our data can be perfectly separated using a hyperplane, then there will in fact exist an infinite number of such hyperplanes. This is because a given separating hyperplane can usually be shifted a tiny bit up or down, or rotated, without coming into contact with any of the observations. In order to construct a classifier based upon a separating hyperplane, we must have a reasonable way to decide which of the infinite possible separating hyperplanes to use. 

A natural choice is the maximal margin hyperplane (also known as the optimal separating hyperplane), which is the separating hyperplane that is farthest from the training observations. That is, we can compute the (perpendicular) distance from each training observation to a given separating hyperplane; the smallest such distance is the minimal distance from the observations to the hyperplane, and is known as the margin. The maximal margin hyperplane is the separating hyperplane for which the margin is largest—that is, it is the hyperplane that has the farthest minimum distance to the training observations. We can then classify a test observation based on which side of the maximal margin hyperplane it lies. This is known as the maximal margin classifier. We hope that a classifier that has a large maximal margin classifier margin on the training data will also have a large margin on the test data, and hence will classify the test observations correctly. 

# Construction of the Maximal Margin Classifier

We now consider the task of constructing the maximal margin hyperplane based on a set of $$ n $$ training observations $$ x_1, \ldots, x_n \in \mathbb{R}^p $$ and associated class labels $$ y_1, \ldots, y_n \in \{-1, 1\} $$. Briefly, the maximal margin hyperplane is the solution to the optimization problem:

$$
\underset{\beta_0, \beta_1, \ldots, \beta_p, M}{\text{maximise}} \ M \tag{2}
$$

subject to

$$
\sum_{j=1}^{p} \beta_j^2 = 1, \tag{3}
$$

$$
y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) \geq M \quad \forall i = 1, \ldots, n. \tag{4}
$$

The constraint in (4) guarantees that each observation will be on the correct side of the hyperplane, provided that $$ M $$ is positive. (Actually, for each observation to be on the correct side of the hyperplane, we would simply need $$ y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) > 0 $$, so the constraint in (4) in fact requires that each observation be on the correct side of the hyperplane, with some cushion, provided that $$ M $$ is positive.) Second, note that (3) is not really a constraint on the hyperplane, since if $$ \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip} = 0 $$ defines a hyperplane, then so does $$ k(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) = 0 $$ for any $$ k \neq 0 $$. However, (3) adds meaning to (4); one can show that with this constraint the perpendicular distance from the $$ i $$th observation to the hyperplane is given by $$ y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) $$.

Therefore, the constraints (3) and (4) ensure that each observation is on the correct side of the hyperplane and at least a distance $$ M $$ from the hyperplane. Hence, $$ M $$ represents the margin of our hyperplane, and the optimization problem chooses $$ \beta_0, \beta_1, \ldots, \beta_p $$ to maximize $$ M $$. This is exactly the definition of the maximal margin hyperplane! 

You can think of an SVM classifier as fitting the widest possible street between the classes. This is called _large margin classification_. Notice that adding more training instances “off the street” will not affect the decision boundary at all: it is fully determined (or “supported”) by the instances located on the edge of the street. These instances are called the support vectors. These observations are known as support vectors, since they “support” the maximal margin hyperplane in the sense that if these points were moved slightly then the maximal margin hyperplane would move as well. Interestingly, the maximal margin hyperplane depends directly on the support vectors, but not on the other observations: a movement to any of the other observations would not affect the separating hyperplane, provided that the observation’s movement does not cause it to cross the boundary set by the margin. Although the maximal margin classifier is often successful, it can also lead to overfitting when $p$ is large.

Note: SVMs are sensitive to the feature scales.

# Soft Margin Classifiers or Support Vector Classifiers

If we strictly impose that all instances must be off the street and on the right side, this is called _hard margin classification_. There are two main issues with hard margin classification. First, it only works if the data is linearly separable. Second, it is sensitive to outliers. The fact that the maximal margin hyperplane is extremely sensitive to a change in a single observation suggests that it may have overfit the training data. In this case, we might be willing to consider a classifier based on a hyperplane that does not perfectly separate the two classes, in the interest of

- Greater robustness to individual observations, and
- Better classification of most of the training observations.

That is, it could be worthwhile to misclassify a few training observations in order to do a better job in classifying the remaining observations. The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side). This is called _soft margin classification_ (The margin is soft because it can be violated by some of the training observations). 

Note: Unlike Logistic Regression classifiers, SVM classifiers do not output probabilities for each class.

# Construction of the Support Vector Classifier

The support vector classifier classifies a test observation depending on which side of a hyperplane it lies. The hyperplane is chosen to correctly separate most of the training observations into the two classes but may misclassify a few observations. It is the solution to the optimization problem:

$$
\underset{\beta_0, \beta_1, \ldots, \beta_p, \xi_1, \ldots, \xi_n, M}{\text{maximise}} \ M \tag{5}
$$
subject to

$$
\sum_{j=1}^{p} \beta_j^2 = 1, \tag{6}
$$

$$
y_i(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}) \geq M(1 - \xi_i), \tag{7}
$$

$$
\xi_i \geq 0, \tag{8}
$$

$$
\sum_{i=1}^{n} \xi_i \leq C, \tag{9}
$$

where $$ C $$ is a nonnegative tuning parameter. $$ \xi_1, \ldots, \xi_n $$ are slack variables that allow individual observations to be on the wrong side of the margin or the hyperplane. Once we have solved (5)--(9), we classify a test observation $$ x^* $$ as before, by simply determining on which side of the hyperplane it lies. That is, we classify the test observation based on the sign of $$ f(x^*) = \beta_0 + \beta_1x_1^* + \ldots + \beta_p x_p^* $$.

The slack variable $$ \xi_i $$ tells us where the $$ i $$th observation is located, relative to the hyperplane and relative to the margin. If $$ \xi_i = 0 $$, then the $$ i $$th observation is on the correct side of the margin. If $$ \xi_i > 0 $$, then the $$ i $$th observation is on the wrong side of the margin, and we say that the $$ i $$th observation has violated the margin. If $$ \xi_i > 1 $$, then it is on the wrong side of the hyperplane.


We now consider the role of the tuning parameter $$ C $$. In (9), $$ C $$ bounds the sum of the $$ \xi $$'s, and so it determines the number and severity of the violations to the margin (and to the hyperplane) that we will tolerate. We can think of $$ C $$ as a budget for the amount that the margin can be violated by the $$ n $$ observations. If $$ C = 0 $$, then there is no budget for violations to the margin, and it must be the case that $$ \xi_1 = \ldots = \xi_n = 0 $$, in which case (5)--(9) simply amounts to the maximal margin hyperplane optimization problem (2)--(4). (Of course, a maximal margin hyperplane exists only if the two classes are separable.) For $$ C > 0 $$, no more than $$ C $$ observations can be on the wrong side of the hyperplane because if an observation is on the wrong side of the hyperplane, then $$ \xi_i > 1 $$, and (9) requires that
$$
\sum_{i=1}^{n} \xi_i \leq C.
$$
As the budget $$ C $$ increases, we become more tolerant of violations to the margin, and so the margin will widen. Conversely, as $$ C $$ decreases, we become less tolerant of violations to the margin, and so the margin narrows.

In practice, $$ C $$ is treated as a tuning parameter that is generally chosen via cross-validation. As with the tuning parameters that we have seen throughout this book, $$ C $$ controls the bias-variance trade-off of the statistical learning technique. When $$ C $$ is small, we seek narrow margins that are rarely violated; this amounts to a classifier that is highly fit to the data, which may have low bias but high variance. On the other hand, when $$ C $$ is larger, the margin is wider, and we allow more violations to it; this amounts to fitting the data less hard and obtaining a classifier that is potentially more biased but may have lower variance.

It turns out that only observations that either lie on the margin or that violate the margin will affect the hyperplane and hence the classifier obtained. In other words, an observation that lies strictly on the correct side of the margin does not affect the support vector classifier! Changing the position of that observation would not change the classifier at all, provided that its position remains on the correct side of the margin. Observations that lie directly on the margin or on the wrong side of the margin for their class are known as support vectors. These observations do affect the support vector classifier. The fact that only support vectors affect the classifier is in line with our previous assertion that $$ C $$ controls the bias-variance trade-off of the support vector classifier. When the tuning parameter $$ C $$ is large, then the margin is wide, many observations violate the margin, and so there are many support vectors. In this case, many observations are involved in determining the hyperplane. This classifier has low variance (since many observations are support vectors) but potentially high bias. In contrast, if $$ C $$ is small, then there will be fewer support vectors, and hence the resulting classifier will have low bias but high variance. 
