---
layout: layout
title: Naive Bayes and LDA
---
# Introduction

Consider the problem of predicting a label $$y$$ on the basis of a vector of features $$x = (x_1, \ldots, x_d)$$. Let $f_k(x) \equiv P[X = x \ \vert \ Y = k]$ denote the density function of $X$ for an observation that comes from the $k$th class. In other words, $f_k(x)$ is relatively large if there is a high probability that an observation in the $k$th class has $X \approx x$, and $f_k(x)$ is small if it is very unlikely that an observation in the $k$th class has $X \approx x$. Let $\pi_k$ represent the overall or prior probability that a randomly chosen observation comes from the $k$th class; this is the probability that a given observation is associated with the $k$th category of the response variable $Y$. Also, let $p_k(X) = P(Y = k\ \vert \ X)$. We refer to $p_k(x)$ as the posterior probability that an observation $X = x$ belongs to the $k$th class. That is, it is the probability that the observation belongs to the $k$th class, given the predictor value for that observation.

Then Bayes' theorem states that:

$$
p_k(X) = P(Y = k\ \vert \ X = x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^{K} \pi_l f_l(x)}. \tag{1}
$$

The Bayes optimal classifier is given by- 

$$
\begin{eqnarray}
h_{\text{Bayes}}(x) & = & \underset{k}{\text{argmax}} \ p_k(X) \\
& = & \underset{k}{\text{argmax}} \ \pi_k f_k(x) \tag{2}
\end{eqnarray}
$$

The Bayes classifier, which classifies an observation to the class for which $p_k(X)$ is largest, has the lowest possible error rate out of all classifiers. This is, of course, only true if the terms in Eq. 1 are all correctly specified.

To describe the probability function $$p_k(X)$$, we need $$k^d$$ parameters. This implies that the number of examples we need grows exponentially with the number of features. In general, estimating $\pi_k$ is easy if we have a random sample of $Y$s from the population: we simply compute the fraction of the training observations that belong to the $k$th class i.e. 
$$\pi_k = \hat{\pi}_k = \frac{n_k}{n}$$, where $n_k$ is the number of observations in class $k$. However, estimating $f_k(X)$ tends to be more challenging, unless we assume some simple forms for these densities. Therefore, if we can find a way to estimate $f_k(X)$, then we can develop a classifier that approximates the Bayes classifier. 

Suppose that $\hat{f}_k(x)$ is a reasonably good estimator of $f_k(x)$. Then we could approximate the Bayes rule in Eq. 2 by-

$$
h_{\text{Bayes}}(x) = \underset{k}{\text{argmax}} \ \hat{\pi}_k \hat{f}_k(x)
$$

In the Naive Bayes approach, we make the (rather naive) generative assumption that given the label, the features are independent of each other. That is,

$$
P[X = x \ \vert \ Y = y] = \prod_{i=1}^d P[X_i = x_i \ \vert \ Y = y].
$$

With this assumption and using the Bayes rule, the Bayes optimal classifier can be further simplified:

$$
\begin{eqnarray}
h_{\text{Bayes}}(x) & = & 
\underset{y \in \{0,1\}}{\text{argmax}} \ \ P[Y = y \ \vert \ X = x] \nonumber \\
& = & \underset{y \in \{0,1\}}{\text{argmax}} \ \  P[Y = y]P[X = x \ \vert \ Y = y] \nonumber \\
& = & \underset{y \in \{0,1\}}{\text{argmax}} \ \ P[Y = y] \prod_{i=1}^d P[X_i = x_i \ \vert \ Y = y] \tag{2}
\end{eqnarray}
$$

That is, now the number of parameters we need to estimate is only $$2d + 1$$.

When we also estimate the parameters using the maximum likelihood principle, the resulting classifier is called the Naive Bayes classifier.

# Linear Discriminant Analysis

As in the Naive Bayes classifier, we consider the problem of predicting a label $$y \in \{0,1\}$$ on the basis of a vector of features $$x = (x_1, \ldots, x_d)$$. But now the generative assumption is as follows. First, we assume that $$P[Y = 1] = P[Y = 0] = 1/2$$. Second, we assume that the conditional probability of $$X$$ given $$Y$$ is a Gaussian distribution. Finally, the covariance matrix of the Gaussian distribution is the same for both values of the label. Formally, let $$\mu_0, \mu_1 \in \mathbb{R}^d$$ and let $$\Sigma$$ be a covariance matrix. Then, the density distribution is given by:

$$
P[X = x\ \vert \ Y = y] = \frac{1}{(2\pi)^{d/2}\ \vert \ \Sigma\ \vert \ ^{1/2}} \exp \left(-\frac{1}{2}(x - \mu_y)^T \Sigma^{-1}(x - \mu_y)\right) \tag{3}
$$

As we have shown in the previous section, using the Bayes rule we can write:

$$
h_{\text{Bayes}}(x) = \underset{y \in \{0,1\}}{\text{argmax}} \ P[Y = y]\ P[X = x\ \vert \ Y = y] \tag{4}
$$

This means that we will predict $$h_{\text{Bayes}}(x) = 1$$ iff:

$$
\log\left(\frac{P[Y = 1]P[X = x\ \vert \ Y = 1]}{P[Y = 0]P[X = x\ \vert \ Y = 0]}\right) > 0.
$$

This ratio is often called the log-likelihood ratio. In our case, the log-likelihood ratio becomes:

$$
\frac{1}{2}(x - \mu_0)^T \Sigma^{-1}(x - \mu_0) - \frac{1}{2}(x - \mu_1)^T \Sigma^{-1}(x - \mu_1).
$$

We can rewrite this as $$w^Tx + b$$ where

$$
w = (\mu_1 - \mu_0)^T \Sigma^{-1} \quad \text{and} \quad b = \frac{1}{2}\left(\mu_0^T \Sigma^{-1}\mu_0 - \mu_1^T \Sigma^{-1}\mu_1 \right) \tag{5}
$$


As a result of the preceding derivation, we obtain that under the aforementioned generative assumptions, the Bayes optimal classifier is a linear classifier. Additionally, one may train the classifier by estimating the parameters $$\mu_0, \mu_1$$, and $$\Sigma$$ from the data, using, for example, the maximum likelihood estimator. With those estimators at hand, the values of $$w$$ and $$b$$ can be calculated as in Equation (5).

---


The kernel density classifier uses a nonparametric kernel density estimator to estimate the density function of $X$ given $Y$, and then plugs the density estimator into the Bayes formula to estimate the Bayes rule. 



To carry out the idea in (12.14), we often use the kernel density estimator (KDE) (Parzen, 1962; Epanechnikov, 1969; Silverman, 1998). Let $X_1, \ldots, X_n$ be a random sample from a $p$-dimensional distribution with density $f(x)$. Then a kernel density estimator of $f(x)$ is given by

$$
\hat{f}_h(x) = \frac{1}{n} \sum_{i=1}^{n} h^{-p}K\left(\frac{X_i - x}{h}\right),
$$

where $K$ is a non-negative density function, called a kernel function, so that $f(x)$ is a legitimate density function. For example, a commonly-used kernel function is the standard Gaussian density. The parameter $h$ is called the bandwidth.

KDE simply distributes the point $X_i$ by a smoothed function:

$$
h^{-p}K\left(\frac{X_i - x}{h}\right)
$$

for a small $h$. See Figure 12.2 for the case with $p = 1$. Typically, the kernel function is fixed, and the bandwidth $h$ is chosen to trade off biases and variances. There are many papers on data-driven choices of the bandwidth for KDE (Sheather and Jones, 1991; Jones, Marron, and Sheather, 1996). Basically, the optimal $h$ depends on the kernel function and the underlying density function.

Now let us go back to equation (12.14). For each sub-sample of class $k$, we apply the kernel density estimator to construct $f_k(x)$ and then use the $\text{argmax}$ rule to predict the class label at $x$. The kernel density classifier is quite straightforward both conceptually and computationally, but it is not recommended when the dimension is 3 or higher, due to the “curse-of-dimensionality”.

The naive Bayes classifier can be viewed as a much simplified kernel density classifier when the dimension is relatively large. The basic idea is very simple. Assume that given the class label, the features are conditionally independent. That is,

$$
f_k(x) = \prod_{j=1}^{p} f_{jk}(x_j) \text{ for all } k.
$$

The above independence assumption drastically simplifies the problem of density estimation. Instead of estimating a $p$-dimensional density function, we now estimate $p$ univariate density functions. Combining (12.14) and (12.19) yields the naive Bayes classifier

$$
\underset{c_k \in C}{\text{argmax}} \ \pi_bk \prod_{j=1}^{p} f_{bjk}(x_j), \tag{12.20}
$$

where $f_{bjk}(x_j)$ is the univariate KDE for variable $X_j$ based on the $k$-th class of data. If $X_j$ is continuous, we can use the kernel density estimator for $f_{bjk}(x_j)$.

Although the conditional independence assumption is very convenient, it is rather naive and too optimistic to be remotely true in reality. Hence one might wonder if the naive Bayes classifier is practically useful at all. Surprisingly, naive Bayes classifiers have worked very well in many complex real-world applications such as text classification (McCallum and Nigam, 1998). A possible explanation is that although individual class density estimates ($\prod_{j=1}^{p} f_{jk}(x_j)$) are poor estimators for the joint conditional density of the predictor vector, they might be good enough to separate the most probable class from the rest.

Another important use of the naive Bayes rule is to create augmented features.

---