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

# Kernel Density Classifier

The kernel density classifier uses a nonparametric _kernel density estimator_ to estimate the density function of $X$ given $Y$, and then plugs the density estimator into the Bayes formula to estimate the Bayes rule. Let $X_1, \ldots, X_n$ be a random sample from a $d$-dimensional distribution with density $f(x)$. Then a kernel density estimator of $f(x)$ is given by

$$
\hat{f}_h(x) = \frac{1}{n} \sum_{i=1}^{n} h^{-d}K\left(\frac{X_i - x}{h}\right), \tag{3}
$$

where $K$ is a non-negative density function, called a kernel function, so that $f(x)$ is a legitimate density function. For example, a commonly-used kernel function is the standard Gaussian density. The parameter $h$ is called the bandwidth. KDE simply distributes the point $X_i$ by a smoothed function:

$$
h^{-d}K\left(\frac{X_i - x}{h}\right)
$$

for a small $h$. Typically, the kernel function is fixed, and the bandwidth $h$ is chosen to trade off biases and variances. Basically, the optimal $h$ depends on the kernel function and the underlying density function. For each sub-sample of class $k$, we apply the kernel density estimator to construct $f_k(x)$ and then use the $\text{argmax}$ rule to predict the class label at $x$. 

<blockquote style="background-color: #FFFFE0; padding: 10px;">
<b>
The kernel density classifier is quite straightforward both conceptually and computationally, but it is not recommended when the dimension is 3 or higher, due to the “curse-of-dimensionality”.
</b>
</blockquote>

# Linear and Quadratic Discriminant Analysis

Linear discriminant analysis (LDA) and quadratic discriminant analysis (QDA) are model-based methods for classification. LDA and QDA models start with the assumption that the conditional distribution of $$X$$ given $$Y$$ is a multivariate normal distribution (also known as Gaussian Distribution) i.e. $$(X\ \vert \ y = k) \sim N(μ_k, Σ_k)$$. Hence,

$$
f_k(x) = \frac{1}{\sqrt{2\pi \ \vert \Sigma_k\ \vert}} \exp\left(-\frac{1}{2}(x - μ_k)^T Σ_k^{-1}(x - μ_k)\right)
$$

Under this assumption, the Bayes rule as in Eq. 2 can be written as

$$
\underset{k}{\text{argmax}} \ \delta_k(x)
$$

where

$$
\delta_k(x) = \log\pi_k - \frac{1}{2}\log\ \vert \ \Sigma_k\ \vert \  - \frac{1}{2}(x - μ_k)^T \Sigma_k^{-1}(x - μ_k) \tag{4}
$$

<blockquote style="background-color: #FFFFE0; padding: 10px;">
<b>
We can see that the Bayes rule is a quadratic function of $x$, hence the name <i>Quadratic Discriminant</i>. This classification rule is very intuitive: except for some constants to reflect prior knowledge, the QDA classifies a point $x$ according to its Mahalanobis distance to the centroids $μ_k$, defined by $\left(x - μ_k\right)^T \Sigma_k^{-1}\left(x - μ_k\right)$. When $\pi_k$ and $\Sigma_k$ are independent of $k$, this is exactly the nearest centroid classification.
</b>
</blockquote>



QDA substitutes the parameters in (4) with the following estimates:

$$
\begin{eqnarray}
\hat{\pi}_k & = & \frac{n_k}{n}, \\
\hat{\mu}_k & = & \frac{1}{n_k} \sum_{Y_i=c_k} X_i, \\
\hat{\Sigma}_k & = & \frac{1}{n_k - 1} \sum_{Y_i=c_k} (X_i - \hat{\mu}_k)(X_i - \hat{\mu}_k)^T,
\end{eqnarray}
$$

where $$n_k$$ is the number of observations in class $$k$$. Thus, the QDA rule is:

$$
\underset{k}{\text{argmax}} \left\{\log \hat{\pi}_k - \frac{1}{2} \log \ \vert \ \hat{\Sigma}_k\ \vert \  - \frac{1}{2}(x - \hat{\mu}_k)^T \hat{\Sigma}_k^{-1}(x - \hat{\mu}_k)\right\}.
$$


LDA uses the additional homogeneous covariance assumption:

$$
\Sigma_k = \Sigma \text{ for all } k.
$$

As a result, the quadratic term $$x^T \hat{\Sigma}_k^{-1} x$$ and the $$\log \vert \hat{\Sigma}_k\vert$$ are independent of $$k$$. Hence, the Bayes rule has a simpler form:

$$
\underset{k}{\text{argmax}} \ \delta_{\text{lda}}^k(x)
$$

where

$$
\delta_{\text{lda}}^k(x) = \mu_k^T \Sigma^{-1} x - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log \pi_k. \tag{5}
$$

Now the Bayes rule is a linear function of $$x$$ and hence a linear discriminant. It is also referred to as Fisher's discriminant analysis. Note that Fisher's original derivation was geometric, not based on the probabilistic LDA model. In applications, it is observed that LDA can perform quite well although the LDA model assumptions are clearly violated. Let $$K$$ be the number of classes. LDA estimates $$\Sigma$$ with the pooled sample covariance:

$$
\hat{\Sigma} = \frac{1}{\sum_{k=1}^{K} (n_k - 1)} \sum_{k=1}^{K} (n_k - 1) \hat{\Sigma}_k.
$$

The LDA rule is:

$$
\underset{k}{\text{argmax}} \left\{\hat{\mu}_k^T \hat{\Sigma}^{-1} x - \frac{1}{2}\hat{\mu}_k^T \hat{\Sigma}^{-1} \hat{\mu}_k + \log \hat{\pi}_k\right\}. \tag{6}
$$

In particular, if there are only two classes $$\{0, 1\}$$, then the rule classifies an observation to Class 1 if and only if:

$$
\log\left(\frac{P[Y = 1]P[X = x\ \vert \ Y = 1]}{P[Y = 0]P[X = x\ \vert \ Y = 0]}\right) > 0.
$$

This ratio is often called the log-likelihood ratio. In our case, this gives:

$$
(x - \hat{\mu}_a)^T \hat{\Sigma}^{-1}(\hat{\mu}_1 - \hat{\mu}_0) + \log \frac{\hat{\pi}_1}{\hat{\pi}_0} > 0. \tag{7}
$$

where $$\mu_a = \frac{\hat{\mu}_0 + \hat{\mu}_1}{2}$$.

If we further assume that, $$\hat{\pi}_0 = \hat{\pi}_1$$, then Eq. 7 can be written as -

$$
\frac{1}{2}(x - \mu_0)^T \Sigma^{-1}(x - \mu_0) - \frac{1}{2}(x - \mu_1)^T \Sigma^{-1}(x - \mu_1) > 0
$$

We can rewrite this as $$w^Tx + b > 0 $$ where

$$
w = (\mu_1 - \mu_0)^T \Sigma^{-1} \quad \text{and} \quad b = \frac{1}{2}\left(\mu_0^T \Sigma^{-1}\mu_0 - \mu_1^T \Sigma^{-1}\mu_1 \right)
$$


# Regularized Discriminant Analysis

When $$p$$ is moderately large, $$\Sigma$$ cannot be accurately estimated for moderate sample size, and the computation of $$\hat{\Sigma}^{-1}$$ can be very unstable. If $$p \geq n$$, $$\hat{\Sigma}$$ is not even full rank. Friedman (1989) suggested regularized discriminant analysis (RDA) which uses the following shrinkage estimators:

$$
\hat{\Sigma}^{\text{rda}}(\gamma) = \gamma \hat{\Sigma} + (1 - \gamma)\frac{\text{tr}(\hat{\Sigma})}{p}\hat{\mathbf{I}}, \quad 0 \leq \gamma \leq 1 \tag{8}
$$

is shrunken toward an identity matrix (up to a scalar) as $$\gamma \rightarrow 0$$ and

$$
\hat{\Sigma}^{\text{rda}}_k(\alpha, \gamma) = \alpha \hat{\Sigma}_k + (1 - \alpha)\hat{\Sigma}^{\text{rda}}(\gamma). \tag{9}
$$

which is shrunken toward a common covariance matrix as $$\alpha \rightarrow 0$$. In practice, $$(α, γ)$$ are chosen from the data by cross-validation. 

# Naive Bayes


The naive Bayes classifier can be viewed as a much simplified kernel density classifier when the dimension is relatively large. The basic idea is very simple. Assume that given the class label, the features are conditionally independent. That is,

$$
f_k(x) = \prod_{j=1}^{d} f_{jk}(x_j) \text{ for all } k.
$$

The above independence assumption drastically simplifies the problem of density estimation. Instead of estimating a $d$-dimensional density function, we now estimate $d$ univariate density functions. With this assumption and using the Bayes rule, the Bayes optimal classifier can be further simplified:

$$
\begin{eqnarray}
h_{\text{Bayes}}(x) & = & 
\underset{k}{\text{argmax}} \ \ P[Y = y \ \vert \ X = x] \nonumber \\
& = & \underset{k}{\text{argmax}} \ \  P[Y = y]P[X = x \ \vert \ Y = y] \nonumber \\
& = & \underset{k}{\text{argmax}} \ \ P[Y = y] \prod_{i=1}^d P[X_i = x_i \ \vert \ Y = y] \\
& = & \underset{k}{\text{argmax}} \ \hat{\pi}_k \prod_{j=1}^{d} \hat{f}_{jk}(x_j)
\end{eqnarray}
$$


where $$\hat{f}_{jk}(x_j)$$ is the univariate KDE for variable $X_j$ based on the $k$-th class of data. If $X_j$ is continuous, we can use the kernel density estimator for $$\hat{f}_{jk}(x_j)$$. That is, now the number of parameters we need to estimate is only $$kd + 1$$. When we also estimate the parameters using the maximum likelihood principle, the resulting classifier is called the Naive Bayes classifier.


Although the conditional independence assumption is very convenient, it is rather naive and too optimistic to be remotely true in reality. Hence one might wonder if the naive Bayes classifier is practically useful at all. Surprisingly, naive Bayes classifiers have worked very well in many complex real-world applications such as text classification. A possible explanation is that although individual class density estimates ($\prod_{j=1}^{d} f_{jk}(x_j)$) are poor estimators for the joint conditional density of the predictor vector, they might be good enough to separate the most probable class from the rest. Another important use of the naive Bayes rule is to create augmented features.
