---
layout: post
title: Simple Linear Regression
description: an example of a blog post with some math
tags: formatting math
categories: sample-posts
related_posts: false
permalink: /simple-linear-regression/
---

# Introduction
Let’s say that we have some points that roughly look like they are forming a line. The goal of linear regression is to draw the line $$y = \beta_0 + \beta_1 x$$ that passes as close to these points as possible. It is a very straightforward simple linear approach for predicting a quantitative response $$Y$$ on the basis of a single predictor variable $$X$$. It assumes that there is approximately a linear relationship between X and Y .


# Estimating the Coefficients  by the method of Least-Squares

Let $$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$ be the prediction for $$Y$$ based on the $$i$$th value of $$X$$. Then $$e_i = y_i - \hat{y}_i$$ represents the $$i$$th residual—this is the difference between the $$i$$th observed response value and the $i$th response value that is predicted by our linear model. We define the residual sum of squares (RSS) as:

$$
RSS = e_1^2 + e_2^2 + \ldots + e_n^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \tag{1}
$$

Alternatively, we can express RSS as:

$$
RSS = (y_1 - \hat{\beta}_0 - \hat{\beta}_1 x_1)^2 + (y_2 - \hat{\beta}_0 - \hat{\beta}_1 x_2)^2 + \ldots + (y_n - \hat{\beta}_0 - \hat{\beta}_1 x_n)^2.
$$


Using some calculus, we can show that RSS is minimized when:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}, \tag{2}
$$


where $$\bar{y}$$ is the sample mean of $$y$$, defined as $$\bar{y} \equiv \frac{1}{n}\sum_{i=1}^{n} y_i$$, and $$\bar{x}$$ is the sample mean of $$x$$, defined as $$\bar{x} \equiv \frac{1}{n}\sum_{i=1}^{n} x_i$$.

# Standard Errors for $$\hat{\beta}_0$$ and $$\hat{\beta}_1$$

$$
SE(\hat{\beta}_0)^2 = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2} \right], \quad SE(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}, \tag{3}
$$

where $$\sigma^2 = \text{Var}(\epsilon).$$ In general, $\sigma^2$ is not known but can be estimated from the data. The estimate of $\sigma$ is known as the residual standard error and is given by the formula:

$$
RSE = \sqrt{\frac{RSS}{n - 2}.} \tag{4}
$$

# Confidence Intervals for $$\hat{\beta}_0$$ and $$\hat{\beta}_1$$

The 95% confidence interval for $$\beta_1$$ is-

$$
\hat{\beta}_1 \pm 2 \cdot SE(\hat{\beta}_1) \tag{5}
$$

The 95% confidence interval for $$\beta_0$$ is-

$$
\hat{\beta}_0 \pm 2 \cdot SE(\hat{\beta}_0) \tag{6}
$$

# Evaluating the Model

## Residual Standard Error (RSE)

RSE is the average amount that the response will deviate from the true regression line. It is computed using the formula:

$$
RSE = \sqrt{\frac{1}{n - 2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\frac{1}{n - 2} \cdot RSS} \tag{7}
$$

where RSS was as defined in equation (1).

The RSE is considered  as a measure of the lack of fit of the model $$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$ to the data. If the predictions obtained using the model are very close to the true outcome values—that is, if $$\hat{y}_i \approx y_i$$ for $$i = 1,\ldots,n$$—then RSE will be small, and we can conclude that the model fits the data very well. On the other hand, if $$\hat{y}_i$$ is very far from $$y_i$$ for one or more observations, then the RSE may be quite large, indicating that the model doesn't fit the data well.

## The $$R^2$$ Statistic
One drawback of RSE is that it provides an absolute measure of the lack of fit of the model to the data. But since it is measured in the units of $Y$, it is not always clear what constitutes a good RSE. The $$R^2$$ statistic provides an alternative measure of fit. It takes the form of a proportion—the proportion of variance explained—and so it always takes on a value between 0 and 1, and is independent of the scale of $$Y$$.

To calculate $$R^2$$, we use the formula:

$$
R^2 = 1 - \frac{RSS}{TSS},
$$

where $$TSS \equiv \sum_{i=1}^{n} (y_i - \bar{y})^2$$ is the total sum of squares.


$$TSS$$ measures the total variance in the response $$Y$$, and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, $$RSS$$ measures the amount of variability that is left unexplained after performing the regression. Hence, $$TSS - RSS$$ measures the amount of variability in the response that is explained (or removed) by performing the regression, and $$R^2$$ measures the proportion of variability in $$Y$$ that can be explained using $$X$$. An $$R^2$$ statistic that is close to 1 indicates that a large proportion of the variability in the response has been explained by the regression. A number near 0 indicates that the regression did not explain much of the variability in the response; this might occur because the linear model is wrong, or the inherent error $$\sigma^2$$ is high, or both.


