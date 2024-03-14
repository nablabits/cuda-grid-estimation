# CUDA Grid Estimation
![Python version](https://img.shields.io/badge/python-3.9+-green)
![Cuda toolkit](https://img.shields.io/badge/CUDA_toolkit-12.3+-green)
![Compute Capability](https://img.shields.io/badge/compute_capability-7.5+-green)


## TL;DR
A primer on CUDA, c++ when one is familiar with Python's scientific ecosystem (pandas, numpy & scipy)


> [!TIP]
> Open the outline there on the top right corner of the github interface.

## Introduction
### Motivation
At some point last year with all this hype of LLMs and Deep Learning overall I somehow realized that getting acquainted with this whole CUDA thing, which is at the core of the heavy lifting when it comes to train deep learning models, may well be a good idea. In addition, I had no previous experience with `c++` which made this project appealing to me as a first contact point with compiled languages. As I'm quite familiar with python's scientific ecosystem (namely, pandas, numpy, scipy...) I thought that kicking off from there would be efficient to have a guide on what to implement.

### Intended Public
If you are starting with `c++` and familiar with python's scientific ecosystem, this repository may well be a good support to your learning and/or give you some idea about where to start. Just bear in mind that this project is a sort of a crash course on the subject, so be extra careful with the content you find here. More than ever, perform your own parallel searches to discover possible pitfalls or errors I may have overlooked.

### Brief Description
The main goal of the algorithm is to predict using bayesian statistics the parameters of a normal distribution, namely $\mu$ and $\sigma$, out of some set of observations.

Specifically, let's say that you go out there and gather some data about some process. Of course, before inspecting it, you have for whatever reason, a reasonable guess that the process you are investigating is normally distributed. You might want to know the parameters of the distribution because that will give you an insight about how your system might behave in the long run and, crucially, the level of credence that you should put in those parameters, although, we won't compute this latter.

Of course, this is a highly contrived example because the goal is not to learn bayesian statistics but rather to learn CUDA. If you want to learn more about the former, the origin of the algorithm is in the [Think Bayes Book](https://github.com/nablabits/ThinkBayes2/blob/master/notebooks/13-inference.ipynb) by [Allen Downey](https://github.com/AllenDowney) which is a great resource to get started in the topic when one is familiar with Python.

#### Quick Python Code
In the `quick-prototyping.ipynb` notebook you will find more detailed explanations about the in and outs of the algorithm, but as a quick reference this is what the `c++` code will implement:

```python
mu, sigma = 20, 2  # the values we want to infer
np.random.seed(42)
observations = np.random.normal(mu, sigma, 50).astype(np.float32)

mu_range = np.linspace(18, 22, grid_size, dtype=np.float32)
sigma_range = np.linspace(1, 3, grid_size, dtype=np.float32)

# Generate the outer product of the ranges and the observations
mu_grid, sigma_grid, observations_grid = np.meshgrid(
    mu_range, sigma_range, observations
)

# build the prior
prior = np.ones((grid_size, grid_size))

# compute the likelihood
densities = norm(mu_grid, sigma_grid).pdf(observations_grid)
likes = densities.prod(axis=2)

# compute the posterior
posterior = likes * prior
posterior /= posterior.sum()

# Print the expectations
print(
    np.sum(mu_range * posterior.sum(axis=1)),
    np.sum(sigma_range * posterior.sum(axis=0))
)
```

#### Visualization
Basically you can think of the posterior as a kind of hill like this:

![normal distribution](assets/normal-distribution.jpg)

Where the $(x, y)$ plane are ranges of possible values of $\mu$ and $\sigma$ we want to explore. The crux of the problem, though, is how the height of the hill is calculated for each pair $(\mu, \sigma)$ out of the observations.

These observations are a sort of joint probability, this is, you need to know the probability of finding **all those values** under the assumption of some $(\mu_i, \sigma_i)$. If we were evaluating the observations under $(20, 2)$ this is what we would find:

![observations](assets/observations.png)

The resulting joint probability will then be:

$$x_1\;\cdot x_2\;\cdot x_3$$

where $x_1, x_2, x_3$ are the observations

> [!NOTE] Probability densities are not probabilities  
> We are loosely talking all the time about joint probability but, recall that probability densities are not probabilities, to find out the probability one needs to integrate over a range. However, bayesian statistics deals perfectly well with these densities as the only thing it cares about are proportions and, after all, we will be normalizing the outcomes at the end so all the volume under the hill will be $1$

Now, there's a point where the joint probability get's maximized for a pair of parameters, i.e., the peak of the hill which in our case will match the expected value as this is a very contrived example based on normal distributions. To find it out, we need to compute the marginals, that is, isolate each of the parameters and compute the expectation. These marginals look like the green and blue lines in the walls of the plot, i.e., the margins:

![marginals](assets/normal-distribution-with-marginals.png)


## Run It!
### Installation
- python and cuda requirements

### Execution
(it does not accept parameters, for the time being)

### Tests

## Making Sense

### File Structure
(may not be necessary)

### Logic Flow

## Resources

### Getting Started with CUDA and C++

### Additional Resources
I'd guess that not a lot of people will reach this part alive and if they do they will be unlikely
to check these references. If you are in that negligible tier, here you go, though.

