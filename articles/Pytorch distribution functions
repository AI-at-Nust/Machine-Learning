PyTorch is a popular open-source machine learning and deep learning framework. It uses tensors that are optimized for deep learning using GPUs and CPUs.

The PyTorch documentation defines a tensor as

> a multi-dimensional matrix containing elements of a single data type.

As Data Scientists, we need to work with several probability distributions to model behavior such as natural phenomena, financial assets, number of failures before success, and time elapsed between events.

The  `torch.Tensor`  class includes some methods that can be used to fill a tensor with random values from a continuous distribution of our choice. Five examples of such methods are

1.  `uniform_()`  - fills a tensor with values from a uniform distribution

2.  `normal_()`  - fills a tensor with values from a normal distribution

3.  `log_normal_()`  - fills a tensor with values from a log-normal distribution

4.  `exponential_()`  - fills a tensor with values from an exponential distribution

5.  `geometric_()`  - fills a tensor with values from a geometric distribution

Let’s start off by importing the torch library and creating an empty 5x3 tensor:

![Image for post](https://miro.medium.com/max/60/1*c9mKFIrZTWQLjXNaU9r_CQ.png?q=20)

![Image for post](https://miro.medium.com/max/563/1*c9mKFIrZTWQLjXNaU9r_CQ.png)

Declare an initialized 5x3 tensor called mat

For each function, I will present two working examples and one example that causes an exception to be raised.

----------

## Function 1 — uniform_(from=0, to=1)

_Keyword arguments:_

from --  minimum value of the distribution (default 0)to -- maximum value of the distribution (default 1)

_Returns:_

Tensor filled with random values from the uniform distribution inside the range specified by from and to

![Image for post](https://miro.medium.com/max/60/1*Hg5A51IFA6emdb_Ix9BZEw.png?q=20)

![Image for post](https://miro.medium.com/max/850/1*Hg5A51IFA6emdb_Ix9BZEw.png)

If X is a random variable uniformly distributed on [0, 1], then  `mat.uniform_()`  is uniformly distributed on [0, 1]. This method modifies  `mat`  in place by filling it with values from this continuous uniform distribution, and then returns it.

![Image for post](https://miro.medium.com/max/60/1*-tpakmOqY-6ZcIHy8-IdKQ.png?q=20)

![Image for post](https://miro.medium.com/max/854/1*-tpakmOqY-6ZcIHy8-IdKQ.png)

The same as the previous example except with different range i.e. [5, 10]. If the values in this tensor are plotted, they will form a rectangle.

![Image for post](https://miro.medium.com/max/60/1*HiV4mZ9uxGVgxgPzVgNc0g.png?q=20)

![Image for post](https://miro.medium.com/max/903/1*HiV4mZ9uxGVgxgPzVgNc0g.png)

This example breaks because when dealing with a uniform distribution, the range of values we specify must be strictly increasing. In this case  `from = 10`  and  `to = 5`, which specifies a decreasing range from 10 to 5. This causes a runtime error to be thrown.

You generally use uniform distributions when dealing with situations in which each outcome in a sample space is equally likely. For example, say you want to generate a Monte Carlo simulation of flipping a coin or rolling dice. You would then use a uniform distribution in this case. However, in such situations, the discrete uniform distribution is used.

In this case, since I explained the continuous uniform distribution, the applications are a bit different. Some examples include calculating:

-   Risk Analysis
-   Position of a particular air molecule in a room
-   The point on a car tire where the next puncture will occur
-   The length of time that one may have to wait for a train.

## Function 2 — normal_(mean = 0, std = 1)

Keyword arguments:

mean --  mean value of the distribution (default 0)std -- standard deviation of the distribution (default 1)

Returns:

Tensor filled with random values from the normal distribution parameterized by mean and std

![Image for post](https://miro.medium.com/max/60/1*_EXHHLr33aTQSqnq1Nog6w.png?q=20)

![Image for post](https://miro.medium.com/max/1131/1*_EXHHLr33aTQSqnq1Nog6w.png)

If X is a random variable normally distributed with mean = 0, std = 1, then  `mat.normal_()`  is normally distributed with mean = 0, std = 1. This method modifies  `mat`  in place by filling it with values from this continuous normal distribution and then returns it.

![Image for post](https://miro.medium.com/max/60/1*AQMVHh5KP20wK771zWtW-A.png?q=20)

![Image for post](https://miro.medium.com/max/1170/1*AQMVHh5KP20wK771zWtW-A.png)

The same as the previous example except with mean = 50 and standard deviation = 10. If the values in this tensor are plotted, they will form a bell curve that peaks at 50.

![Image for post](https://miro.medium.com/max/60/1*9qxUZXrtJhm4Wx_UaE8HTA.png?q=20)

![Image for post](https://miro.medium.com/max/911/1*9qxUZXrtJhm4Wx_UaE8HTA.png)

Although the value of mean can be any integer, it is compulsory for  `std > 0`. This makes sense because you can't create a normal distribution in which the values don't vary at all or in which they deviate from one another negatively.

Normal distributions are used almost everywhere when we want to model real-life phenomena. You can use this function to generate samples of a normal distribution if you know its mean and standard deviation. Some common examples are

1.  Height
2.  IQ
3.  Stocks
4.  Income Distribution In Economy
5.  Shoe Size
6.  Birth Weight
7.  Student Grades

## Function 3 — log_normal_(mean = 1, std = 2)

Keyword arguments:

mean --  mean value of the distribution (default 1)std -- standard deviation of the distribution (default 2)

Returns:

Tensor filled with random values from the log-normal distribution parameterized by mean and std

![Image for post](https://miro.medium.com/max/60/1*x9MIpAKk6IW0urU4qwk_iA.png?q=20)

![Image for post](https://miro.medium.com/max/1199/1*x9MIpAKk6IW0urU4qwk_iA.png)

If X is a random variable log-normally distributed with mean = 1, std = 2, then  `mat.log_normal_()`  is log-normally distributed with mean = 1, std = 2. This method modifies  `mat`  in place by filling it with values from this continuous log-normal distribution and then returns it.

![Image for post](https://miro.medium.com/max/60/1*j6eAbxNJyB2bUfADJOsoYw.png?q=20)

![Image for post](https://miro.medium.com/max/1226/1*j6eAbxNJyB2bUfADJOsoYw.png)

The same as the previous example except with mean = 50 and standard deviation = 10. This differs from a regular normal distribution by generating the natural logarithm of normally distributed values.

![Image for post](https://miro.medium.com/max/60/1*MGKNOg_Z-xmQe1SWm1vaTw.png?q=20)

![Image for post](https://miro.medium.com/max/916/1*MGKNOg_Z-xmQe1SWm1vaTw.png)

Just like a normal distribution, a log-normal distribution is allowed a  `mean <= 0`  but not a  `std <= 0`.

Log-normal distributions are commonly used in quantitative finance, especially when calculating share prices. Since asset prices cannot be negative, it is suitable for this purpose because it transforms negative values (due to a negative mean) into positive ones.

## Function 4 — exponential_(lambd = 1)

Keyword arguments:

lambd -- rate parameter of the distribution (default 1)

Returns:

Tensor filled with random values from the exponential distribution parameterized by lambd

![Image for post](https://miro.medium.com/max/60/1*fK4dNM2r6qxcrucuJNIvPw.png?q=20)

![Image for post](https://miro.medium.com/max/414/1*fK4dNM2r6qxcrucuJNIvPw.png)

If X is a random variable exponentially distributed with rate-parameter = 1, then  `mat.log_normal_()`  is exponentially distributed with rate-parameter = 1. This method modifies  `mat`  in place by filling it with values from this exponential distribution and then returns it.

![Image for post](https://miro.medium.com/max/60/1*AJhXi3ftNTY5WDAkCeu9jg.png?q=20)

![Image for post](https://miro.medium.com/max/455/1*AJhXi3ftNTY5WDAkCeu9jg.png)

Replaces the old values in the matrix by random values from an exponential distribution that has a rate-parameter of 0.5, i.e. average time/space between events (successes) that follow a Poisson Distribution is 0.5.

![Image for post](https://miro.medium.com/max/60/1*j4rQGTwYM8bUXqm862DJeQ.png?q=20)

![Image for post](https://miro.medium.com/max/913/1*j4rQGTwYM8bUXqm862DJeQ.png)

This example fails because  `lambd < 0`, which is an invalid value. This is because lambd describes a time difference, which cannot be negative.

The exponential distribution is useful to model phenomena involving time intervals. For example,

-   The time until a radioactive particle decays, or the time between clicks of a Geiger counter
-   The time it takes before your next telephone call
-   The time until default (on payment to company debt holders) in reduced form credit risk modeling

## Function 5 — geometric_(p)

Positional argument(s):

p -- probability of success

Returns:

Tensor filled with random values from the log-normal distribution parameterized by mean and st

![Image for post](https://miro.medium.com/max/60/1*PC49BVKiOJHjkWGRJhEYNg.png?q=20)

![Image for post](https://miro.medium.com/max/953/1*PC49BVKiOJHjkWGRJhEYNg.png)

For X number of Bernoulli trials, where the probability of success is  `p = 0.5`,  `mat.geometric_(0.5)`  fills  `mat`  with values from this geometric distribution and returns  `mat`. Note that unlike the previous distributions, this method doesn't have a default parameter.

![Image for post](https://miro.medium.com/max/60/1*WoTdzKVjHRAjWw-QIarCSQ.png?q=20)

![Image for post](https://miro.medium.com/max/976/1*WoTdzKVjHRAjWw-QIarCSQ.png)

Creates a geometric distribution where each trial has a probability of success  `p = 0.3142`, fills  `mat`  with random values from this distribution, and returns it.

![Image for post](https://miro.medium.com/max/60/1*1dyifC3AZYvFqunIMQn_Mg.png?q=20)

![Image for post](https://miro.medium.com/max/908/1*1dyifC3AZYvFqunIMQn_Mg.png)

This fails because the value of p is outside the accepted interval of [0,1]. This is because  `p`  is a probability, and probabilities are real numbers that lie in between [0,1].

The geometric distribution is useful for determining the likelihood of a success given a limited number of trials:

-   In sports, particularly in baseball, a geometric distribution is useful in analyzing the probability a batsman earns a hit before he receives three strikes; here, the goal is to reach success within 3 trials.
-   In cost-benefit analyses, such as a company deciding whether to fund research trials that, if successful, will earn the company some estimated profit, the goal is to reach success before the cost outweighs the potential gain.
-   In time management, the goal is to complete a task before some set amount of time.

# Conclusion

In this article, we reviewed how we can use several probability distributions in statistics to fill a tensor. It’s important to keep in mind that you need to declare a tensor before using these functions. We also saw some common uses for each distribution; so, if your project is related to the applications we discussed, consider using the random data made from these distributions in your training sets.

We mostly covered continuous distributions in this article, but PyTorch also supports discrete distributions. You can learn more about the classes of distributions in the  [PyTorch Documentation](https://jovian.ml/outlink?url=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Fdistributions.html).

You can also find the Jupyter notebook used to run these functions via this link:  [https://jovian.ml/qasimkhan5x/01-tensor-operations](https://jovian.ml/qasimkhan5x/01-tensor-operations)

# Reference Links

-   [Official documentation for](https://pytorch.org/docs/stable/tensors.html) `[torch.Tensor](https://pytorch.org/docs/stable/tensors.html)`
-   [Uniform Distribution](https://jovian.ml/outlink?url=https%3A%2F%2Fwww.itl.nist.gov%2Fdiv898%2Fhandbook%2Feda%2Fsection3%2Feda3662.htm)
-   [Normal Distribution](https://jovian.ml/outlink?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FNormal_distribution)
-   [Log-normal Distribution](https://jovian.ml/outlink?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLog-normal_distribution)
-   [Exponential Distribution](https://jovian.ml/outlink?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FExponential_distribution)
-   [Geometric Distribution](https://jovian.ml/outlink?url=https%3A%2F%2Fbrilliant.org%2Fwiki%2Fgeometric-distribution%2F%23practical-applications)
