from numpy import mean, random, exp
import numpy
from scipy.stats import poisson
import math
import pacal

class NumberGenerator:

    def gen(self):
        return 1

    def mean(self):
        return 1


class GenerateFromSample(NumberGenerator):

    def __init__(self, samples):
        self.samples = samples
        self.average = mean(samples)

    def gen(self):
        return random.choice(self.samples)

    def gen_n(self, n):
        return random.choice(self.samples, n)

    def mean(self):
        return self.average

    def sample(self, n):
        return self.gen_n(n)


class GenerateBinomial(NumberGenerator):
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def gen(self):
        return random.binomial(self.n, self.p)

    def gen_n(self, n):
        return random.binomial(self.n, self.p, size=n)

    def mean(self):
        return self.n * self.p


class GeneratePoisson(NumberGenerator):
    def __init__(self, mu):
        self.mu = mu

    def gen(self):
        return random.poisson(self.mu)

    def gen_n(self, n):
        return random.poisson(self.mu, size=n)

    def mean(self):
        return self.mu


class GenerateTruncatedPoisson(NumberGenerator):
    def __init__(self, mu, trunk):
        self.mu = mu
        self.limit = 0
        while poisson.cdf(self.limit, mu) < 1 - trunk:
            self.limit += 1

    def gen(self):
        ret = random.poisson(self.mu)
        while ret > self.limit:
            ret = random.poisson(self.mu)
        return ret

    def gen_n(self, n):
        return random.poisson(self.mu, size=n)

    def mean(self):
        return self.mu

class GenerateTruncatedGeneralizedPoisson(NumberGenerator):
    def __init__(self, mu, alpha, trunk):
        self.mu = mu
        self.alpha = alpha
        self.trunk = trunk
        self.usages = []
        self.probs = []

        if mu>0:
            p = self.GenPoissonPdf(self.mu,self.alpha,0)
            self.usages.append(0)
            self.probs.append(p)

            for i in range(1,1000):
               p = self.GenPoissonPdf(self.mu,self.alpha,i)
               self.usages.append(i)
               if self.probs[-1]>p  and p<self.trunk:
                   self.probs.append(1-sum(self.probs))
                   break
               else:
                   self.probs.append(p)
                   if sum(self.probs)>1:
                        self.probs[-1] = 1-sum(self.probs[:-1])
                        break
           #print(usages,probs,self.scale,a)
        else:
            self.usages = [0]
            self.probs = [1]

    def GenPoissonPdf(self,o=1,a=0,x=0):
        o = o*(1-a)
        return (o*(o+a*x)**(x-1))*(exp(-o-a*x))/math.factorial(x)

    def gen(self):
        #print(self.usages,self.probs,self.alpha,self.mu)
        #print(random.choice(numpy.array(self.usages,p=self.probs))
        return random.choice(self.usages,p=self.probs)

    def gen_n(self, n):
        return random.choice(self.usages,p=self.probs, size=n)

    def mean(self):
        return self.mu























class GenerateDeterministic(NumberGenerator):

    def __init__(self, value):
        self.value = value

    def gen(self):
        return self.value

    def gen_n(self, n):
        return [self.value] * n

    def mean(self):
        return self.value


class GenerateFromNormal(NumberGenerator):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def gen(self):
        return random.normal(self.mean, self.std)

    def mean(self):
        return self.mean

    def sample(self, n):
        return [self.gen() for i in range(n)]


class GenerateFromPositiveNormal(NumberGenerator):
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.mean = mean(self.sample(10000))

    def gen(self):
        x = random.normal(self.mu, self.std)
        while x < 0:
            x = random.normal(self.mu, self.std)
        return x

    def mean(self):
        return self.mean

    def sample(self, n):
        return [self.gen() for i in range(n)]


class GenerateFromLogNormal(NumberGenerator):
    def __init__(self, mu, sigma):
        self.mu = mean
        self.sigma = sigma

    def gen(self):
        return random.lognormal(self.mu, self.sigma)

    def gen_n(self, n, discrete=True):
        if discrete:
            return list(int(x) for x in random.lognormal(self.mu, self.sigma, n))
        else:
            return random.lognormal(self.mu, self.sigma, n)

    def mean(self):
        return math.exp(self.mu + self.sigma ** 2 / 2)

    def sample(self, n):
        return [self.gen() for i in range(n)]


class GenerateFromScaledLogNormal(NumberGenerator):
    """ x ~ C * x1 where C is a constant and x1 is a LogNormal RV"""

    def __init__(self, mu, sigma, c):
        self.mu = mu
        self.sigma = sigma
        self.c = c

    def gen(self):
        return self.c * random.lognormal(self.mu, self.sigma)

    def gen_n(self, n):
        return self.c * random.lognormal(self.mu, self.sigma, n)

    def mean(self):
        return self.c * math.exp(self.mu + self.sigma ** 2 / 2)

    def sample(self, n):
        return [self.gen() for i in range(n)]
