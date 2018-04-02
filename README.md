# DeconvolveDistribution

[![Build Status](https://travis-ci.org/maximerischard/DeconvolveDistribution.jl.svg?branch=master)](https://travis-ci.org/maximerischard/DeconvolveDistribution.jl)

[![Coverage Status](https://coveralls.io/repos/maximerischard/DeconvolveDistribution.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/maximerischard/DeconvolveDistribution.jl?branch=master)

[![codecov.io](http://codecov.io/github/maximerischard/DeconvolveDistribution.jl/coverage.svg?branch=master)](http://codecov.io/github/maximerischard/DeconvolveDistribution.jl?branch=master)

If $X$ comes from an unknown distribution
$$
    X_i \overset{iid}{\sim} F_X
$$
but we can only make noisy measurements
$$
    W_i = X_i + U_i
$$
where the distribution of $U_i$ is *known*,
then the `Fhat` function provided by this package gives
and estimate of $F_X$ given the sample $W$ and the
error distributions of $U$.

```julia
using DeconvolveDistribution
using Distributions

# simulate some data
n = 100
F_X = MixtureModel([Normal(-1, 1), Normal(2, 0.8)], [0.6, 0.4])
σ_distr = Gamma(1.0, 4.0)
srand(1)
X = rand(F_X, n)
σ = rand(σ_distr, n)
U_distr = Normal.(0.0, σ)
U = rand.(U_distr)
W = X .+ U

# estimate F_X from simulated data
n_xx = 120
F_xx = collect(linspace(extrema(W)..., n_xx))
num_t = 50
h = 0.5
Fhat_xx = Fhat(F_xx, W, num_t, h, U_distr)
```
