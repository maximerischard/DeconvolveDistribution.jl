module DeconvolveDistribution

using FastGaussQuadrature
import Distributions: params, partype,
                      logpdf, pdf, cf,
                      UnivariateDistribution,
                      DiscreteUnivariateDistribution, 
                      insupport, 
                      Normal
using Statistics: var
using LinearAlgebra: diagm, norm
using RCall
import Optim

include("fourier.jl")
include("expospline.jl")

"""
    Estimation using Fourier transforms leads
    to estimates of the CDF that are not strictly valid:
    they don't go from 0 to 1 and aren't monotonic.
    This is a function to "fix them up", intended
    as a post-processing step.
"""
function fix_CDF!(Fhat_xx)
    imedian = argmin(abs.(Fhat_xx .- 0.5))
    # monotonically increasing from median up
    Fhat_xx[imedian:end] = accumulate(max, Fhat_xx[imedian:end])
    # monotonically decreasing from median down
    Fhat_xx[imedian:-1:1] = accumulate(min, Fhat_xx[imedian:-1:1])
    n_xx = length(Fhat_xx)
    for i in 1:n_xx
        Fhat_xx[i] = min(Fhat_xx[i], 1.0)
        Fhat_xx[i] = max(Fhat_xx[i], 0.0)
    end
    Fhat_xx[1] = 0.0
    Fhat_xx[end] = 1.0
    return Fhat_xx
end

export Fhat, fix_CDF!

end # module
