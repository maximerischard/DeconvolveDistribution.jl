module DeconvolveDistribution

using FastGaussQuadrature
import Random
import Distributions
import Distributions: params, partype,
                      logpdf, pdf, cf,
                      logcdf, cdf,
                      rand,
                      UnivariateDistribution,
                      insupport, 
                      Normal, Uniform, MixtureModel
using Statistics: var
using StatsBase: Weights
using LinearAlgebra: diagm, norm, dot
using RCall
import Optim

include("fourier.jl")
include("expospline.jl")
include("interface.jl")

export Fhat, fix_CDF!, decon,
       FourierDeconv, EfronDeconv

end # module
