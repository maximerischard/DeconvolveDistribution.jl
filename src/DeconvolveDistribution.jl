module DeconvolveDistribution

using FastGaussQuadrature
import Distributions
import Distributions: params, partype,
                      logpdf, pdf, cf,
                      UnivariateDistribution,
                      DiscreteUnivariateDistribution, 
                      insupport, 
                      Normal, Uniform, MixtureModel
using Statistics: var
using StatsBase: Weights
using LinearAlgebra: diagm, norm
using RCall
import Optim

include("fourier.jl")
include("expospline.jl")
include("interface.jl")

export Fhat, fix_CDF!

end # module
