abstract type DeconvolutionMethod end

#######################################
# Interface for Fourier Deconvolution #
#######################################
struct FourierDeconv <: DeconvolutionMethod
    bandwidth::Float64
    Xgrid::Vector{Float64}
    num_t::Int
end

function piecewise_uniform(bin_edges::AbstractVector, weights::Weights)
    @assert length(bin_edges) == length(weights)+1
    @assert all(diff(bin_edges) .> 0)
    bin_uniforms = Uniform.(bin_edges[1:end-1], prevfloat.(bin_edges[2:end]))
    mixture = MixtureModel(bin_uniforms, values(weights) / sum(weights)) # mixture of uniforms with equal weights
end

# function decon(X::Vector, σ_X::Vector, bw::Float64, grid::Vector; num_t=50, fixup=true)
function decon(dm::FourierDeconv, X, ϵdistr)
    Fhat_xx = Fhat(dm.Xgrid, X, dm.num_t, dm.bandwidth, ϵdistr)
    fix_CDF!(Fhat_xx)
    return piecewise_uniform(dm.Xgrid, Weights(diff(Fhat_xx)))
    # Phat_xx = diff(Fhat_xx)
    # return Distributions.Generic(dm.Xgrid[2:end], Phat_xx)
end

##################################################
# Interface for Exponential Spline Deconvolution #
##################################################
struct EfronDeconv <: DeconvolutionMethod
    dimension::Int
    Xgrid::Vector{Float64}
    c0::Float64
end

function decon(dm::EfronDeconv, X, ϵdistr)
    d = ExpoSpline(dm.Xgrid, zeros(dm.dimension))
    d_optim = DeconvolveDistribution.decon(d, ϵdistr, X, dm.c0)
    return d_optim
end
