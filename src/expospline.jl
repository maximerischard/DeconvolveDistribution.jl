function spline_basis(domain, p)
    R_spline = R"splines::ns($domain, $p)"
    Q = convert(Matrix{Float64}, R_spline)
    return Q
end

struct ExpoSpline{T<:Real} <: DiscreteUnivariateDistribution
    domain::Vector{Float64}
    Qbasis::Matrix{Float64} # basis matrix
    αcoef::Vector{T} # α
    Qα::Vector{T}
    φ_α::T # normalization constant
end
function ExpoSpline(domain, Qbasis::Matrix{Float64}, αcoef::Vector{T}) where {T <: Real}
    Qα = Qbasis * αcoef
    φ_α = log_sum_exp(Qα) # equation (10)
    return ExpoSpline(domain, Qbasis, αcoef, Qα, φ_α)
end
function ExpoSpline(domain, αcoef::Vector{T}) where {T <: Real}
    p = length(αcoef)
    Q = spline_basis(domain, p)
    return ExpoSpline(domain, Q, αcoef)
end 
## doesn't work for ExpoSpline immutable ##
# function update_coefs!(d::ExpoSpline{T}, α::Vector{T}) where {T <: Real}
    # d.αcoef = α
    # LinAlg.A_mul_B!(d.Qα, d.Qbasis, α)
    # d.φ_α = log_sum_exp(d.Qα)
    # return d
# end

params(d::ExpoSpline) = (d.αcoef, )
coefs(d::ExpoSpline) = d.αcoef
partype(d::ExpoSpline{T}) where {T<:Real} = T

function log_sum_exp(x)
    # https://en.wikipedia.org/wiki/LogSumExp
    max_x = maximum(x)
    log_sum_exp_x = max_x + log(sum(xi -> exp(xi-max_x), x))
    return log_sum_exp_x
end
logpdf(d::ExpoSpline{T}) where T = d.Qα .- d.φ_α             # equation (9)
pdf(d::ExpoSpline{T}) where T = exp.(logpdf(d))
logpdf(d::ExpoSpline{T}, j::Int) where {T} = d.Qα[j] - d.φ_α # equation (9)
function logpdf(d::ExpoSpline{T}, x::Real) where {T}
    j = argmin(abs.(d.domain .- x)) # not the fastest way to do this
    return logpdf(d, j)
end
function pdf(d::ExpoSpline{T}, x::Real) where {T}
    return exp(logpdf(d, x))
end
function get_logPi(domain::Vector{Float64}, noise::UnivariateDistribution, X_i::Float64)
    # unnormalized
    return logpdf.(noise, X_i.-domain)
end
function get_Pi(domain::Vector{Float64}, noise::UnivariateDistribution, X_i::Float64)
    # unnormalized
    return pdf.(noise, X_i.-domain)
end
get_logPi(d::ExpoSpline, noise, X_i) = get_logPi(d.domain, noise, X_i)
get_Pi(d::ExpoSpline, noise, X_i) = get_Pi(d.domain, noise, X_i)
function loglikelihood(logPi::AbstractVector{Float64}, loggα::Vector{T}) where T<:Real
    max_x = maximum(logPi) + maximum(loggα)
    sum_exp_x = zero(max_x)
    @inbounds for (logPi_j, loggα_j) in zip(logPi, loggα)
        sum_exp_x += exp(logPi_j + loggα_j - max_x)
    end
    return max_x + log(sum_exp_x)
end
function loglikelihood(d::ExpoSpline, logPi::AbstractVector{Float64})
    loggα = logpdf(d)
    return loglikelihood(logPi, loggα)
end
function getWi(d::ExpoSpline, logPi::AbstractVector{Float64})
    # equation (14)-(15)
    loggα = logpdf(d)
    logfi = loglikelihood(logPi, loggα)
    Wi = exp.(loggα) .* (exp.(logPi .- logfi) .- 1)
    return Wi
end
function dloglik(d::ExpoSpline, logPi::AbstractVector{Float64})
    # equation (16)
    return d.Qbasis'getWi(d, logPi)
end
function d2loglik(d::ExpoSpline, logPi::AbstractVector{Float64})
    gα = pdf(d)
    Wi = getWi(d, logPi)
    return -d.Qbasis'*(Wi*Wi' + Wi*gα' + gα*Wi' - diagm(Wi))*d.Qbasis
end 
function loglikelihood(d::ExpoSpline, noise::UnivariateDistribution, X_i::Float64)
    logPi = get_logPi(d.domain, noise, X_i)
    return loglikelihood(d, logPi)
end

function make_loglik_w_noise_pre(d::ExpoSpline, noise, X::Vector{Float64})
    logP = get_logPi.(d, noise, X)
    mloglik_w_noise_pre = function (α)
        dα = ExpoSpline(d.domain, d.Qbasis, α)
        loglik = sum(Pi -> loglikelihood(dα, Pi), logP)
        return -loglik
    end
    return mloglik_w_noise_pre
end
function make_loglik_noisefree(d::ExpoSpline, θ::Vector{Float64})
    mloglik = (α) -> -sum(
        logpdf.( # compute the log-PDF under the…
            ExpoSpline(d.domain, d.Qbasis, α), # …candidate distribution…
            θ # …of the noise-free data
            )
        )
    return mloglik
end
##############
# with Prior #
##############
function make_penalized_w_noise_pre(d::ExpoSpline, noise, X::Vector{Float64}, c0::Real)
    logP = get_logPi.(d, noise, X)
    target = function (α)
        dα = ExpoSpline(d.domain, d.Qbasis, α)
        loglik = sum(Pi -> loglikelihood(dα, Pi), logP)
        penalty = c0 * norm(α, 2)
        return -loglik + penalty
    end
    return target
end

################
# Optimization #
################
function decon(d, noise, X::Vector{Float64}, c0::Real)
    f = make_penalized_w_noise_pre(d, noise, X, c0)
    td = Optim.TwiceDifferentiable(f, copy(coefs(d)); autodiff=:forward)
    opt_reg = Optim.optimize(td, zero(coefs(d)), Optim.Newton())
    @assert Optim.converged(opt_reg)
    d_optim = ExpoSpline(d.domain, d.Qbasis, Optim.minimizer(opt_reg))
    return d_optim
end
