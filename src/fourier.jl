ϕK(t) = -1 < t < 1 ? (1.0-t^2) : 0.0

# sin(tx)/t
function sinc(t, x)
    if abs(t) <= 1e-4
        return x - t^2*x^3/6.0
    else
        return sin(t*x) / t
    end
end
""" 
    very literal implementation of equation 8.5
    in Wang&Wang 2011
"""
function Fhatx_brute(x::Real, W::AbstractVector, num_t::Int, h::Real,
        U::AbstractVector{D} where D <: UnivariateDistribution)
    quad_nodes, quad_weight = FastGaussQuadrature.gausslegendre(num_t)
    htt = quad_nodes
    tt = htt ./ h
    t_weight = quad_weight ./ h

    ϕKht = ϕK.(htt)
    n = length(W)
    s = 0.0
    sum_abs_ϕUkt = abs2.(cf.(U[1], tt))
    for k in 2:n
        sum_abs_ϕUkt .+= abs2.(cf.(U[k], tt))
    end
    mean_abs_ϕUkt = sum_abs_ϕUkt/n
    @inbounds for j in 1:n
        sinctxW = sinc.(tt, x-W[j])
        ϕUj = cf.(U[j], -tt)
        inv_ψUj = ϕUj ./ mean_abs_ϕUkt 
        s += dot(t_weight, sinctxW .* ϕKht  .* inv_ψUj)
        isfinite(s) || throw(AssertionError("s=$s is not finite, j=$j"))
    end
    Fx = 0.5 + 1/(2π*n) * s # 2h is the Jacobian
    @assert isfinite(Fx)
    (imag(Fx) ≈ 0.0) || AssertionError("Fx has non-negligible imaginary part $Fx") 
    return real(Fx)
end

@inline function real_cf_Uj_t(Uj::UnivariateDistribution, t::Real)
    return @fastmath real(cf(Uj, t))
end
@inline function real_cf_Uj_t(Uj::Normal, t::Real)
    # assumes Uj has mean zero!
    return exp(-var(Uj)*t^2/2)
end

function Fhat(xx::Vector{Float64}, W::Vector{Float64}, num_t::Int, h::Real, 
        U::Vector{D} where D <: UnivariateDistribution)
    quad_nodes, t_weight = FastGaussQuadrature.gausslegendre(num_t)
    # FastGaussQuadrature outputs nodes from -1 to 1
    # we need nodes from 0 to 1
    htt = (quad_nodes .+ 1.0) ./ 2.0
    tt = htt ./ h
    ϕKht = ϕK.(htt)
    n = length(W)
    sum_abs_ϕUkt = abs2.(cf.(U[1], tt))
    for k in 2:n
        sum_abs_ϕUkt .+= abs2.(cf.(U[k], tt))
    end
    inv_mean_abs_ϕUkt = n ./ sum_abs_ϕUkt
    s_cache = Vector{Float64}[]
    for j in 1:n
        Uj = U[j]
        sUj = real_cf_Uj_t.(Uj, -tt) .* inv_mean_abs_ϕUkt .* t_weight .* ϕKht
        push!(s_cache, sUj)
    end
    F_xx = zeros(Float64, length(xx))
    @inbounds for (ix, x) in enumerate(xx)
        s = zero(Float64)
        for j in 1:n
            Wj, Uj = W[j], U[j]
            scache_j = s_cache[j]
            for (it, t) in enumerate(tt)
                # sinctxW = @fastmath sin(t*(x-Wj))/t
                sinctxW = sinc(t, x-Wj)
                s += scache_j[it] * sinctxW
            end
        end
        F_xx[ix] = 0.5 + 1/(π*n) * s / (2*h) # 2h is the Jacobian
    end 
    return F_xx
end

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
