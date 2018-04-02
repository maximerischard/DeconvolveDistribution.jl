using DeconvolveDistribution
using Distributions
using Base.Test

@testset "Fhat" begin
    n = 100
    F_X = MixtureModel([Normal(-1, 1), Normal(2, 0.8)], [0.6, 0.4])
    σ_distr = Gamma(1.0, 4.0)
    srand(1)
    X = rand(F_X, n)
    σ = rand(σ_distr, n)
    U = Normal.(0.0, σ)
    ϵ = rand.(U)
    W = X .+ ϵ

    n_xx = 120
    F_xx = collect(linspace(extrema(W)..., n_xx))
    num_t = 50
    h = 0.5
    Fhat_xx = Fhat(F_xx, W, num_t, h, U)

    @test length(Fhat_xx) == n_xx
    # CDF starts at 0
    @test -0.01 < Fhat_xx[1] < 0.01
    # CDF ends at 1
    @test -0.99 < Fhat_xx[end] < 1.01
    # CDF is monotonic
    @test all(diff(Fhat_xx) .>= -0.005)
end # testset

@testset "fix_CDF" begin
    n = 100
    F_X = MixtureModel([Normal(-1, 1), Normal(2, 0.8)], [0.6, 0.4])
    σ_distr = Gamma(1.0, 4.0)
    srand(1)
    X = rand(F_X, n)
    σ = rand(σ_distr, n)
    U = Normal.(0.0, σ)
    ϵ = rand.(U)
    W = X .+ ϵ

    n_xx = 120
    F_xx = collect(linspace(extrema(W)..., n_xx))
    num_t = 20
    h = 0.5
    Fhat_xx = Fhat(F_xx, W, num_t, h, U)

    fix_CDF!(Fhat_xx)

    @test length(Fhat_xx) == n_xx
    # CDF starts at 0
    @test Fhat_xx[1] == 0.0
    # CDF ends at 1
    @test Fhat_xx[end] == 1.0
    # CDF doesn't go outside of [0, 1]
    for Fx in Fhat_xx
        @test 0.0 <= Fx <= 1.0
    end
    # CDF is monotonic
    for dFx in diff(Fhat_xx)
        @test dFx >= 0.0
    end
end
