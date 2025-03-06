using Optim
using ForwardDiff
using LinearAlgebra
using GaussianProcesses

includet("ARModels.jl"); using .ARModels
includet("RTSSmoothers.jl"); using .RTSSmoothers


function optGP(tsteps, signal; max_iters=1)

    ll = rand()
    lσ = rand()
    
    kernel = Mat12Iso(ll, lσ)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)

    optimize!(gp, noise=false, domean=false, kern=true, lik=false, iterations=max_iters)

    return Dict(:ll => log(gp.kernel.ℓ), :lσ => log(sqrt(gp.kernel.σ2)))
end

function optTGP_Mat12(tsteps, signal, state0; max_iters=1)

    function J(hparams)
        expλ = exp(hparams[1])
        expσ = exp(hparams[2])
        A = [-expλ]
        Q = [2*expλ*expσ^2]
        C = [1.0]
        R = [1e-8]
        model = RTSSmoother(A,C,Q,R,state0)
        return log_marginal_likelihood(model,signal)
    end

    opt = Optim.Options(g_tol = 1e-12,
                        iterations = max_iters,
                        store_trace = false,
                        show_trace = false,
                        show_warnings = true)
    res = optimize(J, zeros(2), LBFGS(), opt)
    mins = Optim.minimizer(res)

    return Dict(:ll => mins[1], :lσ => mins[2])
end

function optAR(tsteps, signal; μ0=1.0, Λ0=[1.0], α0=2.0, β0=1/2, M=1, len_horizon=1)

    model = ARModel(μ0,Λ0,α0,β0, order=M, time_horizon=len_horizon)
    for y_k in signal
        ARModels.update!(model, y_k)
    end

    # Bound estimate
    if model.μ[1] >= 1; model.μ[1] = .99999; end

    # Reverting variable substitution
    ll_hat = log(Δt./(1 - model.μ[1]))
    lσ_hat = log( sqrt( model.β./( 2*(model.α - 1)*(1 - model.μ[1])*Δt^2 ) ) )

    return Dict(:ll => ll_hat, :lσ => lσ_hat)
end
