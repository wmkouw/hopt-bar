using Optim
using ForwardDiff
using LinearAlgebra
using GaussianProcesses

includet("ARModels.jl"); using .ARModels
# includet("RTSSmoothers.jl"); using .RTSSmoothers


function optGP(tsteps, signal; max_iters=1)
    
    kernel = Mat12Iso(rand(), rand())
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)

    optimize!(gp, noise=false, domean=false, kern=true, iterations=max_iters)

    return Dict(:λ => gp.kernel.ℓ, :σ => gp.kernel.σ2)
end

function optTGP(tsteps, signal; num_iters=1)


    model = RTSSmoother( )

    for (k,t) in enumerate(tsteps)

        x_k = model.buffer[:]
        _,ppar_m[k],ppar_s[k] = ARModels.posterior_predictive(model, x_k)

        # Update parameters
        ARModels.update!(model, signal[k])

    end
    return Dict(:μ => μs[:,end], :Λ => Λs[:,:,end], :α => αs[end], :β => βs[end])
end

function optAR(tsteps, signal; μ0=1.0, Λ0=[1.0], α0=2.0, β0=1/2, M=1, len_horizon=1)

    model = ARModel(μ0,Λ0,α0,β0, order=M, time_horizon=len_horizon)
    for (k,t) in enumerate(tsteps)
        ARModels.update!(model, signal[k])
    end

    λ_hat = (1 - model.μ[1])./Δt
    σ_hat = sqrt( model.β./( model.α - 1 ) )

    return Dict(:λ => λ_hat, :σ => σ_hat)
end
