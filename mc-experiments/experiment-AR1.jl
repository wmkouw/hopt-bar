
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using BenchmarkTools
using Distributions
using LinearAlgebra
using JLD2
using ProgressMeter
using Optim
using GaussianProcesses
includet("../optimization.jl");
includet("../evaluation.jl");

# Identities
experiment_ids = 1:50

# Time parameters
Δt = 0.1
len_time = 100
tspan = (0.0, Δt*(len_time-1))
tsteps = range(0, step=Δt, length=len_time)
len_horizon = 3

# Optimization settings
max_iters = 1000

# Initial condition
u_0 = 1.0

# Model parameters
M = 1
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.1
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

@showprogress for nn in experiment_ids

    "Draw system parameters"

    λ_true = rand(Beta(10., 4.))
    τ_true = rand(Gamma(10., 1.))

    "Generate signal"

    global signal = zeros(len_time)
    signal[1] = u_0
    for k in 2:len_time
        global signal[k] = (1 -λ_true*Δt)*signal[k-1] + sqrt(inv(τ_true))*rand(Normal(0,Δt))
    end
    signal /= std(signal)

    """Optimize Gaussian processes"""

    params_GP = optGP(tsteps,signal, max_iters=max_iters)
    time_GP = @belapsed optGP(tsteps,signal, max_iters=max_iters)

    # Test performance of found hyperparameters
    performance_GP = test_params(params_GP[:ll], params_GP[:lσ], tsteps, signal)    

    """Optimize temporal Gaussian processes"""

    # state0 = Normal(0.0, 1.0)

    # params_TGP = optTGP_Mat12(tsteps,signal,state0, max_iters=max_iters)
    # time_TGP = @belapsed optTGP_Mat12(tsteps,signal,state0, max_iters=max_iters)

    # # Test performance of found hyperparameters
    # performance_TGP = test_params(params_TGP[:ll], params_TGP[:lσ], tsteps, signal)    

    """Infer hyperparameters"""

    params_AR1 = optAR(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, M=Dx, len_horizon=len_horizon)    
    time_AR1 = @belapsed optAR(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, M=Dx, len_horizon=len_horizon)

    # Test performance of found hyperparameters
    performance_AR1 = test_params(params_AR1[:ll], params_AR1[:lσ], tsteps, signal)    
    
    trialnum = lpad(nn, 3, '0')
    jldsave("mc-experiments/results/experiment-AR1-trialnum$trialnum.jld2"; 
        tsteps, signal, λ_true, τ_true,
        performance_GP, params_GP, time_GP,
        # performance_TGP, params_TGP, time_TGP, 
        performance_AR1, params_AR1, time_AR1)

end