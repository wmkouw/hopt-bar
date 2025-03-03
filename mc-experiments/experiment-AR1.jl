
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using BenchmarkTools
using DifferentialEquations
using Distributions
using LinearAlgebra
using JLD2
using ProgressMeter
using RxInfer
using Optim
using GaussianProcesses
using Plots; 
default(label="", linewidth=3, margin=15Plots.pt);
includet("../optimization.jl");
includet("../evaluation.jl");

# Identities
experiment_ids = 1:10

# Time parameters
Δt = 0.1
len_time = 100
tspan = (0.0, Δt*(len_time-1))
tsteps = range(0, step=Δt, length=len_time)
len_horizon = 3

# Initial condition
u_0 = 1.0

# Model parameters
M = 1
Dy = 1
Dx = Dy*M

# Prior parameters
α0 = 2.0
β0 = 0.01
Λ0 = 1e-3*diagm(ones(Dx))
μ0 = zeros(Dx)

@showprogress for nn in experiment_ids

    "Draw system parameters"

    λ_true = rand(Beta(1., 3.))
    τ_true = rand(Gamma(2., 1.))
    σ_true = sqrt(inv(τ_true))

    "Generate signal"

    global signal = zeros(len_time)
    signal[1] = u_0
    for k in 2:len_time
        global signal[k] = (1 -λ_true*Δt)*signal[k-1] + σ_true*rand(Normal(0,Δt))
    end
    signal /= std(signal)

    """Optimize Gaussian processes"""

    time_GP = @belapsed optGP(tsteps,signal)
    params_GP = optGP(tsteps,signal)

    if params_GP[:λ] <= 0.0; error("bam"); end
    if params_GP[:σ] <= 0.0; error("bim"); end

    # Test performance of found hyperparameters
    performance_GP = test_params(params_GP[:λ], params_GP[:σ], tsteps, signal)    

    # """Optimize temporal Gaussian processes"""

    # time_GP = @belapsed optTGP(tsteps,signal)
    # params_GP = optTGP(tsteps,signal)

    # # Test performance of found hyperparameters
    # RMS_GP, NLE_GP = test_params(params_GP[:λ], params_GP[:σ], tsteps, signal)    


    """Infer hyperparameters"""

    time_AR = @belapsed optAR(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, M=Dx, len_horizon=len_horizon)
    params_AR = optAR(tsteps,signal, μ0=μ0,Λ0=Λ0,α0=α0,β0=β0, M=Dx, len_horizon=len_horizon)    

    # Test performance of found hyperparameters
    performance_AR = test_params(params_AR[:λ], params_AR[:σ], tsteps, signal)    
    
    trialnum = lpad(nn, 3, '0')
    jldsave("mc-experiments/results/experiment-AR1-trialnum$trialnum.jld2"; tsteps,signal, performance_GP, params_GP, time_GP, performance_AR, params_AR, time_AR, λ_true, σ_true)

end