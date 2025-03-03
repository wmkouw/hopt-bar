using Statistics
using LinearAlgebra
using GaussianProcesses

function test_params(λ::Float64,σ::Float64, tsteps, signal)

    kernel = Mat12Iso(λ, σ)
    kmean  = MeanZero()
    gp = GP(tsteps, signal, kmean, kernel)
    pp_m, pp_s = predict_y(gp, tsteps)
    RMS = sqrt(mean( (pp_m .- signal).^2 ))
    NLE = mean( -logpdf.(Normal.(pp_m,pp_s), signal) )
    MML = mean( pdf.(Normal.(pp_m,pp_s), signal) )

    return RMS,NLE,MML
end