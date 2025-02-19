module MAR

using Distributions
using SpecialFunctions
using LinearAlgebra

export MARModel, update!, predictions, posterior_predictive


mutable struct MARModel
    """
    Multivariate Auto-Regressive model.
    """

    Dy              ::Integer
    Dx              ::Integer
    ybuffer         ::Matrix{Float64}
    delay           ::Integer
    thorizon        ::Integer

    M               ::Matrix{Float64}   # Coefficients mean matrix
    Λ               ::Matrix{Float64}   # Coefficients row-covariance
    Ω               ::Matrix{Float64}   # Precision scale matrix
    ν               ::Float64           # Precision degrees-of-freedom

    free_energy     ::Float64

    function MARModel(coefficients_mean_matrix,
                      coefficients_row_covariance,
                      precision_scale,
                      precision_degrees;
                      Dy::Integer=2,
                      delay::Integer=1, 
                      time_horizon::Integer=1)

        ybuffer = zeros(Dy,delay)
        Dx = Dy*delay

        free_energy = Inf

        return new(Dy,
                   Dx,
                   ybuffer,
                   delay,
                   time_horizon,
                   coefficients_mean_matrix,
                   coefficients_row_covariance,
                   precision_scale,
                   precision_degrees,
                   free_energy)
    end
end

function update!(agent::MARModel, y_k::Vector)

    # Short-hand
    M = agent.M
    Λ = agent.Λ
    Ω = agent.Ω
    ν = agent.ν

    # Reshape buffer to vector
    x_k = agent.ybuffer[:]

    # Update performance metric
    agent.free_energy = -logevidence(agent, y_k, x_k)

    # Auxiliary variables
    X = x_k*x_k'
    Ξ = (x_k*y_k' + Λ*M)

    # Update rules
    agent.ν = ν + 1
    agent.Λ = Λ + X
    agent.Ω = Ω + y_k*y_k' + M'*Λ*M - Ξ'*inv(Λ+X)*Ξ
    agent.M = inv(Λ+X)*Ξ

    # Update output buffer
    agent.ybuffer = backshift(agent.ybuffer, y_k)
end

function params(agent::MARModel)
    return agent.M, agent.U, agent.V, agent.ν
end

function logevidence(agent::MARModel, y,x)
    η, μ, Ψ = posterior_predictive(agent, x)
    return -1/2*(agent.Dy*log(η*π) -logdet(Ψ) - 2*logmultigamma(agent.Dy, (η+agent.Dy)/2) + 2*logmultigamma(agent.Dy, (η+agent.Dy-1)/2) + (η+agent.Dy)*log(1 + 1/η*(y-μ)'*Ψ*(y-μ)) )
end

function posterior_predictive(agent::MARModel, x_t)
    "Posterior predictive distribution is multivariate T-distributed."

    η_t = agent.ν - agent.Dy + 1
    μ_t = agent.M'*x_t
    Ψ_t = (agent.ν-agent.Dy+1)*inv(agent.Ω)*inv(1 + x_t'*inv(agent.Λ)*x_t)

    return η_t, μ_t, Ψ_t
end

function predictions(agent::MARModel; time_horizon=1)
    
    m_y = zeros(agent.Dy,time_horizon)
    S_y = zeros(agent.Dy,agent.Dy,time_horizon)

    ybuffer = agent.ybuffer
    
    for t in 1:time_horizon
        
        # Reshape buffer to vector
        x_t = ybuffer[:]

        # Prediction
        η_t, μ_t, Ψ_t = posterior_predictive(agent, x_t)
        m_y[:,t] = μ_t
        S_y[:,:,t] = inv(Ψ_t) * η_t/(η_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[:,t])
        
    end
    return m_y, S_y
end

function backshift(x::AbstractMatrix, a::Vector)
    "Shift elements rightwards and add element"
    return [a x[:,1:end-1]]
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function multigamma(p,a)
    result = π^(p*(p-1)/4)
    for j = 1:p 
        result *= gamma(a + (1-j)/2)
    end
    return result
end

function logmultigamma(p,a)
    result = p*(p-1)/4*log(π)
    for j = 1:p 
        result += loggamma(a + (1-j)/2)
    end
    return result
end

function sqrtm(M::AbstractMatrix)
    "Square root of matrix"

    if size(M) == (2,2)
        "https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix"

        A,C,B,D = M

        # Determinant
        δ = A*D - B*C
        s = sqrt(δ)

        # Trace
        τ = A+D
        t = sqrt(τ + 2s)

        return 1/t*(M+s*Matrix{eltype(M)}(I,2,2))
    else
        "Babylonian method"

        Xk = Matrix{eltype(M)}(I,size(M))
        Xm = zeros(eltype(M), size(M))

        while sum(abs.(Xk[:] .- Xm[:])) > 1e-3
            Xm = Xk
            Xk = (Xm + M/Xm)/2.0
        end
        return Xk
    end
end

end
