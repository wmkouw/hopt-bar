module ARModels

using Optim
using ForwardDiff
using Distributions
using SpecialFunctions
using LinearAlgebra

export ARModel, update!, predictions, posterior_predictive, backshift


mutable struct ARModel
    """
    Auto-Regressive model.

    Parameters are inferred through Bayesian filtering.
    """

    buffer          ::Vector{Float64}
    order           ::Integer
    thorizon        ::Integer

    μ               ::Vector{Float64}   # Coefficients mean
    Λ               ::Matrix{Float64}   # Coefficients precision
    α               ::Float64           # Likelihood precision shape
    β               ::Float64           # Likelihood precision rate

    free_energy     ::Float64

    function ARModel(coefficients_mean,
                     coefficients_precision,
                     noise_shape,
                     noise_rate; 
                     order::Integer=1,
                     time_horizon::Integer=1)

        if order != length(coefficients_mean) 
            error("Dimensionality of coefficients and model order do not match.")
        end
        buffer = zeros(order)

        free_energy = Inf

        return new(buffer,
                   order,
                   time_horizon,
                   coefficients_mean,
                   coefficients_precision,
                   noise_shape,
                   noise_rate,
                   free_energy)
    end
end

function update!(model::ARModel, y::Float64)

    μ0 = model.μ
    Λ0 = model.Λ
    α0 = model.α
    β0 = model.β

    x = model.buffer

    model.μ = inv(x*x' + Λ0)*(x*y + Λ0*μ0)
    model.Λ = x*x' + Λ0
    model.α = α0 + 1/2
    model.β = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - (x*y + Λ0*μ0)'*inv(x*x' + Λ0)*(x*y + Λ0*μ0))

    model.buffer = backshift(model.buffer, y)

    model.free_energy = -log_marginal_likelihood(model, (μ0, Λ0, α0, β0))
end

function params(model::ARModel)
    return model.μ, model.Λ, model.α, model.β
end

function marginal_likelihood(model::ARModel, prior_params)

    μn, Λn, αn, βn = params(model)
    μ0, Λ0, α0, β0 = prior_params

    return (det(Λn)^(-1/2)*gamma(αn)*βn^αn)/(det(Λ0)^(-1/2)*gamma(α0)*β0^α0) * (2π)^(-1/2)
end

function log_marginal_likelihood(model::ARModel, prior_params)

    μn, Λn, αn, βn = params(model)
    μ0, Λ0, α0, β0 = prior_params

    return -1/2*logdet(Λn) + log(gamma(αn)) + αn*log(βn) -(-1/2*logdet(Λ0) +log(gamma(α0)) + α0*log(β0)) -1/2*log(2π)
end

function posterior_predictive(model::ARModel, x_t)
    "Posterior predictive distribution is location-scale t-distributed"

    ν_t = 2*model.α
    m_t = dot(model.μ, x_t)
    s2_t = model.β/model.α*(1 + x_t'*inv(model.Λ)*x_t)

    return ν_t, m_t, s2_t
end

function predictions(model::ARModel, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    buffer = model.buffer
    
    for t in 1:time_horizon

        ν_t, m_t, s2_t = posterior_predictive(model, buffer)
        
        # Prediction
        m_y[t] = m_t
        v_y[t] = s2_t * ν_t/(ν_t - 2)
        
        # Update previous 
        buffer = backshift(buffer, m_y[t])
        
    end
    return m_y, v_y
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

end
