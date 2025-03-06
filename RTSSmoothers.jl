module RTSSmoothers

using Optim
using ForwardDiff
using Distributions
using SpecialFunctions
using LinearAlgebra

export RTSSmoother, filter, smoothen, marginal_likelihood, log_marginal_likelihood


mutable struct RTSSmoother
    """
    Rauch-Tung-Striebel smoother
    """

    Dx      :: Integer # State dimensionality
    Dy      :: Integer # Measurement dimensionality
    A       :: AbstractArray   # State transition matrix
    C       :: AbstractArray   # Measurement matrix
    Q       :: AbstractArray   # Process noise covariance matrix
    R       :: AbstractArray   # Measurement noise covariance matrix
    state0  :: Union{Normal,MvNormal}

    function RTSSmoother(A::AbstractArray,
                         C::AbstractArray,
                         Q::AbstractArray,
                         R::AbstractArray, 
                         state0::Union{Normal,MvNormal})

        Dx = size(A,1)
        Dy = size(C,1)

        return new(Dx,Dy,A,C,Q,R,state0)
    end
end

function filter(model::RTSSmoother, signal)

    # Time horizon
    N = length(signal)

    if model.Dx == 1

        a = model.A[1]
        c = model.C[1]
        q = model.Q[1]
        r = model.R[1]

        # Initialize estimate arrays
        m_ = zeros(N)
        s_ = zeros(N)

        # Initial state prior
        m_0 = mean(model.state0)
        s_0 = var( model.state0)

        # Start previous state variable
        m_kmin = m_0
        s_kmin = s_0

        for k = 1:N

            # Forward prediction step
            m_k_pred = a*m_kmin
            s_k_pred = a^2*s_kmin + q

            # Forward update step
            v_k = signal[k] - c*m_k_pred
            r_k = c^2*s_k_pred + r
            k_k = s_k_pred*c/r_k
            m_k = m_k_pred + k_k*v_k
            s_k = s_k_pred - k_k^2*r_k

            if s_k <= 0.0; s_k = 1e-8; end
            
            # Store estimates
            m_[k] = m_k
            s_[k] = s_k

            # Update previous state variable
            m_kmin = m_k
            s_kmin = s_k

        end
        return m_, s_
    else

        # Initialize estimate arrays
        m_ = zeros(model.Dx, N)
        S_ = zeros(model.Dx, model.Dx, N)

        # Initial state prior
        m_0 = mean(model.state0)
        S_0 = cov( model.state0)

        # Start previous state variable
        m_kmin = m_0
        S_kmin = S_0

        for k = 1:N

            # Forward prediction step
            m_k_pred = model.A*m_kmin
            S_k_pred = model.A*S_kmin*model.A' .+ model.Q

            # Forward update step
            v_k = signal[k] .- model.C*m_k_pred
            R_k = model.C*S_k_pred*model.C' .+ model.R
            K_k = S_k_pred*model.C'*inv(R_k)
            m_k = m_k_pred .+ K_k*v_k
            S_k = S_k_pred .- K_k*R_k*K_k'
            
            # Store estimates
            m_[:,k] = m_k
            S_[:,:,k] = S_k

            # Update previous state variable
            m_kmin = m_k
            S_kmin = S_k

        end
        return m_, S_
    end
end

function smoothen(model::RTSSmoother, signal)

    # Time horizon
    N = length(signal)

    if model.Dx == 1

        a = model.A[1]
        q = model.Q[1]

        # Forward pass
        m_,s_ = filter(model,signal)

        # Initialize smoothing estimate arrays
        msk = zeros(N)
        ssk = zeros(N)

        # Smoothed estimates at time horizon
        msk[N] = m_[N]
        ssk[N] = s_[N]

        for k = N-1:-1:1
            
            # Backward prediction
            m_kplus = a*m_[k]
            s_kplus = a^2*s_[k] + q

            if s_kplus < 1e-12; s_kplus = 1e-12; end

            # Backward update step
            g_k = s_[k]*a/s_kplus
            msk[k] = m_[k] + g_k*(msk[k+1] - m_kplus)
            ssk[k] = s_[k] + g_k^2*(ssk[k+1] - s_kplus)

            if ssk[k] <= 0.0; ssk[k] = 1e-12; end

        end
        return msk, ssk
    else

        m_,S_ = filter(model,signal)

        # Initialize smoothing estimate arrays
        msk = zeros(model.Dx, N)
        Ssk = zeros(model.Dx, model.Dx, N)

        # Smoothed estimates at time horizon
        msk[:,N] = m_[:,N]
        Ssk[:,:,N] = S_[:,:,N]

        for k = N-1:-1:1
            
            # Backward prediction
            m_kplus = model.A*m_[:,k]
            S_kplus = model.A*S_[:,:,k]*model.A' .+ model.Q

            # Backward update step
            G_k = S_[:,:,k]*model.A' * inv(S_kplus)
            msk[:,k] = m_[:,k] + G_k*(msk[:,k+1] - m_kplus)
            Ssk[:,:,k] = S_[:,:,k] + G_k*(Ssk[:,:,k+1] - S_kplus)*G_k'

        end
        return msk, Ssk
    end
end

function marginal_likelihood(model::RTSSmoother, signal)

    if model.Dx == 1
        m,s = smoothen(model, signal)
        return prod([pdf(Normal(m[k],s[k]),signal[k]) for k in eachindex(signal)])
    else
        m,S = smoothen(model, signal)
        return prod([pdf(MvNormal(m[:,k],S[:,:,k]),signal[k]) for k in eachindex(signal)])
    end
end

function log_marginal_likelihood(model::RTSSmoother, signal)

    if model.Dx == 1
        m,s = smoothen(model, signal)
        clamp!(s, 1e-12,Inf)
        return sum([logpdf(Normal(m[k],s[k]),signal[k]) for k in eachindex(signal)])
    else
        m,S = smoothen(model, signal)
        return sum([logpdf(MvNormal(m[:,k],S[:,:,k]),signal[k]) for k in eachindex(signal)])
    end
end

end
