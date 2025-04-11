using Random: Random
using LinearAlgebra
using IterativeSolvers: cg, cg!, CGStateVariables
using JOLI: joMatrix
using Statistics

export assimilate_data

T_LOG = Union{<:AbstractDict,<:Nothing}

function assimilate_data(
    filter::EnKF, prior_state::T1, prior_obs::T2, y_obs::Ty, log_data::T_LOG=nothing
) where {T1<:AbstractArray,T2<:AbstractArray,Ty<:Union{<:AbstractArray,<:Number}}
    return assimilate_data(filter, prior_state, prior_obs, prior_obs, y_obs, log_data)
end

_ensure_1d(a::T) where {T<:Number} = T[a]
_ensure_1d(a::AbstractArray{T,1}) where {T} = a

_ensure_2d(a::T) where {T<:Number} = T[a;;]
_ensure_2d(a::AbstractArray{T,1}) where {T} = reshape(a, (1, size(a)...))
_ensure_2d(a::AbstractArray{T,2}) where {T} = a

function assimilate_data(
    filter::EnKF,
    prior_state::T1,
    prior_obs_clean::T2,
    prior_obs_noisy::T3,
    y_obs::Ty,
    log_data::T_LOG=nothing,
) where {
    T1<:AbstractArray,
    T2<:AbstractArray,
    T3<:AbstractArray,
    Ty<:Union{<:AbstractArray,<:Number},
}
    prior_state = _ensure_2d(prior_state)
    prior_obs_clean = _ensure_2d(prior_obs_clean)
    prior_obs_noisy = _ensure_2d(prior_obs_noisy)
    y_obs = _ensure_1d(y_obs)
    return assimilate_data(
        filter, prior_state, prior_obs_clean, prior_obs_noisy, y_obs, log_data
    )
end

function assimilate_data(
    filter::EnKF,
    prior_state::AbstractArray{T1,2},
    prior_obs_clean::AbstractArray{T2,2},
    prior_obs_noisy::AbstractArray{T3,2},
    y_obs::AbstractArray{T4,1},
    log_data::T_LOG=nothing,
) where {T1<:Number,T2<:Number,T3<:Number,T4<:Number}
    X = Float64.(prior_state)
    x_mean = mean(X; dims=2)
    dX = X .- x_mean
    if filter.rho != 0
        # Inflate prior covariance by `1 + filter.rho`.
        dX .*= sqrt(1 + filter.rho)
        X .= x_mean .+ dX
    end

    Y = Float64.(prior_obs_clean)
    Y_noisy = Float64.(prior_obs_noisy)

    y_mean = mean(Y; dims=2)
    if filter.include_noise_in_y_covariance
        dY = Y_noisy .- y_mean
    else
        dY = Y .- y_mean
    end

    nx, N = size(X)
    ny = size(Y, 1)

    y_obs = Float64.(y_obs)
    pred_err = y_obs .- Y_noisy

    dY_op = joMatrix(dY)
    dX_op = joMatrix(dX)
    R_op = get_noise_covariance_operator(ny, filter.R)

    ## dX and dY are typically divided by sqrt(N - 1), but I prefer moving that to R.
    ##   (dX dY' / a) (dY dY' / a + R)^{-1}
    ## ==   dX dY'   a^{-1}(dY dY' / a + R)^{-1}
    ## ==   dX dY'   (a(dY dY' / a + R))^{-1}
    ## ==   dX dY'   (dY dY' + a * R)^{-1}
    obs_covariance = dY_op * dY_op' + (N - 1) * R_op
    cross_covariance = dX_op * dY_op'

    ## @time X_update = dX_op * dY_op' * (obs_covariance \ pred_err)
    Y_update = zeros(size(pred_err))
    statevars = CGStateVariables(similar(y_obs), similar(y_obs), similar(y_obs))
    if isnothing(log_data)
        for i in 1:N
            Y_update_i = view(Y_update, :, i)
            cg!(Y_update_i, obs_covariance, pred_err[:, i]; initially_zero=true, statevars)
        end
    else
        log_data[:assimilate_data_linear_solve] = []
        for i in 1:N
            Y_update_i = view(Y_update, :, i)
            _, history = cg!(
                Y_update_i,
                obs_covariance,
                pred_err[:, i];
                initially_zero=true,
                statevars,
                log=true,
            )
            push!(log_data[:assimilate_data_linear_solve], history)
        end
    end
    X .+= dX * dY' * Y_update
    return X
end

get_noise_covariance_operator(n, R) = joMatrix(R)
get_noise_covariance_operator(n, R::joMatrix) = R
function get_noise_covariance_operator(n, R::T) where {T<:Number}
    return joMatrix(Diagonal(fill(R, n)))
end
