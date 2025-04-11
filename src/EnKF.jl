export EnKF

struct EnKF
    R::Any
    include_noise_in_y_covariance::Any
    rho
end

function EnKF(R; params)
    include_noise_in_y_covariance = params["include_noise_in_y_covariance"]
    rho = params["multiplicative_prior_inflation"]
    return EnKF(R, include_noise_in_y_covariance, rho)
end
