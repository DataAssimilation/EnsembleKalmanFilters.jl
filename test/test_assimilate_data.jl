@testset "assimilate_data" begin
    R = reshape([2.0], (1, 1))
    filter = EnKF(R, false, 0)

    # Two ensemble members with value ±1.
    prior_state = [1.0 -1.0]

    # Identity observation operator with no noise.
    prior_obs_clean = [1.0 -1.0]
    prior_obs_noisy = [1.0 -1.0]

    # True state is the mean of the prior.
    y_obs = [0.0]

    # Assimilate.
    posterior = assimilate_data(
        filter, prior_state, prior_obs_clean, prior_obs_noisy, y_obs
    )

    # Assimilation should change each ensemble member the same amount towards zero. 
    @test posterior[1] == -posterior[2]

    # The sample prior covariance and sample observation covariance are both 2, so the result 
    #   is weighted evenly between the prior and y_obs.  
    @test posterior[1] == 0.5
end
