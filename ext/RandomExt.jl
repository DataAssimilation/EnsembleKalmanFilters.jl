"This module extends EnsembleKalmanFilter with functionality from Random."
module RandomExt

using EnsembleKalmanFilter: EnsembleKalmanFilter
using Random

"""
    greeting()

Call [`EnsembleKalmanFilter.greeting`](@ref) with a random name.


# Examples

```jldoctest
julia> @test true;

```

"""
EnsembleKalmanFilter.greeting() = EnsembleKalmanFilter.greeting(rand(5))

end
