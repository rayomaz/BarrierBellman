### Algorithm types
abstract type StochasticBarrierAlgorithm end
abstract type ConstantBarrierAlgorithm <: StochasticBarrierAlgorithm end
abstract type SumOfSquaresBarrierAlgorithm <: StochasticBarrierAlgorithm end

# Result types
abstract type BarrierResult end
psafe(res::BarrierResult, time_horizon) = 1.0 - (eta(res) + beta(res) * time_horizon)