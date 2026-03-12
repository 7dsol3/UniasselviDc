include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore

cfg = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "config", "default.toml")
IANeuralCore.run_cli(cfg)
