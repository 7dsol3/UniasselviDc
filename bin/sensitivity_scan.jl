include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore
using Printf

function _argval(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

input_layers = _argval("--input", "")
output_sens = _argval("--output", joinpath(@__DIR__, "..", "data", "quant", "sensitivity.jls"))
k = parse(Float64, _argval("--k", "0.1"))
n_layers = parse(Int, _argval("--layers", "12"))
rows = parse(Int, _argval("--rows", "256"))
cols = parse(Int, _argval("--cols", "256"))
save_layers_path = _argval("--save-layers", joinpath(@__DIR__, "..", "data", "quant", "layers_fp.jls"))

layers = if !isempty(input_layers) && isfile(input_layers)
    println("Carregando layers de: $(abspath(input_layers))")
    load_layers(input_layers)
else
    println("Gerando layers sinteticos: n_layers=$n_layers shape=($rows,$cols)")
    generate_synthetic_layers(n_layers = n_layers, rows = rows, cols = cols)
end

saved_layers = save_layers(save_layers_path, layers)
println("Layers base salvos em: $saved_layers")

sens = sensitivity_scan(layers; k = k)
save_sensitivity(output_sens, sens)

println("\nScan de sensibilidade concluido.")
println("Saida: $(abspath(output_sens))")
println("Layers: $(length(sens))")

sorted_s = sort(sens, by = s -> s.sensitivity, rev = true)
println("\nTop 5 sensitivities:")
for s in first(sorted_s, min(5, length(sorted_s)))
    @printf("- %-12s S=%.8f n=%d e2=%.8e e4=%.8e e8=%.8e\n",
        s.name, s.sensitivity, s.n_params, s.errors[2], s.errors[4], s.errors[8])
end
