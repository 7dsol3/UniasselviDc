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

layers_path = _argval("--layers", joinpath(@__DIR__, "..", "data", "quant", "layers_fp.jls"))
alloc_path = _argval("--alloc", joinpath(@__DIR__, "..", "data", "quant", "allocation.jls"))
output_q = _argval("--output", joinpath(@__DIR__, "..", "data", "quant", "model_mixed.jls"))
model_name = _argval("--name", "unique_model_mixed")

layers = load_layers(layers_path)
alloc = load_allocation(alloc_path)
qm = quantize_mixed(layers, alloc; model_name = model_name)
save_quantized_model(output_q, qm)

total_params = sum(length(vec(v)) for v in values(layers))
fp16_bits = total_params * 16
ratio = fp16_bits / alloc.total_bits

println("Quantizacao mista concluida.")
println("Saida: $(abspath(output_q))")
println("Layers quantizados: $(length(qm.layers))")
println("Bits usados: $(alloc.total_bits)")
@printf("Compressao FP16->mixed: %.4fx\n", ratio)
