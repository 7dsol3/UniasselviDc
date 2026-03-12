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

function _parse_bits(s::String)
    vals = parse.(Int, split(s, ","))
    for b in vals
        b in ALLOWED_BITS || error("bit invalido: $b; permitidos=$(collect(ALLOWED_BITS))")
    end
    return sort(unique(vals))
end

sens_path = _argval("--sens", joinpath(@__DIR__, "..", "data", "quant", "sensitivity.jls"))
output_alloc = _argval("--output", joinpath(@__DIR__, "..", "data", "quant", "allocation.jls"))
bits = _parse_bits(_argval("--bits", "2,3,4,8"))

sens = load_sensitivity(sens_path)
total_params = sum(s.n_params for s in sens)
default_budget = Int(round(total_params * 3.2)) # media alvo de 3.2 bits por parametro
budget = parse(Int, _argval("--budget", string(default_budget)))

alloc = bit_allocator(sens; budget_bits = budget, bit_choices = bits)
save_allocation(output_alloc, alloc)

println("Bit allocation concluido.")
println("Saida: $(abspath(output_alloc))")
println("Feasible: $(alloc.feasible)")
println("Budget: $(alloc.budget_bits)")
println("Used:   $(alloc.total_bits)")
@printf("Objective: %.10e\n", alloc.objective)

if alloc.feasible
    counts = Dict{Int, Int}(b => 0 for b in bits)
    for b in values(alloc.bits)
        counts[b] = get(counts, b, 0) + 1
    end
    println("\nDistribuicao de bits por camada:")
    for b in bits
        println("- $b-bit: $(counts[b]) camada(s)")
    end
end
