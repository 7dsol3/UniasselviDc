const ALLOWED_BITS = (2, 3, 4, 8)

struct LayerSensitivity
    name::String
    n_params::Int
    sensitivity::Float64
    errors::Dict{Int, Float64}
end

struct QuantizedLayer
    name::String
    bits::Int
    shape::Tuple
    scale::Float32
    qvalues::Vector{Int8}
end

struct QuantizedModel
    layers::Vector{QuantizedLayer}
    metadata::Dict{String, Any}
end

struct AllocationResult
    bits::Dict{String, Int}
    total_bits::Int
    objective::Float64
    budget_bits::Int
    feasible::Bool
end

struct EvalReport
    layer_mse::Dict{String, Float64}
    weighted_error::Float64
    mean_mse::Float64
    max_mse::Float64
    total_params::Int
    model_size_bits_fp16::Int
    model_size_bits_quant::Int
    compression_ratio::Float64
    igato_start::Float64
    igato_end::Float64
end

function _to_f32_array(x)
    return Float32.(x)
end

function generate_synthetic_layers(;
    n_layers::Int = 12,
    rows::Int = 512,
    cols::Int = 512,
    seed::Int = 1234,
)
    rng = Random.MersenneTwister(seed)
    layers = Dict{String, Array{Float32}}()
    for i in 1:n_layers
        name = @sprintf("layer_%02d", i)
        std = 0.02f0 + 0.002f0 * i
        layers[name] = randn(rng, Float32, rows, cols) .* std
    end
    return layers
end

function save_layers(path::AbstractString, layers::Dict{String, <:AbstractArray})
    full = abspath(String(path))
    mkpath(dirname(full))
    serializable = Dict{String, Any}()
    for (k, w) in layers
        serializable[k] = _to_f32_array(w)
    end
    open(full, "w") do io
        serialize(io, serializable)
    end
    return full
end

function load_layers(path::AbstractString)
    full = abspath(String(path))
    isfile(full) || throw(ArgumentError("arquivo de layers nao encontrado: $full"))
    data = open(full, "r") do io
        deserialize(io)
    end
    out = Dict{String, Array{Float32}}()
    for (k, v) in data
        out[String(k)] = _to_f32_array(v)
    end
    return out
end

function _surrogate_gradients(w::AbstractArray{<:Real})
    fw = Float64.(vec(w))
    centered = fw .- mean(fw)
    g = abs.(centered) .+ 0.1 .* abs.(fw)
    return g
end

function _quantize_dequantize_flat(w::Vector{Float64}, bits::Int)
    bits in ALLOWED_BITS || throw(ArgumentError("bits nao suportado: $bits"))
    qmax = max(1, Int(2^(bits - 1) - 1))
    maxabs = maximum(abs.(w))
    if maxabs == 0
        return fill(0, length(w)), fill(0.0, length(w)), 1.0f0
    end
    scale = Float32(maxabs / qmax)
    q = round.(Int16, clamp.(w ./ scale, -qmax, qmax))
    deq = Float64.(q) .* Float64(scale)
    return q, deq, scale
end

function quantization_error_mse(w::AbstractArray{<:Real}, bits::Int)
    flat = Float64.(vec(w))
    _, deq, _ = _quantize_dequantize_flat(flat, bits)
    return mean((flat .- deq) .^ 2)
end

function sensitivity_scan(
    layers::Dict{String, <:AbstractArray};
    gradients::Union{Nothing, Dict{String, <:AbstractArray}} = nothing,
    k::Float64 = 0.1,
    bit_choices::Vector{Int} = collect(ALLOWED_BITS),
)
    k < 0 && throw(ArgumentError("k deve ser >= 0"))
    out = LayerSensitivity[]

    for name in sort(collect(keys(layers)))
        w = Float64.(vec(layers[name]))
        g = if !isnothing(gradients) && haskey(gradients, name)
            abs.(Float64.(vec(gradients[name])))
        else
            _surrogate_gradients(w)
        end

        if length(g) != length(w)
            throw(ArgumentError("gradiente com shape incompativel para layer $name"))
        end

        s = mean(abs.(g) ./ (1 .+ k .* abs.(w)))
        errs = Dict{Int, Float64}()
        for b in bit_choices
            errs[b] = quantization_error_mse(w, b)
        end
        push!(out, LayerSensitivity(name, length(w), s, errs))
    end
    return out
end

function save_sensitivity(path::AbstractString, sens::Vector{LayerSensitivity})
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        serialize(io, sens)
    end
    return full
end

function load_sensitivity(path::AbstractString)
    full = abspath(String(path))
    isfile(full) || throw(ArgumentError("arquivo de sensibilidade nao encontrado: $full"))
    return open(full, "r") do io
        deserialize(io)
    end
end

function _prune_states(states::Dict{Int, Tuple{Float64, Vector{Int}}}; max_states::Int = 8000)
    if length(states) <= max_states
        return states
    end
    sorted = sort(collect(states), by = x -> x[2][1])
    keep = first(sorted, max_states)
    return Dict{Int, Tuple{Float64, Vector{Int}}}(keep)
end

function bit_allocator(
    sens::Vector{LayerSensitivity};
    budget_bits::Int,
    bit_choices::Vector{Int} = collect(ALLOWED_BITS),
)
    isempty(sens) && throw(ArgumentError("lista de sensitivities vazia"))
    min_cost = sum(s.n_params * minimum(bit_choices) for s in sens)
    if budget_bits < min_cost
        return AllocationResult(Dict{String, Int}(), 0, Inf, budget_bits, false)
    end

    states = Dict{Int, Tuple{Float64, Vector{Int}}}()
    states[0] = (0.0, Int[])

    for s in sens
        nxt = Dict{Int, Tuple{Float64, Vector{Int}}}()
        for (used, (obj, choice_path)) in states
            for b in bit_choices
                haskey(s.errors, b) || continue
                cost = used + s.n_params * b
                cost > budget_bits && continue
                new_obj = obj + s.sensitivity * s.errors[b]
                existing = get(nxt, cost, (Inf, Int[]))
                if new_obj < existing[1]
                    nxt[cost] = (new_obj, [choice_path; b])
                end
            end
        end
        states = _prune_states(nxt)
    end

    isempty(states) && return AllocationResult(Dict{String, Int}(), 0, Inf, budget_bits, false)

    best_cost = -1
    best_obj = Inf
    best_path = Int[]
    for (cost, (obj, path)) in states
        if obj < best_obj || (obj == best_obj && cost > best_cost)
            best_obj = obj
            best_cost = cost
            best_path = path
        end
    end

    bits = Dict{String, Int}()
    for (i, s) in enumerate(sens)
        bits[s.name] = best_path[i]
    end
    return AllocationResult(bits, best_cost, best_obj, budget_bits, true)
end

function save_allocation(path::AbstractString, alloc::AllocationResult)
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        serialize(io, alloc)
    end
    return full
end

function load_allocation(path::AbstractString)
    full = abspath(String(path))
    isfile(full) || throw(ArgumentError("arquivo de allocation nao encontrado: $full"))
    return open(full, "r") do io
        deserialize(io)
    end
end

function _quantize_layer(name::String, w::AbstractArray{<:Real}, bits::Int)
    flat = Float64.(vec(w))
    q, _, scale = _quantize_dequantize_flat(flat, bits)
    q8 = Int8.(q)
    return QuantizedLayer(name, bits, size(w), scale, q8)
end

function dequantize_layer(layer::QuantizedLayer)
    qf = Float32.(layer.qvalues) .* layer.scale
    return reshape(qf, layer.shape)
end

function quantize_mixed(
    layers::Dict{String, <:AbstractArray},
    alloc::AllocationResult;
    model_name::String = "unique_quant_model",
)
    alloc.feasible || throw(ArgumentError("allocation inviavel"))
    qlayers = QuantizedLayer[]
    for name in sort(collect(keys(layers)))
        haskey(alloc.bits, name) || throw(ArgumentError("layer sem bit alocado: $name"))
        b = alloc.bits[name]
        push!(qlayers, _quantize_layer(name, layers[name], b))
    end
    meta = Dict{String, Any}(
        "name" => model_name,
        "created_at" => string(now()),
        "total_bits" => alloc.total_bits,
        "budget_bits" => alloc.budget_bits,
        "objective" => alloc.objective,
    )
    return QuantizedModel(qlayers, meta)
end

function save_quantized_model(path::AbstractString, qm::QuantizedModel)
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        serialize(io, qm)
    end
    return full
end

function load_quantized_model(path::AbstractString)
    full = abspath(String(path))
    isfile(full) || throw(ArgumentError("arquivo quantizado nao encontrado: $full"))
    return open(full, "r") do io
        deserialize(io)
    end
end

function indice_gato(
    t::Real;
    alpha::Float64 = inv(sqrt(2)),
    beta::Float64 = inv(sqrt(2)),
    gamma0::Float64 = 0.02,
    lambdaR::Float64 = 1e-3,
    R::Float64 = 0.0,
    k::Float64 = 0.1,
    E::Float64 = 1.0,
)
    Γ = gamma0 + lambdaR * abs(R)
    return 2 * abs(alpha * beta) * exp(-Γ * Float64(t)) / (1 + k * abs(E))
end

function igato_regularization(
    t::Real;
    base_noise::Float64 = 1.0,
    eta::Float64 = 1.0,
    kwargs...,
)
    i = indice_gato(t; kwargs...)
    return base_noise * (1 + eta * (1 - i))
end

function eval_unique(
    baseline_layers::Dict{String, <:AbstractArray},
    qmodel::QuantizedModel;
    fp_bits::Int = 16,
    t0::Float64 = 0.0,
    t1::Float64 = 10.0,
    igato_kwargs::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
)
    layer_mse = Dict{String, Float64}()
    total = 0
    mse_acc = 0.0
    max_mse = 0.0
    weighted_acc = 0.0
    quant_bits = 0

    for ql in qmodel.layers
        haskey(baseline_layers, ql.name) || continue
        base = Float32.(baseline_layers[ql.name])
        dq = dequantize_layer(ql)
        mse = mean((Float64.(vec(base)) .- Float64.(vec(dq))) .^ 2)
        layer_mse[ql.name] = mse
        n = length(base)
        total += n
        mse_acc += mse * n
        weighted_acc += mse * ql.bits * n
        max_mse = max(max_mse, mse)
        quant_bits += n * ql.bits
    end

    fp_total = total * fp_bits
    ig0 = isempty(igato_kwargs) ? indice_gato(t0) : indice_gato(t0; igato_kwargs...)
    ig1 = isempty(igato_kwargs) ? indice_gato(t1) : indice_gato(t1; igato_kwargs...)

    return EvalReport(
        layer_mse,
        total == 0 ? 0.0 : weighted_acc / (total * 8),
        total == 0 ? 0.0 : mse_acc / total,
        max_mse,
        total,
        fp_total,
        quant_bits,
        quant_bits == 0 ? 0.0 : fp_total / quant_bits,
        ig0,
        ig1,
    )
end

function save_eval_report(path::AbstractString, r::EvalReport)
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        serialize(io, r)
    end
    return full
end
