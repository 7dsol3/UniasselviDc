struct StrategyMetrics
    name::String
    mean_mse::Float64
    max_mse::Float64
    compression_ratio::Float64
    latency_ms::Float64
    total_bits::Int
    objective::Float64
end

struct QuantComparisonReport
    mixed::StrategyMetrics
    uniform4::StrategyMetrics
    mse_ratio_mixed_over_uniform::Float64
    latency_speedup_uniform_over_mixed::Float64
    created_at::String
    note::String
end

function uniform_allocation(layers::Dict{String, <:AbstractArray}; bits::Int = 4)
    bits in ALLOWED_BITS || throw(ArgumentError("bits uniforme invalido: $bits"))
    alloc = Dict{String, Int}()
    used = 0
    for name in keys(layers)
        alloc[name] = bits
        used += length(vec(layers[name])) * bits
    end
    return AllocationResult(alloc, used, 0.0, used, true)
end

function _fit_vec(x::Vector{Float32}, n::Int)
    lx = length(x)
    if lx == n
        return x
    elseif lx > n
        return x[1:n]
    else
        y = zeros(Float32, n)
        y[1:lx] .= x
        return y
    end
end

function _layer_forward_proxy(layer::QuantizedLayer, x::Vector{Float32})
    w = dequantize_layer(layer)
    if ndims(w) >= 2
        m = reshape(w, size(w, 1), :)
        xin = _fit_vec(x, size(m, 2))
        y = m * xin
        return tanh.(Float32.(y))
    else
        v = Float32.(vec(w))
        xin = _fit_vec(x, length(v))
        return Float32[tanh(dot(v, xin))]
    end
end

function benchmark_proxy_latency_ms(
    qmodel::QuantizedModel;
    iters::Int = 8,
    warmup::Int = 2,
    seed::Int = 1234,
)
    rng = Random.MersenneTwister(seed)
    if isempty(qmodel.layers)
        return 0.0
    end

    input_dim = begin
        s = qmodel.layers[1].shape
        if length(s) >= 2
            Int(s[2])
        else
            max(1, Int(prod(s)))
        end
    end

    for _ in 1:warmup
        x = randn(rng, Float32, input_dim)
        for layer in qmodel.layers
            x = _layer_forward_proxy(layer, x)
        end
    end

    t0 = time_ns()
    for _ in 1:iters
        x = randn(rng, Float32, input_dim)
        for layer in qmodel.layers
            x = _layer_forward_proxy(layer, x)
        end
    end
    elapsed_ns = time_ns() - t0
    return elapsed_ns / 1_000_000 / iters
end

function _strategy_metrics(
    name::String,
    layers::Dict{String, <:AbstractArray},
    qmodel::QuantizedModel,
    alloc::AllocationResult;
    latency_iters::Int = 8,
)
    er = eval_unique(layers, qmodel)
    lat = benchmark_proxy_latency_ms(qmodel; iters = latency_iters)
    return StrategyMetrics(
        name,
        er.mean_mse,
        er.max_mse,
        er.compression_ratio,
        lat,
        alloc.total_bits,
        alloc.objective,
    )
end

function compare_mixed_vs_uniform4(
    layers::Dict{String, <:AbstractArray},
    mixed_alloc::AllocationResult;
    latency_iters::Int = 8,
)
    mixed_q = quantize_mixed(layers, mixed_alloc; model_name = "mixed_authoral")
    uniform_alloc = uniform_allocation(layers; bits = 4)
    uniform_q = quantize_mixed(layers, uniform_alloc; model_name = "uniform4")

    mm = _strategy_metrics("mixed_authoral", layers, mixed_q, mixed_alloc; latency_iters = latency_iters)
    uu = _strategy_metrics("uniform_4bit", layers, uniform_q, uniform_alloc; latency_iters = latency_iters)

    mse_ratio = uu.mean_mse == 0 ? 0.0 : mm.mean_mse / uu.mean_mse
    speedup = mm.latency_ms == 0 ? 0.0 : uu.latency_ms / mm.latency_ms

    return QuantComparisonReport(
        mm,
        uu,
        mse_ratio,
        speedup,
        string(now()),
        "latencia medida por proxy de forward dequant+matmul",
    )
end

function save_comparison_report_md(path::AbstractString, r::QuantComparisonReport)
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        println(io, "# Benchmark: Mixed Autoral vs Uniforme 4-bit")
        println(io)
        println(io, "- criado_em: $(r.created_at)")
        println(io, "- nota: $(r.note)")
        println(io)
        println(io, "| estrategia | mean_mse | max_mse | compressao | latencia_ms | total_bits | objective |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for s in (r.mixed, r.uniform4)
            println(io, "| $(s.name) | $(s.mean_mse) | $(s.max_mse) | $(s.compression_ratio) | $(s.latency_ms) | $(s.total_bits) | $(s.objective) |")
        end
        println(io)
        println(io, "- mse_ratio_mixed_over_uniform: $(r.mse_ratio_mixed_over_uniform)")
        println(io, "- latency_speedup_uniform_over_mixed: $(r.latency_speedup_uniform_over_mixed)")
    end
    return full
end
