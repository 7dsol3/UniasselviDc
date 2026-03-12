abstract type AbstractBackend end

struct RuleBackend <: AbstractBackend end

struct LlamaCppBackend <: AbstractBackend
    executable::String
    model::String
    n_ctx::Int
    n_predict::Int
    temperature::Float64
end

struct OllamaBackend <: AbstractBackend
    model::String
end

struct UniqueQuantBackend <: AbstractBackend
    model_path::String
    max_layers::Int
end

const _UNIQUE_QMODEL_CACHE = Dict{String, QuantizedModel}()

function _extract_latest_user(prompt::String)
    m = match(r"Usuario:\s*(.*)\nAssistente:\s*$"s, prompt)
    return isnothing(m) ? strip(prompt) : strip(m.captures[1])
end

function generate(::RuleBackend, prompt::String; temperature::Float64 = 0.3, max_tokens::Int = 512)
    _ = temperature
    _ = max_tokens
    user = _extract_latest_user(prompt)
    return "Modo local (RuleBackend) ativo. Recebi: \"$(user)\". Para resposta de LLM real, configure backend = llamacpp ou ollama."
end

function _load_unique_qmodel(path::String)
    full = abspath(path)
    if haskey(_UNIQUE_QMODEL_CACHE, full)
        return _UNIQUE_QMODEL_CACHE[full]
    end
    q = load_quantized_model(full)
    _UNIQUE_QMODEL_CACHE[full] = q
    return q
end

function _hash_embed_text(text::String, n::Int)
    n <= 0 && return Float32[]
    x = zeros(Float32, n)
    for tok in split(lowercase(replace(text, r"[^\p{L}\p{N}]+" => " ")))
        isempty(tok) && continue
        idx = mod(hash(tok), n) + 1
        x[idx] += 1f0
    end
    d = sqrt(sum(abs2, x))
    d > 0 && (x ./= d)
    return x
end

function _fit_vec_backend(x::Vector{Float32}, n::Int)
    if length(x) == n
        return x
    elseif length(x) > n
        return x[1:n]
    end
    y = zeros(Float32, n)
    y[1:length(x)] .= x
    return y
end

function _forward_unique_backend(q::QuantizedModel, user::AbstractString, max_layers::Int)
    isempty(q.layers) && return Float32[]
    s = q.layers[1].shape
    in_dim = length(s) >= 2 ? Int(s[2]) : max(1, Int(prod(s)))
    x = _hash_embed_text(String(user), in_dim)

    for layer in first(q.layers, min(max_layers, length(q.layers)))
        w = dequantize_layer(layer)
        if ndims(w) >= 2
            m = reshape(w, size(w, 1), :)
            xin = _fit_vec_backend(x, size(m, 2))
            x = tanh.(Float32.(m * xin))
        else
            v = Float32.(vec(w))
            xin = _fit_vec_backend(x, length(v))
            x = Float32[tanh(dot(v, xin))]
        end
    end
    return x
end

function _choose_response_mode(user::AbstractString, state::Vector{Float32})
    lower = lowercase(String(user))
    if occursin("codigo", lower) || occursin("script", lower) || occursin("julia", lower)
        return :code
    elseif occursin("erro", lower) || occursin("bug", lower) || occursin("falha", lower)
        return :debug
    elseif occursin("benchmark", lower) || occursin("latencia", lower) || occursin("mse", lower)
        return :metrics
    end
    isempty(state) && return :steps
    marker = abs(sum(state))
    bucket = mod(Int(floor(marker * 1000)), 4)
    return bucket == 0 ? :steps : bucket == 1 ? :summary : bucket == 2 ? :code : :metrics
end

function generate(backend::UniqueQuantBackend, prompt::String; temperature::Float64 = 0.3, max_tokens::Int = 512)
    _ = temperature
    _ = max_tokens
    user = _extract_latest_user(prompt)
    model_file = abspath(backend.model_path)
    isfile(model_file) || return "Erro: modelo quantizado .jls nao encontrado em \"$(backend.model_path)\". Rode o pipeline_unique.jl primeiro."

    q = try
        _load_unique_qmodel(model_file)
    catch err
        return "Erro ao carregar modelo quantizado: $(err)"
    end

    state = _forward_unique_backend(q, user, backend.max_layers)
    mode = _choose_response_mode(user, state)
    n_layers = length(q.layers)
    bits_avg = n_layers == 0 ? 0.0 : mean([Float64(l.bits) for l in q.layers])

    if mode == :code
        return "UniqueQuantBackend ativo (layers=$(n_layers), avg_bits=$(round(bits_avg, digits=2))).\nSugestao objetiva: implemente em Julia com teste automatizado e benchmark.\nProximo comando: rode `pipeline_unique.jl` e depois `benchmark_compare.jl`."
    elseif mode == :debug
        return "UniqueQuantBackend ativo. Para depurar: 1) valide import_safetensors, 2) valide allocation factivel, 3) compare mixed vs uniform4 em MSE e latencia."
    elseif mode == :metrics
        return "UniqueQuantBackend ativo. Foque em metricas: mean_mse, max_mse, compressao, latency_ms. Decisao boa = menor MSE com compressao e latencia aceitaveis."
    elseif mode == :summary
        return "UniqueQuantBackend ativo. Resumo: use sensibilidade por camada, aloque bits com orcamento e valide contra baseline uniforme 4-bit."
    else
        return "UniqueQuantBackend ativo. Passos: (1) importar pesos reais, (2) sensitivity_scan, (3) bit_allocator, (4) quantize_mixed, (5) benchmark_compare."
    end
end

function _resolve_executable(bin::String)
    if isfile(bin)
        return abspath(bin)
    end
    found = Sys.which(bin)
    return isnothing(found) ? "" : found
end

function generate(backend::LlamaCppBackend, prompt::String; temperature::Float64 = backend.temperature, max_tokens::Int = backend.n_predict)
    exe = _resolve_executable(backend.executable)
    isempty(exe) && return "Erro: executavel do llama.cpp nao encontrado em \"$(backend.executable)\"."
    isfile(backend.model) || return "Erro: modelo GGUF nao encontrado em \"$(backend.model)\"."

    cmd = Cmd([
        exe,
        "-m", backend.model,
        "-p", prompt,
        "-n", string(max_tokens),
        "-c", string(backend.n_ctx),
        "--temp", string(temperature),
        "--no-display-prompt",
    ])

    return try
        strip(read(cmd, String))
    catch err
        "Erro ao executar llama.cpp: $(err)"
    end
end

function generate(backend::OllamaBackend, prompt::String; temperature::Float64 = 0.3, max_tokens::Int = 512)
    _ = temperature
    _ = max_tokens
    exe = _resolve_executable("ollama")
    isempty(exe) && return "Erro: ollama nao encontrado no PATH."
    cmd = Cmd([exe, "run", backend.model, prompt])
    return try
        strip(read(cmd, String))
    catch err
        "Erro ao executar ollama: $(err)"
    end
end
