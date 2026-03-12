mutable struct NeuralAgent
    backend::AbstractBackend
    memory::MemoryStore
    config::AgentConfig
    history::Vector{ChatMessage}
end

function NeuralAgent(
    backend::AbstractBackend;
    memory::MemoryStore = MemoryStore(),
    config::AgentConfig = AgentConfig(),
)
    return NeuralAgent(backend, memory, config, ChatMessage[])
end

function _push_history!(agent::NeuralAgent, role::Symbol, content::AbstractString)
    push!(agent.history, ChatMessage(role, String(content)))
    while length(agent.history) > agent.config.max_history
        popfirst!(agent.history)
    end
end

function clear_history!(agent::NeuralAgent)
    empty!(agent.history)
    return nothing
end

function _history_block(agent::NeuralAgent; max_items::Int = 10)
    if isempty(agent.history)
        return "(sem historico)"
    end
    tail = first(reverse(agent.history), min(max_items, length(agent.history)))
    lines = String[]
    for msg in reverse(tail)
        push!(lines, string(msg.role, ": ", msg.content))
    end
    return join(lines, "\n")
end

function _context_block(agent::NeuralAgent, query::AbstractString)
    hits = search(agent.memory, String(query); topk = agent.config.retrieval_topk)
    if isempty(hits)
        return "(sem contexto recuperado)", hits
    end
    blocks = String[]
    for h in hits
        header = @sprintf("[fonte=%s score=%.3f]", h.source, h.score)
        push!(blocks, header * "\n" * h.text)
    end
    return join(blocks, "\n\n"), hits
end

function _build_prompt(agent::NeuralAgent, user_input::AbstractString)
    context, hits = _context_block(agent, user_input)
    hist = _history_block(agent)
    prompt = """
System:
$(agent.config.system_prompt)

Contexto:
$(context)

Historico:
$(hist)

Usuario: $(user_input)
Assistente:
"""
    return prompt, hits
end

function respond!(agent::NeuralAgent, user_input::AbstractString)
    user = strip(String(user_input))
    isempty(user) && return "(entrada vazia)", NamedTuple[]

    _push_history!(agent, :usuario, user)
    prompt, hits = _build_prompt(agent, user)
    answer = generate(
        agent.backend,
        prompt;
        temperature = agent.config.temperature,
        max_tokens = agent.config.max_tokens,
    )
    _push_history!(agent, :assistente, answer)
    return answer, hits
end

function save_agent!(agent::NeuralAgent, path::AbstractString)
    full = abspath(String(path))
    mkpath(dirname(full))
    data = Dict{String, Any}(
        "history" => agent.history,
        "memory" => agent.memory,
    )
    open(full, "w") do io
        serialize(io, data)
    end
    return full
end

function load_agent_state!(agent::NeuralAgent, path::AbstractString)
    full = abspath(String(path))
    isfile(full) || throw(ArgumentError("arquivo nao encontrado: $full"))
    data = open(full, "r") do io
        deserialize(io)
    end
    haskey(data, "history") && (agent.history = data["history"])
    haskey(data, "memory") && (agent.memory = data["memory"])
    return agent
end
