mutable struct ChatMessage
    role::Symbol
    content::String
    timestamp::DateTime
end

ChatMessage(role::Symbol, content::String) = ChatMessage(role, content, now())

Base.show(io::IO, msg::ChatMessage) = print(io, "[$(msg.role)] $(msg.content)")

mutable struct AgentConfig
    system_prompt::String
    max_history::Int
    retrieval_topk::Int
    temperature::Float64
    max_tokens::Int
end

function AgentConfig(;
    system_prompt::String = "Voce e uma IA tecnica, objetiva e segura. Responda em portugues com foco em execucao.",
    max_history::Int = 20,
    retrieval_topk::Int = 4,
    temperature::Float64 = 0.3,
    max_tokens::Int = 512,
)
    return AgentConfig(system_prompt, max_history, retrieval_topk, temperature, max_tokens)
end
