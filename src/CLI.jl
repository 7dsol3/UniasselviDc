function _default_config()
    return Dict{String, Any}(
        "backend" => Dict(
            "type" => "unique_quant",
            "executable" => "..\\llama.cpp\\build\\bin\\llama-cli.exe",
            "model" => "..\\models\\model.gguf",
            "quant_model" => "..\\data\\real\\model_mixed.jls",
            "max_layers" => 8,
            "n_ctx" => 4096,
            "n_predict" => 512,
            "temperature" => 0.3,
        ),
        "agent" => Dict(
            "system_prompt" => "Voce e uma IA tecnica e pragmatica. Diga quando nao souber.",
            "max_history" => 20,
            "retrieval_topk" => 4,
            "temperature" => 0.3,
            "max_tokens" => 512,
        ),
    )
end

function _load_config(path::AbstractString)
    full = abspath(String(path))
    if !isfile(full)
        return _default_config(), full
    end
    cfg = TOML.parsefile(full)
    merged = _default_config()
    for (section, values) in cfg
        if section in keys(merged) && values isa AbstractDict
            for (k, v) in values
                merged[section][k] = v
            end
        else
            merged[section] = values
        end
    end
    return merged, full
end

function _resolve_relative(base_cfg_path::String, maybe_relative::String)
    isabspath(maybe_relative) && return maybe_relative
    return normpath(joinpath(dirname(base_cfg_path), maybe_relative))
end

function _build_backend(cfg::Dict{String, Any}, cfg_path::String)
    b = cfg["backend"]
    btype = lowercase(String(get(b, "type", "rule")))
    if btype == "llamacpp"
        exe = _resolve_relative(cfg_path, String(get(b, "executable", "llama-cli")))
        model = _resolve_relative(cfg_path, String(get(b, "model", "model.gguf")))
        return LlamaCppBackend(
            exe,
            model,
            Int(get(b, "n_ctx", 4096)),
            Int(get(b, "n_predict", 512)),
            Float64(get(b, "temperature", 0.3)),
        )
    elseif btype == "unique_quant"
        qmodel = _resolve_relative(cfg_path, String(get(b, "quant_model", "..\\data\\real\\model_mixed.jls")))
        return UniqueQuantBackend(
            qmodel,
            Int(get(b, "max_layers", 8)),
        )
    elseif btype == "ollama"
        return OllamaBackend(String(get(b, "model", "llama3.1")))
    else
        return RuleBackend()
    end
end

function _build_agent_config(cfg::Dict{String, Any})
    a = cfg["agent"]
    return AgentConfig(
        system_prompt = String(get(a, "system_prompt", "Voce e uma IA tecnica e pragmatica.")),
        max_history = Int(get(a, "max_history", 20)),
        retrieval_topk = Int(get(a, "retrieval_topk", 4)),
        temperature = Float64(get(a, "temperature", 0.3)),
        max_tokens = Int(get(a, "max_tokens", 512)),
    )
end

function _print_help()
    println("Comandos:")
    println("  /help                  mostrar ajuda")
    println("  /add <arquivo|pasta>   indexar conteudo na memoria vetorial")
    println("  /search <query>        buscar contexto indexado")
    println("  /ls [pasta]            listar arquivos")
    println("  /cat <arquivo>         ler arquivo")
    println("  /grep <padrao> [pasta] buscar texto")
    println("  /history               mostrar historico atual")
    println("  /clear                 limpar historico")
    println("  /save <arquivo.bin>    salvar estado do agente")
    println("  /load <arquivo.bin>    carregar estado do agente")
    println("  /exit                  sair")
end

function _show_history(agent::NeuralAgent)
    if isempty(agent.history)
        println("(historico vazio)")
        return
    end
    for m in agent.history
        println("[$(m.role)] $(m.content)")
    end
end

function _parse_two_args(raw::String)
    parts = split(strip(raw), ' ', keepempty = false)
    if length(parts) <= 1
        return "", ""
    elseif length(parts) == 2
        return parts[2], ""
    end
    return parts[2], join(parts[3:end], " ")
end

function run_cli(config_path::AbstractString = joinpath(@__DIR__, "..", "config", "default.toml"))
    cfg, cfg_file = _load_config(config_path)
    backend = _build_backend(cfg, cfg_file)
    agent_cfg = _build_agent_config(cfg)
    agent = NeuralAgent(backend; config = agent_cfg, memory = MemoryStore())

    println(repeat("=", 72))
    println("IA_Neural_Julia :: Chat CLI")
    println("Backend: $(typeof(backend))")
    println("Config:  $cfg_file")
    println("Digite /help para comandos.")
    println(repeat("=", 72))

    while true
        print("\nvoce> ")
        line = try
            readline()
        catch
            println("\nEncerrando.")
            break
        end
        line = strip(line)
        isempty(line) && continue

        if line == "/exit"
            println("Encerrando.")
            break
        elseif line == "/help"
            _print_help()
            continue
        elseif startswith(line, "/add ")
            path = strip(line[6:end])
            chunks = add_path!(agent.memory, path)
            println("Indexacao concluida: $chunks chunk(s) adicionados.")
            continue
        elseif startswith(line, "/search ")
            query = strip(line[9:end])
            hits = search(agent.memory, query; topk = agent.config.retrieval_topk)
            if isempty(hits)
                println("(sem resultados)")
            else
                for (i, h) in enumerate(hits)
                    println("[$i] score=$(round(h.score, digits=3)) fonte=$(h.source)")
                    if isempty(h.text)
                        println("(texto vazio)")
                    else
                        limit = min(lastindex(h.text), 220)
                        println(h.text[1:limit])
                    end
                    println()
                end
            end
            continue
        elseif startswith(line, "/ls")
            arg = strip(replace(line, "/ls" => ""))
            println(tool_list_dir(isempty(arg) ? "." : arg))
            continue
        elseif startswith(line, "/cat ")
            path = strip(line[6:end])
            println(tool_read_file(path))
            continue
        elseif startswith(line, "/grep ")
            a, b = _parse_two_args(line)
            if isempty(a)
                println("Uso: /grep <padrao> [pasta]")
            else
                root = isempty(b) ? "." : b
                println(tool_grep(a, root))
            end
            continue
        elseif line == "/history"
            _show_history(agent)
            continue
        elseif line == "/clear"
            clear_history!(agent)
            println("Historico limpo.")
            continue
        elseif startswith(line, "/save ")
            path = strip(line[7:end])
            saved = save_agent!(agent, path)
            println("Estado salvo em: $saved")
            continue
        elseif startswith(line, "/load ")
            path = strip(line[7:end])
            load_agent_state!(agent, path)
            println("Estado carregado.")
            continue
        end

        reply, hits = respond!(agent, line)
        println("\nia> $reply")
        if !isempty(hits)
            sources = unique([h.source for h in hits])
            shown = first(sources, min(length(sources), 3))
            println("contexto: " * join(shown, " | "))
        end
    end
    return nothing
end
