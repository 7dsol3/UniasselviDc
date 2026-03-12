function tool_list_dir(path::AbstractString = ".")
    full = abspath(String(path))
    isdir(full) || return "Diretorio nao encontrado: $full"
    items = sort(readdir(full))
    isempty(items) && return "(vazio)"
    return join(items, "\n")
end

function tool_read_file(path::AbstractString; max_chars::Int = 8_000)
    full = abspath(String(path))
    isfile(full) || return "Arquivo nao encontrado: $full"
    text = try
        read(full, String)
    catch err
        return "Erro ao ler arquivo: $(err)"
    end
    if length(text) > max_chars
        return text[1:max_chars] * "\n\n...[cortado]..."
    end
    return text
end

function tool_grep(pattern::AbstractString, root::AbstractString = "."; max_hits::Int = 100)
    full = abspath(String(root))
    isdir(full) || return "Diretorio nao encontrado: $full"
    regex = Regex(pattern, "i")
    hits = String[]
    for (dir, _, files) in walkdir(full)
        for f in files
            file = joinpath(dir, f)
            ext = lowercase(splitext(file)[2])
            ext in DEFAULT_EXTENSIONS || continue
            content = try
                read(file, String)
            catch
                continue
            end
            lines = split(content, '\n')
            for (i, line) in enumerate(lines)
                occursin(regex, line) || continue
                push!(hits, "$(file):$(i): $(strip(line))")
                length(hits) >= max_hits && return join(hits, "\n")
            end
        end
    end
    return isempty(hits) ? "(nenhum resultado)" : join(hits, "\n")
end
