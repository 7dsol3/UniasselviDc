const DEFAULT_EXTENSIONS = Set([
    ".jl", ".md", ".txt", ".py", ".json", ".toml", ".yaml", ".yml",
    ".csv", ".ini", ".cfg", ".sql", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".sh", ".ps1", ".bat",
])

mutable struct IndexedChunk
    id::UUID
    source::String
    text::String
    embedding::Vector{Float64}
end

mutable struct MemoryStore
    dims::Int
    chunks::Vector{IndexedChunk}
end

MemoryStore(; dims::Int = 384) = MemoryStore(dims, IndexedChunk[])

function _tokenize(text::String)
    clean = lowercase(replace(text, r"[^\p{L}\p{N}]+" => " "))
    toks = split(clean)
    return [t for t in toks if !isempty(t)]
end

function _embed(text::String, dims::Int)
    vec = zeros(Float64, dims)
    for tok in _tokenize(text)
        idx = mod(hash(tok), dims) + 1
        vec[idx] += 1.0
    end
    n = norm(vec)
    n > 0 && (vec ./= n)
    return vec
end

function _similarity(a::Vector{Float64}, b::Vector{Float64})
    return dot(a, b)
end

function _split_chunks(text::String; chunk_size::Int = 900, overlap::Int = 120)
    if isempty(text)
        return String[]
    end
    chunk_size <= overlap && throw(ArgumentError("chunk_size deve ser maior que overlap"))
    out = String[]
    i = firstindex(text)
    n = lastindex(text)
    while i <= n
        j = min(n, nextind(text, i, chunk_size - 1))
        push!(out, strip(text[i:j]))
        j == n && break
        i = max(i + 1, nextind(text, i, chunk_size - overlap))
    end
    return out
end

function add_text!(
    store::MemoryStore,
    source::AbstractString,
    text::AbstractString;
    chunk_size::Int = 900,
    overlap::Int = 120,
)
    added = 0
    for chunk in _split_chunks(String(text); chunk_size = chunk_size, overlap = overlap)
        isempty(chunk) && continue
        emb = _embed(chunk, store.dims)
        push!(store.chunks, IndexedChunk(uuid4(), String(source), chunk, emb))
        added += 1
    end
    return added
end

function add_file!(
    store::MemoryStore,
    file_path::AbstractString;
    chunk_size::Int = 900,
    overlap::Int = 120,
    max_bytes::Int = 2_000_000,
)
    path = abspath(String(file_path))
    isfile(path) || return 0
    filesize(path) > max_bytes && return 0

    text = try
        read(path, String)
    catch
        return 0
    end

    return add_text!(store, path, text; chunk_size = chunk_size, overlap = overlap)
end

function _is_supported_text_file(path::String)
    ext = lowercase(splitext(path)[2])
    return ext in DEFAULT_EXTENSIONS
end

function add_path!(
    store::MemoryStore,
    path::AbstractString;
    chunk_size::Int = 900,
    overlap::Int = 120,
    max_files::Int = 10_000,
)
    full = abspath(String(path))
    if isfile(full)
        return add_file!(store, full; chunk_size = chunk_size, overlap = overlap)
    end
    isdir(full) || return 0

    total = 0
    seen = 0
    for (root, _, files) in walkdir(full)
        for f in files
            file = joinpath(root, f)
            _is_supported_text_file(file) || continue
            total += add_file!(store, file; chunk_size = chunk_size, overlap = overlap)
            seen += 1
            seen >= max_files && return total
        end
    end
    return total
end

function search(store::MemoryStore, query::AbstractString; topk::Int = 5, min_score::Float64 = 0.05)
    q = _embed(String(query), store.dims)
    scored = NamedTuple{(:score, :source, :text, :id), Tuple{Float64, String, String, UUID}}[]
    for c in store.chunks
        s = _similarity(q, c.embedding)
        s < min_score && continue
        push!(scored, (score = s, source = c.source, text = c.text, id = c.id))
    end
    sort!(scored, by = x -> x.score, rev = true)
    return first(scored, min(topk, length(scored)))
end
