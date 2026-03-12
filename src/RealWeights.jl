using SafeTensors

struct ImportedTensorInfo
    name::String
    file::String
    shape::Tuple
    n_params::Int
end

struct ImportReport
    source::String
    files_scanned::Int
    tensors_seen::Int
    tensors_selected::Int
    total_params::Int
    skipped_name::Int
    skipped_shape::Int
    skipped_size::Int
end

function _collect_safetensor_files(path::String)
    if isfile(path)
        return endswith(lowercase(path), ".safetensors") ? [abspath(path)] : String[]
    end
    isdir(path) || return String[]
    files = String[]
    for (root, _, fs) in walkdir(path)
        for f in fs
            endswith(lowercase(f), ".safetensors") || continue
            push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    return files
end

function _compile_patterns(patterns::Vector{String})
    return [Regex(p, "i") for p in patterns]
end

function _matches_name(name::String, include_re::Vector{Regex}, exclude_re::Vector{Regex})
    include_ok = isempty(include_re) || any(r -> occursin(r, name), include_re)
    include_ok || return false
    return !any(r -> occursin(r, name), exclude_re)
end

function import_hf_safetensors(
    source_path::AbstractString;
    include_patterns::Vector{String} = String[],
    exclude_patterns::Vector{String} = String[],
    max_tensors::Int = 128,
    min_ndims::Int = 2,
    max_ndims::Int = 4,
    min_tensor_params::Int = 256,
    max_tensor_params::Int = 20_000_000,
    total_param_cap::Int = 120_000_000,
    mmap::Bool = true,
)
    src = abspath(String(source_path))
    files = _collect_safetensor_files(src)
    isempty(files) && throw(ArgumentError("nenhum arquivo .safetensors encontrado em: $src"))

    include_re = _compile_patterns(include_patterns)
    exclude_re = _compile_patterns(exclude_patterns)

    layers = Dict{String, Array{Float32}}()
    infos = ImportedTensorInfo[]
    tensors_seen = 0
    selected = 0
    total_params = 0
    skipped_name = 0
    skipped_shape = 0
    skipped_size = 0

    for file in files
        st = SafeTensors.deserialize(file; mmap = mmap)
        for (name_any, tensor_view) in st
            name = String(name_any)
            tensors_seen += 1

            if !_matches_name(name, include_re, exclude_re)
                skipped_name += 1
                continue
            end

            shp = size(tensor_view)
            nd = length(shp)
            if nd < min_ndims || nd > max_ndims
                skipped_shape += 1
                continue
            end

            n = length(tensor_view)
            if n < min_tensor_params || n > max_tensor_params
                skipped_size += 1
                continue
            end

            if haskey(layers, name)
                continue
            end

            arr = Float32.(collect(tensor_view))
            layers[name] = arr
            push!(infos, ImportedTensorInfo(name, file, shp, n))
            selected += 1
            total_params += n

            if selected >= max_tensors || total_params >= total_param_cap
                break
            end
        end
        if selected >= max_tensors || total_params >= total_param_cap
            break
        end
    end

    report = ImportReport(
        src,
        length(files),
        tensors_seen,
        selected,
        total_params,
        skipped_name,
        skipped_shape,
        skipped_size,
    )
    return layers, infos, report
end

function save_import_report_md(path::AbstractString, infos::Vector{ImportedTensorInfo}, r::ImportReport)
    full = abspath(String(path))
    mkpath(dirname(full))
    open(full, "w") do io
        println(io, "# Relatorio de Importacao de Pesos Reais")
        println(io)
        println(io, "- source: `$(r.source)`")
        println(io, "- files_scanned: $(r.files_scanned)")
        println(io, "- tensors_seen: $(r.tensors_seen)")
        println(io, "- tensors_selected: $(r.tensors_selected)")
        println(io, "- total_params: $(r.total_params)")
        println(io, "- skipped_name: $(r.skipped_name)")
        println(io, "- skipped_shape: $(r.skipped_shape)")
        println(io, "- skipped_size: $(r.skipped_size)")
        println(io)
        println(io, "## Tensors Selecionados")
        println(io)
        println(io, "| name | shape | n_params | file |")
        println(io, "|---|---:|---:|---|")
        for item in infos
            println(io, "| `$(item.name)` | `$(item.shape)` | $(item.n_params) | `$(basename(item.file))` |")
        end
    end
    return full
end
