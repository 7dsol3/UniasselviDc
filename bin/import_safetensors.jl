include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore

function _argval(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function _parse_patterns(s::String)
    isempty(strip(s)) && return String[]
    return [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

src = _argval("--source", "")
if isempty(src)
    println("Uso:")
    println("  julia bin/import_safetensors.jl --source <arquivo_ou_pasta_safetensors> [--out layers_fp.jls]")
    println("Exemplo:")
    println("  julia bin/import_safetensors.jl --source .\\hf_model --out .\\data\\real\\layers_fp.jls")
    exit(1)
end

out_layers = _argval("--out", joinpath(@__DIR__, "..", "data", "real", "layers_fp.jls"))
out_report = _argval("--report", joinpath(@__DIR__, "..", "data", "real", "import_report.md"))
include_pat = _parse_patterns(_argval("--include", ".*"))
exclude_pat = _parse_patterns(_argval("--exclude", "optimizer,momentum,adam"))
max_tensors = parse(Int, _argval("--max-tensors", "128"))
max_tensor_params = parse(Int, _argval("--max-tensor-params", "20000000"))
total_cap = parse(Int, _argval("--total-param-cap", "120000000"))

layers, infos, rep = import_hf_safetensors(
    src;
    include_patterns = include_pat,
    exclude_patterns = exclude_pat,
    max_tensors = max_tensors,
    max_tensor_params = max_tensor_params,
    total_param_cap = total_cap,
)

save_layers(out_layers, layers)
save_import_report_md(out_report, infos, rep)

println("Importacao concluida.")
println("layers_fp: $(abspath(out_layers))")
println("report_md: $(abspath(out_report))")
println("tensors selecionados: $(rep.tensors_selected)")
println("total params: $(rep.total_params)")
