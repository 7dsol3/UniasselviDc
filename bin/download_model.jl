using Downloads

if isempty(ARGS)
    println("Uso:")
    println("  julia bin/download_model.jl <url_do_modelo.gguf> [destino.gguf]")
    exit(1)
end

url = ARGS[1]
dest = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "models", basename(url))
dest = abspath(dest)

mkpath(dirname(dest))
println("Baixando:")
println("  URL:  $url")
println("  DEST: $dest")

Downloads.download(url, dest)
println("Concluido.")
