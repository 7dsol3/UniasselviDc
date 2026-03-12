using Downloads

out_dir = length(ARGS) >= 1 ? ARGS[1] : ".\\data\\hf_tiny"
out_dir = abspath(out_dir)
mkpath(out_dir)

url = "https://huggingface.co/hf-internal-testing/tiny-random-t5-v1.1/resolve/main/model.safetensors"
dest = joinpath(out_dir, "model.safetensors")

println("Baixando exemplo tiny safetensors:")
println("URL:  $url")
println("DEST: $dest")
Downloads.download(url, dest)
println("Concluido.")
