using Test
using SafeTensors

include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore

@testset "MemoryStore" begin
    mem = MemoryStore(dims = 128)
    add_text!(mem, "doc1", "julia e uma linguagem rapida para IA e computacao numerica")
    add_text!(mem, "doc2", "python e popular, mas julia tem desempenho alto em ciencia")
    hits = search(mem, "desempenho julia", topk = 2, min_score = 0.01)
    @test !isempty(hits)
    @test occursin("julia", lowercase(hits[1].text))
end

@testset "Agent + RuleBackend" begin
    agent = NeuralAgent(RuleBackend(); memory = MemoryStore(), config = AgentConfig())
    add_text!(agent.memory, "manual", "use /add para indexar arquivos e /search para recuperar contexto")
    reply, hits = respond!(agent, "como eu indexo arquivos?")
    @test !isempty(reply)
    @test !isempty(agent.history)
    @test length(hits) >= 0
end

@testset "Unique Quantization Pipeline" begin
    layers = generate_synthetic_layers(n_layers = 6, rows = 64, cols = 64, seed = 7)
    sens = sensitivity_scan(layers; k = 0.1)
    @test length(sens) == 6
    @test all(s.sensitivity >= 0 for s in sens)
    @test all(haskey(s.errors, 2) && haskey(s.errors, 4) && haskey(s.errors, 8) for s in sens)

    total_params = sum(s.n_params for s in sens)
    budget = Int(round(total_params * 3.1))
    alloc = bit_allocator(sens; budget_bits = budget, bit_choices = [2, 3, 4, 8])
    @test alloc.feasible
    @test alloc.total_bits <= alloc.budget_bits
    @test length(alloc.bits) == length(sens)

    qmodel = quantize_mixed(layers, alloc; model_name = "test_mixed")
    @test length(qmodel.layers) == 6
    @test all(l.bits in (2, 3, 4, 8) for l in qmodel.layers)

    report = eval_unique(layers, qmodel; t0 = 0.0, t1 = 10.0)
    @test report.total_params > 0
    @test report.mean_mse >= 0
    @test report.compression_ratio > 1.0
    @test report.igato_start >= report.igato_end
end

@testset "Real Import + Benchmark Compare" begin
    tmp = mktempdir()
    st_file = joinpath(tmp, "toy_model.safetensors")
    toy = Dict(
        "model.layers.0.self_attn.q_proj.weight" => randn(Float32, 32, 32),
        "model.layers.0.self_attn.k_proj.weight" => randn(Float32, 32, 32),
        "model.layers.0.mlp.up_proj.weight" => randn(Float32, 64, 32),
        "model.layers.0.mlp.down_proj.weight" => randn(Float32, 32, 64),
    )
    SafeTensors.serialize(st_file, toy)

    layers, infos, rep = import_hf_safetensors(
        tmp;
        include_patterns = ["q_proj", "k_proj", "up_proj", "down_proj"],
        max_tensors = 16,
        max_tensor_params = 10_000,
        total_param_cap = 100_000,
    )
    @test rep.tensors_selected == 4
    @test length(layers) == 4
    @test length(infos) == 4

    sens = sensitivity_scan(layers; k = 0.1)
    budget = Int(round(sum(s.n_params for s in sens) * 3.2))
    alloc = bit_allocator(sens; budget_bits = budget, bit_choices = [2, 3, 4, 8])
    @test alloc.feasible

    cmp = compare_mixed_vs_uniform4(layers, alloc; latency_iters = 2)
    @test cmp.mixed.total_bits > 0
    @test cmp.uniform4.total_bits > 0
    @test cmp.mixed.latency_ms >= 0
    @test cmp.uniform4.latency_ms >= 0
end

println("Todos os testes passaram.")
