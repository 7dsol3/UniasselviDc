module IANeuralCore

using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
using TOML
using UUIDs

include("Types.jl")
include("MemoryStore.jl")
include("Quantization.jl")
include("RealWeights.jl")
include("Benchmarking.jl")
include("Backends.jl")
include("Tools.jl")
include("Agent.jl")
include("CLI.jl")

export ChatMessage
export AgentConfig
export NeuralAgent
export MemoryStore
export IndexedChunk
export RuleBackend
export LlamaCppBackend
export OllamaBackend
export UniqueQuantBackend
export add_text!
export add_file!
export add_path!
export search
export respond!
export clear_history!
export save_agent!
export load_agent_state!
export run_cli
export ALLOWED_BITS
export LayerSensitivity
export QuantizedLayer
export QuantizedModel
export AllocationResult
export EvalReport
export generate_synthetic_layers
export save_layers
export load_layers
export sensitivity_scan
export save_sensitivity
export load_sensitivity
export bit_allocator
export save_allocation
export load_allocation
export quantize_mixed
export save_quantized_model
export load_quantized_model
export dequantize_layer
export eval_unique
export save_eval_report
export quantization_error_mse
export indice_gato
export igato_regularization
export ImportedTensorInfo
export ImportReport
export import_hf_safetensors
export save_import_report_md
export StrategyMetrics
export QuantComparisonReport
export uniform_allocation
export benchmark_proxy_latency_ms
export compare_mixed_vs_uniform4
export save_comparison_report_md

end # module
