# Benchmark: Mixed Autoral vs Uniforme 4-bit

- criado_em: 2026-03-12T17:18:13.771
- nota: latencia medida por proxy de forward dequant+matmul

| estrategia | mean_mse | max_mse | compressao | latencia_ms | total_bits | objective |
|---|---:|---:|---:|---:|---:|---:|
| mixed_authoral | 0.009142668763451904 | 0.029581887110549994 | 5.0055341271172225 | 0.19215 | 381632 | 0.02696958728295236 |
| uniform_4bit | 0.008781874562414383 | 0.029581887110549994 | 4.0 | 0.298475 | 477568 | 0.0 |

- mse_ratio_mixed_over_uniform: 1.0410839620257943
- latency_speedup_uniform_over_mixed: 1.553343741868332
