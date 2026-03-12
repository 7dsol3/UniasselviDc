# Mixed-Precision Sensitivity-Guided Quantization with Dynamic Regularization

## Objetivo

Definir uma rota de compressao autoral em que o modelo decide, camada por camada, onde pode perder precisao sem colapso funcional.

## Formula central de sensibilidade

\[
S_l = \mathbb{E}\left[\frac{|\nabla_{w_l}L|}{1 + k|w_l|}\right]
\]

Onde:
- \(w_l\): pesos da camada \(l\)
- \(\nabla_{w_l}L\): gradiente da perda
- \(k\): fator de contraste para normalizar sensibilidade por magnitude

## Alocacao otima de bits

\[
\min_{\{b_l\}} \sum_l S_l \cdot E_l(b_l)
\]

sujeito a:

\[
\sum_l N_l b_l \le B,\qquad b_l \in \{2,3,4,8\}
\]

Onde:
- \(E_l(b_l)\): erro estimado de quantizacao da camada \(l\) no bitwidth \(b_l\)
- \(N_l\): numero de parametros da camada
- \(B\): orcamento total de bits

## Regularizacao dinamica (I_gato)

Indice:

\[
I_{\text{gato}}(t)=\frac{2|\alpha\beta|e^{-(\Gamma_0+\lambda|R|)t}}{1+k|E|}
\]

Uso no treino/QAT (regime adaptativo):
- \(I_{\text{gato}}\) alto: menor brutalidade de regularizacao/ruido
- \(I_{\text{gato}}\) baixo: maior intensidade de compressao

Exemplo de modulacao:

\[
\sigma_{\text{noise}}(t)=\sigma_0\left[1+\eta\left(1-I_{\text{gato}}(t)\right)\right]
\]

## Implementacao no projeto

- `src/Quantization.jl`
  - `sensitivity_scan`
  - `bit_allocator`
  - `quantize_mixed`
  - `eval_unique`
  - `indice_gato` e `igato_regularization`
- `bin/`
  - `sensitivity_scan.jl`
  - `bit_allocator.jl`
  - `quantize_mixed.jl`
  - `eval_unique.jl`
  - `pipeline_unique.jl`

## Resultado esperado

Comparado a quantizacao uniforme, a rota acima tende a:
- preservar melhor camadas criticas,
- reduzir erro agregado para o mesmo orcamento de memoria,
- entregar melhor trade-off qualidade x tamanho.
