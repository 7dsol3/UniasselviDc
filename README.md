
Este projeto foi criado apenas como um experimento inicial.

IANeuralCore


# Mixed-Precision Sensitivity-Guided Quantization with Dynamic Regularization

## Objetivo

Definir uma rota de compressão autoral em que o modelo decide, camada por camada, onde pode perder precisão sem colapso funcional.

---

# Fórmula central de sensibilidade

$$
S_l = \mathbb{E}\left[\frac{|\nabla_{w_l}L|}{1 + k|w_l|}\right]
$$

Onde:

- $w_l$ — pesos da camada $l$
- $\nabla_{w_l}L$ — gradiente da função de perda
- $k$ — fator de contraste para normalizar sensibilidade pela magnitude do peso

---

# Alocação ótima de bits

$$
\min_{\{b_l\}} \sum_l S_l \cdot E_l(b_l)
$$

Sujeito a:

$$
\sum_l N_l b_l \le B
$$

$$
b_l \in \{2,3,4,8\}
$$

Onde:

- $E_l(b_l)$ — erro estimado de quantização da camada $l$ com bitwidth $b_l$
- $N_l$ — número de parâmetros da camada
- $B$ — orçamento total de bits

---

# Regularização dinâmica (Indice Gato)

Índice:

$$
I_{\text{gato}}(t) =
\frac{2|\alpha\beta|e^{-(\Gamma_0+\lambda|R|)t}}
{1+k|E|}
$$

---

# Uso no treino / QAT (regime adaptativo)

- $I_{\text{gato}}$ alto → menor brutalidade de regularização / ruído  
- $I_{\text{gato}}$ baixo → maior intensidade de compressão

---

# Modulação do ruído

$$
\sigma_{\text{noise}}(t) =
\sigma_0
\left[
1 + \eta\left(1 - I_{\text{gato}}(t)\right)
\right]
$$

---

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

7dsolv

![jkkkkkkkkk-imagens-0](https://github.com/user-attachments/assets/55ddb2e4-fc25-46b5-9480-000c555aaf39)

