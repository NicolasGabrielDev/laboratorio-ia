# Projeto PMC3 - Multilayer Perceptron com TDNN

Este projeto aplica uma Rede Neural Perceptron Multicamadas (MLP) utilizando a arquitetura Time Delay Neural Network (TDNN) para a previsão de valores de uma série temporal financeira `f(t)`.

## Topologia e Configuração Mais Adequada
Baseado nas análises e resultados de validação:
- **Topologia:** Rede 2 (10 entradas `p=10`, 15 neurônios ocultos `N1=15`)
- **Treinamento:** T1

**Justificativa:** A Rede 2 obteve consistentemente os menores Erros Relativos Médios (em torno de 0.46) durante a validação no conjunto de testes (amostras t=101 a t=120) e apresentou a menor variância, mostrando uma melhor capacidade de generalização e aderência aos valores desejados em comparação com as Redes 1 e 3.

## Algoritmos de Treinamento Avançados (Vantagens e Características)

### Resilient-Propagation (RProp)
- **Características:** É um algoritmo de otimização heurística voltado para redes feed-forward. Ao contrário do backpropagation tradicional que utiliza a magnitude do gradiente para atualizar os pesos, o RProp usa apenas o sinal do gradiente. Ele adapta independentemente o passo de atualização (step size) para cada peso com base nas mudanças de sinal do gradiente em épocas consecutivas.
- **Vantagens:** 
  - Supera problemas comuns associados a gradientes extremamente pequenos (vanishing gradient problem) que ocorrem ao usar funções de ativação sigmoides em áreas saturadas.
  - Convergência mais rápida em muitos casos do que o backpropagation padrão.
  - Requer a configuração de poucos hiperparâmetros (os parâmetros de aumento e diminuição do passo costumam ser fixos e robustos na literatura).

### Levenberg-Marquardt (LM)
- **Características:** É um método numérico que combina o Método do Gradiente Descendente e o Método de Gauss-Newton. Quando a solução está longe do mínimo, o LM se comporta como o gradiente descendente; conforme a solução se aproxima do mínimo, ele passa a atuar como Gauss-Newton para garantir uma convergência mais rápida.
- **Vantagens:**
  - Apresenta taxa de convergência quadrática (muito rápida) próximo à solução mínima.
  - É considerado um dos algoritmos de treinamento mais rápidos e eficientes para redes de tamanho pequeno a médio com dados baseados em erro quadrático.
  - É mais robusto e estável do que o backpropagation clássico e métodos baseados unicamente em Gauss-Newton, lidando bem com mínimos locais complexos.

---

*(Os gráficos e tabelas solicitados foram gerados em arquivos de imagem e estão na mesma pasta).*
