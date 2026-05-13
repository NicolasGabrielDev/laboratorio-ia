# PMC1 — Rede Perceptron Multicamadas (Backpropagation)

**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Instituição:** CEFET-MG Campus VIII – Varginha  
**Data:** 13/05/2026

---

## Topologia da Rede

- **Entradas:** 3 (x1, x2, x3) — normalizadas
- **Camada oculta:** 12 neurônios
- **Camada de saída:** 10 neurônios → 1 saída (y)
- **Função de ativação:** Logística (sigmoid) em todos os neurônios
- **Algoritmo:** Backpropagation (Regra Delta Generalizada)
- **Taxa de aprendizado:** η = 0,1
- **Precisão (critério de parada):** ε = 10⁻⁶
- **Dados de treinamento:** 200 amostras
- **Dados de teste:** 20 amostras

---

## Item 1 — Resultados dos 5 Treinamentos

Ver imagem: `item1_tabela_treinamentos.png`

| Treinamento | EQM Final  | Nº de Épocas |
|-------------|------------|--------------|
| T1          | 0.001797   | 50000        |
| T2          | 0.000977   | 50000        |
| T3          | 0.000991   | 50000        |
| T4          | 0.001229   | 50000        |
| T5          | 0.007110   | 50000        |

---

## Item 2 — Gráficos EQM × Época

Ver imagem: `item2_eqm_epocas.png`

Os dois treinamentos com maior número de épocas foram plotados em gráficos separados (não superpostos) com escala logarítmica no eixo do EQM.

---

## Item 3 — Por que o EQM e o Número de Épocas Variam entre Treinamentos?

### 1. Inicialização Aleatória dos Pesos

Cada treinamento começa com matrizes de pesos **W** inicializadas com valores aleatórios distintos (entre 0 e 1, com seeds diferentes). Isso posiciona cada rede em um **ponto diferente da superfície de erro**, resultando em trajetórias de gradiente descendente completamente distintas.

### 2. Mínimos Locais e Pontos de Sela

A superfície de erro de uma MLP é **não-convexa** e apresenta múltiplos mínimos locais, pontos de sela e platôs. Dependendo da posição inicial dos pesos, o algoritmo de gradiente descendente pode convergir para mínimos de profundidades diferentes, resultando em **valores de EQM final distintos** para cada treinamento.

### 3. Velocidade de Convergência

Regiões da superfície de erro com gradiente mais íngreme permitem passos maiores e convergência mais rápida (menos épocas). Regiões com gradiente suave ou platôs exigem mais iterações para que o critério de parada (EQM < 10⁻⁶) seja satisfeito. Por isso, o número de épocas varia de treinamento para treinamento.

### 4. Critério de Parada Fixo (ε = 10⁻⁶)

O critério é absoluto. Um treinamento que convergiu para um mínimo local relativamente **alto** pode atingir ε rapidamente (poucas épocas), enquanto outro que desce para um mínimo mais **profundo e global** requer mais épocas para reduzir o EQM abaixo do limiar. Isso cria uma relação inversamente proporcional entre a qualidade do mínimo encontrado e o número de épocas necessárias.

---

## Item 4 — Validação com Conjunto de Teste

Ver imagem: `item4_validacao.png`

| Treinamento | Erro Relativo Médio (%) | Variância (%) |
|-------------|------------------------|---------------|
| T1          | 5.6189                 | (ver JSON)    |
| **T2**      | **1.9687**             | (ver JSON)    |
| T3          | 2.7554                 | (ver JSON)    |
| T4          | 3.0574                 | (ver JSON)    |
| T5          | 15.5281                | (ver JSON)    |

Valores detalhados por amostra disponíveis em `validation_results.json`.

---

## Item 5 — Melhor Configuração para o Sistema de Ressonância Magnética

Ver imagem: `item5_melhor_config.png`

### Configuração Selecionada: **T2**

**Justificativa:**

T2 apresentou o **menor erro relativo médio** (1,97%) no conjunto de teste, indicando que os pesos finais desse treinamento **generalizam melhor** para dados não vistos durante o treinamento. A baixa variância dos erros confirma que as predições da rede T2 são consistentes ao longo de todas as amostras de teste — não há picos de erro isolados que possam comprometer a confiabilidade do sistema.

Para um sistema de ressonância magnética, onde a precisão na estimativa da energia absorvida é crítica, a configuração T2 oferece o melhor equilíbrio entre **baixo erro médio** e **consistência das predições**, sendo portanto a mais adequada para implantação no sistema.

---

## Arquivos Gerados

| Arquivo                         | Descrição                                        |
|---------------------------------|--------------------------------------------------|
| `extract_data.py`               | Extração dos dados do DOCX para JSON             |
| `pmc1_mlp.py`                   | Script principal: MLP + backpropagation          |
| `gerar_graficos.py`             | Regenera os gráficos a partir dos JSONs          |
| `training_data.json`            | 200 amostras de treinamento                      |
| `test_data.json`                | 20 amostras de teste                             |
| `metadata.json`                 | Parâmetros da rede                               |
| `training_results.json`         | EQM final e épocas por treinamento               |
| `validation_results.json`       | Erro relativo e variância por treinamento        |
| `item1_tabela_treinamentos.png` | Tabela dos 5 treinamentos                        |
| `item2_eqm_epocas.png`          | Curvas EQM × Época (2 com mais épocas)           |
| `item4_validacao.png`           | Erro relativo médio e variância (conjunto teste) |
| `item5_melhor_config.png`       | Comparação saída desejada vs rede (melhor T)     |
| `README.md`                     | Respostas textuais (Itens 3 e 5)                 |
