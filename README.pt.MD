# Laboratório de Inteligência Artificial

Repositório com implementações práticas de redes neurais clássicas desenvolvidas para a disciplina de Laboratório de Inteligência Artificial no CEFET-MG. O projeto reúne experimentos supervisionados, não supervisionados e associativos, com scripts em Python, dados estruturados em JSON, gráficos de análise e páginas HTML para apresentação dos resultados.

## O Que Este Projeto Demonstra

- Implementação manual de algoritmos clássicos de redes neurais.
- Preparação e estruturação de dados de treinamento, teste e validação.
- Treinamento, classificação, validação e análise de erro.
- Geração de gráficos e tabelas com `numpy` e `matplotlib`.
- Organização de resultados em JSON para inspeção e reutilização.
- Criação de páginas HTML estáticas para comunicar resultados de forma clara.
- Capacidade de explicar conceitos técnicos por meio de código comentado, visualizações e respostas textuais.

## Experimentos

| Diretório | Tema | Conteúdo principal |
|---|---|---|
| `perceptron/` | Perceptron simples | Classificação binária, múltiplos treinamentos, análise de épocas e limitação para problemas não linearmente separáveis. |
| `adaline/` | ADALINE | Regra Delta, minimização de EQM, comparação entre treinamentos e classificação de amostras. |
| `pmc1/` | Perceptron Multicamadas | MLP com backpropagation para estimação de saída contínua, validação e escolha da melhor configuração. |
| `pmc2/` | Perceptron Multicamadas | Extração de dados, treinamento, validação, gráficos e página HTML de resultados. |
| `pmc3/` | MLP com TDNN | Previsão de série temporal financeira usando janelas temporais e comparação de topologias. |
| `rbf1/` | Rede RBF | Agrupamento, pesos da camada de saída, validação e análise de estratégias. |
| `rbf2/` | Rede RBF | Treinamento e validação com dados extraídos do enunciado, resultados em JSON e gráficos. |
| `kohonen/` | Mapa Auto-Organizável | SOM 4x4, neurônio vencedor, agrupamento por classe dominante e classificação de amostras. |
| `hopfield/` | Rede de Hopfield | Memória associativa, recuperação de padrões com ruído, energia da rede e experimento de robustez. |
| `rnalvq1/` | LVQ-1 | Classificação de perfis de potência elétrica usando protótipos por classe. |
| `rnaart1/` | ART-1 | Agrupamento incremental de vetores binários com diferentes graus de vigilância. |

## Estrutura Padrão

As pastas foram organizadas para seguir um padrão simples e rastreável:

- arquivo `.docx` com o enunciado original;
- script Python principal com a implementação do algoritmo;
- arquivos `.json` com dados e resultados;
- `results_data.js` como manifesto local de dados para páginas HTML;
- imagens `.png` com gráficos, tabelas ou explicações visuais;
- `index.html` ou `README.md` com a apresentação das respostas.

As páginas HTML consomem dados locais por `<script src="results_data.js"></script>` quando precisam de conteúdo dinâmico, evitando dependência de servidor local ou `fetch`.

## Tecnologias

- Python
- NumPy
- Matplotlib
- Pillow
- HTML, CSS e JavaScript básico para visualização local
- JSON para persistência e auditoria dos resultados

As dependências estão em [`requirements.txt`](requirements.txt).

## Como Executar

Crie e ative um ambiente virtual, instale as dependências e execute o script da pasta desejada:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Exemplos:

```bash
python3 rnalvq1/lvq1.py
python3 rnaart1/art1.py
python3 kohonen/kohonen.py
python3 hopfield/hopfield.py
```

Alguns scripts foram escritos para serem executados de dentro da própria pasta, principalmente os mais antigos. Se algum arquivo de entrada não for encontrado, execute o comando dentro do diretório correspondente:

```bash
cd kohonen
python3 kohonen.py
```

Depois da execução, abra o `index.html` da pasta quando existir. Nos módulos mais recentes, a página consome os dados gerados pelo próprio script Python.

## Destaques Técnicos

Este projeto mostra familiaridade com fundamentos que continuam relevantes em engenharia de software aplicada a dados e IA:

- leitura crítica de requisitos;
- modelagem de dados a partir de enunciados;
- implementação de algoritmos sem depender apenas de bibliotecas prontas;
- validação objetiva por métricas, gráficos e arquivos de resultado;
- comunicação técnica para públicos acadêmicos e técnicos;
- organização incremental de entregas reutilizáveis.

Para contextos profissionais, o valor principal deste repositório não está apenas nos modelos em si, mas no processo: entender um problema, implementar uma solução rastreável, gerar evidências e apresentar conclusões de forma clara.
