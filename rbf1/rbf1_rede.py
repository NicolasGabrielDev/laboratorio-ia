"""
RBF1 - Rede de Base Radial (RBF) para Classificação de Radiação em Compostos Nucleares
Topologia: 2 entradas (x1, x2) -> 2 neurônios RBF (Gaussiana) -> 1 saída (linear)
Algoritmo camada oculta: K-means (K=2, apenas padrões com d=1)
Algoritmo camada saída: Regra Delta Generalizada
Taxa de aprendizado: η = 0.01
Precisão: ε = 10⁻⁷
Pós-processamento: Função Sinal (sign)
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CARREGAR DADOS DOS ARQUIVOS JSON
# ============================================================
with open('training_data.json') as arquivo_treinamento:
    registros_treinamento = json.load(arquivo_treinamento)

with open('test_data.json') as arquivo_teste:
    registros_teste = json.load(arquivo_teste)

# Converte para arrays numpy para operações vetoriais
entradas_treinamento = np.array([[r['x1'], r['x2']] for r in registros_treinamento])
saidas_desejadas_treinamento = np.array([r['d'] for r in registros_treinamento])

entradas_teste = np.array([[r['x1'], r['x2']] for r in registros_teste])
saidas_desejadas_teste = np.array([r['d'] for r in registros_teste])

num_amostras_treinamento = len(entradas_treinamento)
num_amostras_teste = len(entradas_teste)

# Filtra apenas os padrões com presença de radiação (d=1) para o k-means
mascara_radiacao = saidas_desejadas_treinamento == 1
entradas_com_radiacao = entradas_treinamento[mascara_radiacao]

print(f"Total de amostras de treinamento: {num_amostras_treinamento}")
print(f"Amostras com presença de radiação (d=1): {len(entradas_com_radiacao)}")
print(f"Amostras de teste: {num_amostras_teste}")

# ============================================================
# 2. ITEM 1: TREINAMENTO DA CAMADA OCULTA COM K-MEANS
# ============================================================
def kmeans(dados, num_clusters, max_iteracoes=1000, semente=42):
    """
    Algoritmo K-means para agrupamento de dados.
    Parâmetros:
        dados: array numpy (N, D) com os padrões de entrada
        num_clusters: número de clusters desejados (K)
        max_iteracoes: limite máximo de iterações
        semente: semente para reprodutibilidade
    Retorna:
        centros: coordenadas dos centros de cada cluster
        rotulos: atribuição de cada ponto a um cluster
    """
    rng = np.random.default_rng(semente)

    # Inicializa os centros selecionando K pontos aleatórios dos dados
    indices_iniciais = rng.choice(len(dados), num_clusters, replace=False)
    centros = dados[indices_iniciais].copy()

    for iteracao in range(max_iteracoes):
        # Calcula a distância euclidiana de cada ponto a cada centro
        distancias = np.array([
            np.linalg.norm(dados - centro, axis=1)
            for centro in centros
        ])

        # Atribui cada ponto ao cluster mais próximo
        rotulos = np.argmin(distancias, axis=0)

        # Recalcula os centros como a média dos pontos de cada cluster
        novos_centros = np.array([
            dados[rotulos == k].mean(axis=0) for k in range(num_clusters)
        ])

        # Verifica convergência (centros não mudaram)
        if np.allclose(centros, novos_centros):
            print(f"  K-means convergiu em {iteracao + 1} iterações.")
            break

        centros = novos_centros

    return centros, rotulos


print("\n--- Item 1: Treinamento K-means ---")
centros_clusters, rotulos_clusters = kmeans(entradas_com_radiacao, num_clusters=2)

# Calcula a variância de cada cluster: σ² = (1/N_j) * Σ ||x_i - c_j||²
variancias_clusters = []
for indice_cluster in range(2):
    pontos_cluster = entradas_com_radiacao[rotulos_clusters == indice_cluster]
    # Variância = média das distâncias euclidianas ao quadrado ao centro
    variancia = np.mean(np.sum(
        (pontos_cluster - centros_clusters[indice_cluster]) ** 2, axis=1
    ))
    variancias_clusters.append(variancia)

variancias_clusters = np.array(variancias_clusters)

# Exibe os resultados do k-means
for indice_cluster in range(2):
    print(f"  Cluster {indice_cluster + 1}:")
    print(f"    Centro: ({centros_clusters[indice_cluster][0]:.4f}, "
          f"{centros_clusters[indice_cluster][1]:.4f})")
    print(f"    Variância: {variancias_clusters[indice_cluster]:.6f}")

# ============================================================
# 3. FUNÇÕES DA REDE RBF
# ============================================================
def ativacao_gaussiana(entradas, centros, variancias):
    """
    Calcula a ativação Gaussiana (RBF) para cada neurônio da camada oculta.
    φ_j(x) = exp(-||x - c_j||² / (2 * σ_j²))
    Parâmetros:
        entradas: array (N, 2) com os padrões de entrada
        centros: array (K, 2) com os centros dos clusters
        variancias: array (K,) com a variância de cada cluster
    Retorna:
        phi: array (N, K) com as ativações gaussianas
    """
    num_padroes = len(entradas)
    num_neuronios = len(centros)
    phi = np.zeros((num_padroes, num_neuronios))

    for j in range(num_neuronios):
        diferenca = entradas - centros[j]
        distancia_quadrada = np.sum(diferenca ** 2, axis=1)
        phi[:, j] = np.exp(-distancia_quadrada / (2 * variancias[j]))

    return phi


# ============================================================
# 4. ITEM 2: TREINAMENTO DA CAMADA DE SAÍDA (REGRA DELTA GENERALIZADA)
# ============================================================
print("\n--- Item 2: Treinamento da camada de saída ---")

# Calcula as ativações RBF para todos os padrões de treinamento
phi_treinamento = ativacao_gaussiana(
    entradas_treinamento, centros_clusters, variancias_clusters
)

# Adiciona coluna de bias (1s) no início: [1, φ1, φ2]
phi_treinamento_com_bias = np.column_stack([
    np.ones(num_amostras_treinamento), phi_treinamento
])

# Parâmetros de treinamento conforme enunciado
taxa_aprendizado = 0.01      # η = 0.01
precisao = 1e-7              # ε = 10⁻⁷
max_epocas = 100000

# Inicializa os pesos da camada de saída em zero: [w0(bias), w1, w2]
pesos_saida = np.zeros(3)
historico_eqm = []

print(f"  Taxa de aprendizado: {taxa_aprendizado}")
print(f"  Precisão (critério de parada): {precisao}")
print(f"  Treinando...", end=' ', flush=True)

for epoca in range(max_epocas):
    # Forward: calcula a saída da rede (ativação linear)
    # y = w0 + w1*φ1 + w2*φ2
    saida_rede_treinamento = phi_treinamento_com_bias @ pesos_saida

    # Calcula o erro para cada padrão
    erros_treinamento = saidas_desejadas_treinamento - saida_rede_treinamento

    # Calcula o Erro Quadrático Médio (EQM)
    eqm_atual = float(np.mean(0.5 * erros_treinamento ** 2))
    historico_eqm.append(eqm_atual)

    # Critério de parada: convergência do EQM (variação menor que ε)
    if epoca > 0 and abs(historico_eqm[-1] - historico_eqm[-2]) < precisao:
        break

    # Atualiza pesos usando a Regra Delta Generalizada (modo batch)
    # Δw_j = η * (1/N) * Σ (d_i - y_i) * φ_j(x_i)
    gradiente_pesos = taxa_aprendizado * (
        phi_treinamento_com_bias.T @ erros_treinamento
    ) / num_amostras_treinamento
    pesos_saida += gradiente_pesos

num_epocas_convergencia = len(historico_eqm)
eqm_final = historico_eqm[-1]

print(f"Concluído!")
print(f"  Épocas: {num_epocas_convergencia}")
print(f"  EQM final: {eqm_final:.10f}")
print(f"  Pesos da camada de saída:")
print(f"    W21,0 (bias) = {pesos_saida[0]:.6f}")
print(f"    W21,1        = {pesos_saida[1]:.6f}")
print(f"    W21,2        = {pesos_saida[2]:.6f}")

# ============================================================
# 5. ITEM 3: PÓS-PROCESSAMENTO COM FUNÇÃO SINAL
# ============================================================
def funcao_sinal(valores):
    """
    Aplica a função sinal para pós-processamento.
    y_pos = 1 se y >= 0
    y_pos = -1 se y < 0
    Utilizada apenas no pós-processamento do conjunto de teste.
    """
    return np.where(valores >= 0, 1, -1)


# ============================================================
# 6. ITEM 4: VALIDAÇÃO COM CONJUNTO DE TESTE
# ============================================================
print("\n--- Item 4: Validação com conjunto de teste ---")

# Calcula as ativações RBF para os padrões de teste
phi_teste = ativacao_gaussiana(
    entradas_teste, centros_clusters, variancias_clusters
)

# Adiciona coluna de bias
phi_teste_com_bias = np.column_stack([
    np.ones(num_amostras_teste), phi_teste
])

# Calcula a saída da rede para o conjunto de teste
saida_rede_teste = phi_teste_com_bias @ pesos_saida

# Aplica o pós-processamento com a função sinal
saida_pos_processada = funcao_sinal(saida_rede_teste)

# Calcula a taxa de acerto
acertos = np.sum(saida_pos_processada == saidas_desejadas_teste)
taxa_acerto = (acertos / num_amostras_teste) * 100.0

print(f"  Acertos: {acertos}/{num_amostras_teste}")
print(f"  Taxa de acerto: {taxa_acerto:.1f}%")

# Exibe resultados detalhados por amostra
print(f"\n  {'Amostra':<10} {'x1':<8} {'x2':<8} {'d':<5} {'y':<12} {'ypos':<5} {'Acerto'}")
print(f"  {'-'*58}")
for i in range(num_amostras_teste):
    acertou = "OK" if saida_pos_processada[i] == saidas_desejadas_teste[i] else "ERRO"
    print(f"  {registros_teste[i]['amostra']:<10} "
          f"{entradas_teste[i][0]:<8.4f} {entradas_teste[i][1]:<8.4f} "
          f"{saidas_desejadas_teste[i]:<5} {saida_rede_teste[i]:<12.6f} "
          f"{saida_pos_processada[i]:<5} {acertou}")

# ============================================================
# 7. SALVAR RESULTADOS EM JSON
# ============================================================
resultados_treinamento = {
    "kmeans": {
        "clusters": [
            {
                "cluster": k + 1,
                "centro": {
                    "x1": float(centros_clusters[k][0]),
                    "x2": float(centros_clusters[k][1])
                },
                "variancia": float(variancias_clusters[k])
            } for k in range(2)
        ]
    },
    "camada_saida": {
        "pesos": {
            "W21_0_bias": float(pesos_saida[0]),
            "W21_1": float(pesos_saida[1]),
            "W21_2": float(pesos_saida[2])
        },
        "num_epocas": num_epocas_convergencia,
        "eqm_final": float(eqm_final),
        "eqm_historico": [float(e) for e in historico_eqm]
    }
}

with open('training_results.json', 'w') as arquivo_resultados:
    json.dump(resultados_treinamento, arquivo_resultados, indent=2, ensure_ascii=False)
print("\nResultados de treinamento salvos -> training_results.json")

resultados_validacao = {
    "taxa_acerto_pct": float(taxa_acerto),
    "acertos": int(acertos),
    "total": int(num_amostras_teste),
    "resultados_por_amostra": [
        {
            "amostra": int(registros_teste[i]['amostra']),
            "x1": float(entradas_teste[i][0]),
            "x2": float(entradas_teste[i][1]),
            "d": int(saidas_desejadas_teste[i]),
            "y": float(saida_rede_teste[i]),
            "y_pos": int(saida_pos_processada[i]),
            "acerto": bool(saida_pos_processada[i] == saidas_desejadas_teste[i])
        } for i in range(num_amostras_teste)
    ]
}

with open('validation_results.json', 'w') as arquivo_validacao:
    json.dump(resultados_validacao, arquivo_validacao, indent=2, ensure_ascii=False)
print("Resultados de validação salvos -> validation_results.json")

# ============================================================
# 8. ESTILOS GLOBAIS DOS GRÁFICOS (TEMA ESCURO)
# ============================================================
PALETA_CORES = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#CE93D8']
FUNDO_ESCURO = '#0D1117'
FUNDO_CARD = '#161B22'
COR_GRADE = '#30363D'
COR_TEXTO = '#E6EDF3'
COR_DESTAQUE = '#58A6FF'


def aplicar_tema_escuro(eixo):
    """Aplica o tema escuro a um eixo matplotlib."""
    eixo.set_facecolor(FUNDO_CARD)
    eixo.tick_params(colors=COR_TEXTO, labelsize=9)
    eixo.xaxis.label.set_color(COR_TEXTO)
    eixo.yaxis.label.set_color(COR_TEXTO)
    eixo.title.set_color(COR_TEXTO)
    for borda in eixo.spines.values():
        borda.set_color(COR_GRADE)
    eixo.grid(True, color=COR_GRADE, linewidth=0.5, alpha=0.7)


# ============================================================
# FIGURA 1 — Item 1: Resultados do K-means (Clusters + Tabela)
# ============================================================
fig_item1, eixos_item1 = plt.subplots(1, 2, figsize=(14, 6))
fig_item1.patch.set_facecolor(FUNDO_ESCURO)
fig_item1.suptitle("Item 1 — Treinamento da Camada Oculta (K-means)",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# --- Subplot esquerdo: Scatter plot dos clusters ---
eixo_scatter = eixos_item1[0]
aplicar_tema_escuro(eixo_scatter)

# Plota TODOS os padrões de treinamento com cores diferentes para d=1 e d=-1
mascara_ausencia = saidas_desejadas_treinamento == -1
eixo_scatter.scatter(
    entradas_treinamento[mascara_ausencia, 0],
    entradas_treinamento[mascara_ausencia, 1],
    c='#E57373', marker='x', s=60, alpha=0.7,
    label='Ausência (d=-1)', zorder=2
)

# Plota padrões com radiação coloridos por cluster
cores_cluster = ['#4FC3F7', '#81C784']
for indice_cluster in range(2):
    pontos = entradas_com_radiacao[rotulos_clusters == indice_cluster]
    eixo_scatter.scatter(
        pontos[:, 0], pontos[:, 1],
        c=cores_cluster[indice_cluster], marker='o', s=70, alpha=0.8,
        label=f'Presença - Cluster {indice_cluster + 1}', zorder=3
    )

# Plota os centros dos clusters
for indice_cluster in range(2):
    eixo_scatter.scatter(
        centros_clusters[indice_cluster][0],
        centros_clusters[indice_cluster][1],
        c='white', marker='*', s=300, edgecolors=cores_cluster[indice_cluster],
        linewidths=2, zorder=5, label=f'Centro {indice_cluster + 1}'
    )

eixo_scatter.set_xlabel("x1", fontsize=11)
eixo_scatter.set_ylabel("x2", fontsize=11)
eixo_scatter.set_title("Distribuição dos Padrões e Clusters K-means",
                       color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_scatter.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                    labelcolor=COR_TEXTO, fontsize=8, loc='upper right')

# --- Subplot direito: Tabela com centros e variâncias ---
eixo_tabela_clusters = eixos_item1[1]
eixo_tabela_clusters.set_facecolor(FUNDO_ESCURO)
eixo_tabela_clusters.axis('off')
eixo_tabela_clusters.set_title("Centros e Variâncias dos Clusters",
                               color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

cabecalho_clusters = ['Cluster', 'Centro (x1, x2)', 'Variância (σ²)']
dados_tabela_clusters = [
    [f"Cluster {k + 1}",
     f"({centros_clusters[k][0]:.4f}, {centros_clusters[k][1]:.4f})",
     f"{variancias_clusters[k]:.6f}"]
    for k in range(2)
]

tabela_clusters = eixo_tabela_clusters.table(
    cellText=dados_tabela_clusters, colLabels=cabecalho_clusters,
    loc='center', cellLoc='center'
)
tabela_clusters.auto_set_font_size(False)
tabela_clusters.set_fontsize(11)
tabela_clusters.scale(1.3, 3.0)

for (linha, coluna), celula in tabela_clusters.get_celld().items():
    celula.set_edgecolor(COR_GRADE)
    if linha == 0:
        celula.set_facecolor(COR_DESTAQUE)
        celula.set_text_props(color='white', fontweight='bold')
    elif linha == 1:
        celula.set_facecolor('#1A2A3A')
        celula.set_text_props(color='#4FC3F7', fontweight='bold')
    else:
        celula.set_facecolor('#1A3A2A')
        celula.set_text_props(color='#81C784', fontweight='bold')

plt.tight_layout()
plt.savefig('item1_clusters_kmeans.png', dpi=150, bbox_inches='tight',
            facecolor=FUNDO_ESCURO)
plt.close()
print("Salvo: item1_clusters_kmeans.png")

# ============================================================
# FIGURA 2 — Item 2: Pesos da Camada de Saída + Convergência EQM
# ============================================================
fig_item2 = plt.figure(figsize=(14, 6))
fig_item2.patch.set_facecolor(FUNDO_ESCURO)
fig_item2.suptitle("Item 2 — Treinamento da Camada de Saída (Regra Delta Generalizada)",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

gs_item2 = gridspec.GridSpec(1, 2, wspace=0.35)

# --- Subplot esquerdo: Tabela de pesos ---
eixo_pesos = fig_item2.add_subplot(gs_item2[0, 0])
eixo_pesos.set_facecolor(FUNDO_ESCURO)
eixo_pesos.axis('off')
eixo_pesos.set_title("Pesos do Neurônio de Saída",
                     color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

cabecalho_pesos = ['Peso', 'Valor']
dados_tabela_pesos = [
    ['W21,0 (bias)', f'{pesos_saida[0]:.6f}'],
    ['W21,1', f'{pesos_saida[1]:.6f}'],
    ['W21,2', f'{pesos_saida[2]:.6f}']
]

tabela_pesos = eixo_pesos.table(
    cellText=dados_tabela_pesos, colLabels=cabecalho_pesos,
    loc='center', cellLoc='center'
)
tabela_pesos.auto_set_font_size(False)
tabela_pesos.set_fontsize(12)
tabela_pesos.scale(1.4, 3.0)

for (linha, coluna), celula in tabela_pesos.get_celld().items():
    celula.set_edgecolor(COR_GRADE)
    if linha == 0:
        celula.set_facecolor(COR_DESTAQUE)
        celula.set_text_props(color='white', fontweight='bold')
    elif linha % 2 == 0:
        celula.set_facecolor('#1C2128')
        celula.set_text_props(color=COR_TEXTO)
    else:
        celula.set_facecolor(FUNDO_CARD)
        celula.set_text_props(color=COR_TEXTO)

# Adiciona informações de convergência abaixo da tabela
eixo_pesos.text(0.5, 0.08,
                f"Épocas: {num_epocas_convergencia}  |  EQM Final: {eqm_final:.2e}",
                transform=eixo_pesos.transAxes, ha='center', color='#81C784',
                fontsize=10, fontweight='bold')

# --- Subplot direito: Curva de convergência EQM ---
eixo_eqm = fig_item2.add_subplot(gs_item2[0, 1])
aplicar_tema_escuro(eixo_eqm)

epocas_eixo = np.arange(1, len(historico_eqm) + 1)
eixo_eqm.semilogy(epocas_eixo, historico_eqm, color='#4FC3F7',
                   linewidth=1.5, alpha=0.9)
eixo_eqm.fill_between(epocas_eixo, historico_eqm, alpha=0.15, color='#4FC3F7')

# Marca o ponto final
eixo_eqm.scatter([epocas_eixo[-1]], [historico_eqm[-1]], color='white',
                 s=60, zorder=5, edgecolors='#4FC3F7', linewidths=2)

# Linha do critério de parada
eixo_eqm.axhline(y=precisao, color='#E57373', linewidth=1.2,
                 linestyle='--', alpha=0.8, label=f'ε = {precisao}')

eixo_eqm.set_title(f"Convergência do EQM  |  {num_epocas_convergencia} Épocas",
                    color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_eqm.set_xlabel("Época", fontsize=10)
eixo_eqm.set_ylabel("EQM (escala log)", fontsize=10)
eixo_eqm.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                labelcolor=COR_TEXTO, fontsize=9)

plt.savefig('item2_pesos_convergencia.png', dpi=150, bbox_inches='tight',
            facecolor=FUNDO_ESCURO)
plt.close()
print("Salvo: item2_pesos_convergencia.png")

# ============================================================
# FIGURA 3 — Item 3: Pós-processamento (Função Sinal)
# ============================================================
fig_item3, eixos_item3 = plt.subplots(1, 2, figsize=(14, 6))
fig_item3.patch.set_facecolor(FUNDO_ESCURO)
fig_item3.suptitle("Item 3 — Pós-processamento: Função Sinal",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# --- Subplot esquerdo: Gráfico da função sinal ---
eixo_sinal = eixos_item3[0]
aplicar_tema_escuro(eixo_sinal)

# Plota a função sinal
valores_x_sinal = np.linspace(-2, 2, 1000)
valores_y_sinal = np.where(valores_x_sinal >= 0, 1, -1)

eixo_sinal.plot(valores_x_sinal, valores_y_sinal, color='#4FC3F7',
                linewidth=2.5, label='yₚₒₛ = sign(y)')
eixo_sinal.axhline(y=0, color=COR_GRADE, linewidth=0.8)
eixo_sinal.axvline(x=0, color='#FFB74D', linewidth=1.5, linestyle='--',
                   alpha=0.8, label='Limiar (y=0)')

# Marca os pontos do conjunto de teste
for i in range(num_amostras_teste):
    cor_ponto = '#81C784' if saida_pos_processada[i] == saidas_desejadas_teste[i] else '#E57373'
    eixo_sinal.scatter(saida_rede_teste[i], saida_pos_processada[i],
                       color=cor_ponto, s=80, zorder=5, edgecolors='white',
                       linewidths=1.0)

eixo_sinal.set_xlabel("Saída da rede (y)", fontsize=10)
eixo_sinal.set_ylabel("Saída pós-processada (yₚₒₛ)", fontsize=10)
eixo_sinal.set_title("Função Sinal Aplicada às Saídas de Teste",
                     color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_sinal.set_ylim(-1.5, 1.5)
eixo_sinal.set_yticks([-1, 0, 1])
eixo_sinal.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                  labelcolor=COR_TEXTO, fontsize=9)

# --- Subplot direito: Tabela antes/depois do pós-processamento ---
eixo_tabela_sinal = eixos_item3[1]
eixo_tabela_sinal.set_facecolor(FUNDO_ESCURO)
eixo_tabela_sinal.axis('off')
eixo_tabela_sinal.set_title("Saídas Antes e Após Pós-processamento",
                            color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

cabecalho_sinal = ['Amostra', 'y (rede)', 'yₚₒₛ (sign)', 'd (desejado)']
dados_tabela_sinal = [
    [str(registros_teste[i]['amostra']),
     f'{saida_rede_teste[i]:.4f}',
     f'{saida_pos_processada[i]:+d}',
     f'{saidas_desejadas_teste[i]:+d}']
    for i in range(num_amostras_teste)
]

tabela_sinal = eixo_tabela_sinal.table(
    cellText=dados_tabela_sinal, colLabels=cabecalho_sinal,
    loc='center', cellLoc='center'
)
tabela_sinal.auto_set_font_size(False)
tabela_sinal.set_fontsize(9)
tabela_sinal.scale(1.3, 2.0)

for (linha, coluna), celula in tabela_sinal.get_celld().items():
    celula.set_edgecolor(COR_GRADE)
    if linha == 0:
        celula.set_facecolor(COR_DESTAQUE)
        celula.set_text_props(color='white', fontweight='bold')
    elif linha >= 1:
        indice_amostra = linha - 1
        if indice_amostra < num_amostras_teste:
            if saida_pos_processada[indice_amostra] == saidas_desejadas_teste[indice_amostra]:
                celula.set_facecolor('#1A3A1A')
                celula.set_text_props(color='#81C784')
            else:
                celula.set_facecolor('#3A1A1A')
                celula.set_text_props(color='#E57373')

plt.tight_layout()
plt.savefig('item3_pos_processamento.png', dpi=150, bbox_inches='tight',
            facecolor=FUNDO_ESCURO)
plt.close()
print("Salvo: item3_pos_processamento.png")

# ============================================================
# FIGURA 4 — Item 4: Validação da Rede (Tabela + Taxa de Acerto)
# ============================================================
fig_item4, eixos_item4 = plt.subplots(1, 2, figsize=(14, 6))
fig_item4.patch.set_facecolor(FUNDO_ESCURO)
fig_item4.suptitle("Item 4 — Validação da Rede com Conjunto de Teste",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# --- Subplot esquerdo: Tabela completa de validação ---
eixo_tabela_val = eixos_item4[0]
eixo_tabela_val.set_facecolor(FUNDO_ESCURO)
eixo_tabela_val.axis('off')
eixo_tabela_val.set_title("Resultados por Amostra de Teste",
                          color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

cabecalho_validacao = ['Amostra', 'x1', 'x2', 'd', 'y', 'yₚₒₛ']
dados_tabela_validacao = [
    [str(registros_teste[i]['amostra']),
     f'{entradas_teste[i][0]:.4f}',
     f'{entradas_teste[i][1]:.4f}',
     f'{saidas_desejadas_teste[i]:+d}',
     f'{saida_rede_teste[i]:.4f}',
     f'{saida_pos_processada[i]:+d}']
    for i in range(num_amostras_teste)
]

# Adiciona linha de taxa de acerto
dados_tabela_validacao.append(['', '', '', '', 'Taxa:', f'{taxa_acerto:.1f}%'])

tabela_validacao = eixo_tabela_val.table(
    cellText=dados_tabela_validacao, colLabels=cabecalho_validacao,
    loc='center', cellLoc='center'
)
tabela_validacao.auto_set_font_size(False)
tabela_validacao.set_fontsize(9)
tabela_validacao.scale(1.2, 1.9)

for (linha, coluna), celula in tabela_validacao.get_celld().items():
    celula.set_edgecolor(COR_GRADE)
    if linha == 0:
        celula.set_facecolor(COR_DESTAQUE)
        celula.set_text_props(color='white', fontweight='bold')
    elif linha == num_amostras_teste + 1:
        celula.set_facecolor('#1A2A3A')
        celula.set_text_props(color='#FFB74D', fontweight='bold')
    elif linha >= 1:
        indice_amostra = linha - 1
        if indice_amostra < num_amostras_teste:
            if saida_pos_processada[indice_amostra] == saidas_desejadas_teste[indice_amostra]:
                celula.set_facecolor('#1A3A1A')
                celula.set_text_props(color='#81C784')
            else:
                celula.set_facecolor('#3A1A1A')
                celula.set_text_props(color='#E57373')

# --- Subplot direito: Gráfico de fronteira de decisão ---
eixo_fronteira = eixos_item4[1]
aplicar_tema_escuro(eixo_fronteira)

# Cria uma grade para plotar a superfície de decisão
resolucao_grade = 200
x1_min, x1_max = 0.0, 1.0
x2_min, x2_max = 0.0, 1.0
grade_x1 = np.linspace(x1_min, x1_max, resolucao_grade)
grade_x2 = np.linspace(x2_min, x2_max, resolucao_grade)
malha_x1, malha_x2 = np.meshgrid(grade_x1, grade_x2)
pontos_grade = np.column_stack([malha_x1.ravel(), malha_x2.ravel()])

# Calcula a saída da rede para todos os pontos da grade
phi_grade = ativacao_gaussiana(pontos_grade, centros_clusters, variancias_clusters)
phi_grade_bias = np.column_stack([np.ones(len(pontos_grade)), phi_grade])
saida_grade = phi_grade_bias @ pesos_saida
classe_grade = funcao_sinal(saida_grade).reshape(malha_x1.shape)

# Plota a superfície de decisão
eixo_fronteira.contourf(malha_x1, malha_x2, classe_grade, levels=[-1.5, 0, 1.5],
                        colors=['#2A1A1A', '#1A2A1A'], alpha=0.5)
eixo_fronteira.contour(malha_x1, malha_x2, classe_grade, levels=[0],
                       colors=['#FFB74D'], linewidths=2, linestyles='--')

# Plota os padrões de teste
for i in range(num_amostras_teste):
    cor = '#81C784' if saida_pos_processada[i] == saidas_desejadas_teste[i] else '#E57373'
    marcador = 'o' if saidas_desejadas_teste[i] == 1 else 'x'
    eixo_fronteira.scatter(entradas_teste[i][0], entradas_teste[i][1],
                           c=cor, marker=marcador, s=100, edgecolors='white',
                           linewidths=1.2, zorder=5)

# Plota os centros dos clusters
for indice_cluster in range(2):
    eixo_fronteira.scatter(
        centros_clusters[indice_cluster][0],
        centros_clusters[indice_cluster][1],
        c='white', marker='*', s=200, edgecolors=cores_cluster[indice_cluster],
        linewidths=2, zorder=6
    )

eixo_fronteira.set_xlabel("x1", fontsize=10)
eixo_fronteira.set_ylabel("x2", fontsize=10)
eixo_fronteira.set_title(f"Fronteira de Decisão  |  Taxa de Acerto: {taxa_acerto:.1f}%",
                         color=COR_TEXTO, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('item4_validacao.png', dpi=150, bbox_inches='tight',
            facecolor=FUNDO_ESCURO)
plt.close()
print("Salvo: item4_validacao.png")

# ============================================================
# FIGURA 5 — Item 5: Estratégias para Aumentar a Taxa de Acerto
# ============================================================
fig_item5, eixos_item5 = plt.subplots(1, 2, figsize=(14, 6))
fig_item5.patch.set_facecolor(FUNDO_ESCURO)
fig_item5.suptitle("Item 5 — Estratégias para Melhorar a Taxa de Acerto",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# --- Subplot esquerdo: Visualização com mais clusters ---
eixo_mais_clusters = eixos_item5[0]
aplicar_tema_escuro(eixo_mais_clusters)

# Plota todos os padrões de treinamento
eixo_mais_clusters.scatter(
    entradas_treinamento[mascara_ausencia, 0],
    entradas_treinamento[mascara_ausencia, 1],
    c='#E57373', marker='x', s=60, alpha=0.7, label='Ausência (d=-1)', zorder=2
)
eixo_mais_clusters.scatter(
    entradas_treinamento[mascara_radiacao, 0],
    entradas_treinamento[mascara_radiacao, 1],
    c='#81C784', marker='o', s=60, alpha=0.7, label='Presença (d=1)', zorder=2
)

# Plota centros atuais
for indice_cluster in range(2):
    eixo_mais_clusters.scatter(
        centros_clusters[indice_cluster][0],
        centros_clusters[indice_cluster][1],
        c='white', marker='*', s=300, edgecolors='#FFB74D',
        linewidths=2, zorder=5
    )

eixo_mais_clusters.set_xlabel("x1", fontsize=10)
eixo_mais_clusters.set_ylabel("x2", fontsize=10)
eixo_mais_clusters.set_title("Distribuição dos Dados de Treinamento",
                             color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_mais_clusters.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                          labelcolor=COR_TEXTO, fontsize=8)

# --- Subplot direito: Tabela com estratégias ---
eixo_estrategias = eixos_item5[1]
eixo_estrategias.set_facecolor(FUNDO_ESCURO)
eixo_estrategias.axis('off')
eixo_estrategias.set_title("Estratégias Propostas",
                           color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

estrategias = [
    ["1", "Aumentar o número\nde clusters (K>2)", "Mais neurônios RBF\ncapturam mais regiões"],
    ["2", "Usar K-means com\ntodos os padrões", "Clusters de ambas as\nclasses (d=1 e d=-1)"],
    ["3", "Ajustar a taxa de\naprendizado (η)", "Teste com η=0.05, 0.1\npara convergência"],
    ["4", "Normalizar os\ndados de entrada", "Padronização para\nmédia=0 e σ=1"],
    ["5", "Usar validação\ncruzada", "Avaliar generalização\ncom k-fold"],
]

tabela_estrategias = eixo_estrategias.table(
    cellText=estrategias,
    colLabels=['#', 'Estratégia', 'Detalhamento'],
    loc='center', cellLoc='center'
)
tabela_estrategias.auto_set_font_size(False)
tabela_estrategias.set_fontsize(9)
tabela_estrategias.scale(1.3, 2.5)

for (linha, coluna), celula in tabela_estrategias.get_celld().items():
    celula.set_edgecolor(COR_GRADE)
    if linha == 0:
        celula.set_facecolor(COR_DESTAQUE)
        celula.set_text_props(color='white', fontweight='bold')
    elif linha % 2 == 0:
        celula.set_facecolor('#1C2128')
        celula.set_text_props(color=COR_TEXTO)
    else:
        celula.set_facecolor(FUNDO_CARD)
        celula.set_text_props(color=COR_TEXTO)

plt.tight_layout()
plt.savefig('item5_estrategias.png', dpi=150, bbox_inches='tight',
            facecolor=FUNDO_ESCURO)
plt.close()
print("Salvo: item5_estrategias.png")

# ============================================================
# RESUMO FINAL NO TERMINAL
# ============================================================
print("\n" + "=" * 60)
print("  RESUMO DOS RESULTADOS - RBF1")
print("=" * 60)
print(f"\n  Cluster 1: Centro ({centros_clusters[0][0]:.4f}, {centros_clusters[0][1]:.4f})"
      f"  sigma2 = {variancias_clusters[0]:.6f}")
print(f"  Cluster 2: Centro ({centros_clusters[1][0]:.4f}, {centros_clusters[1][1]:.4f})"
      f"  sigma2 = {variancias_clusters[1]:.6f}")
print(f"\n  W21,0 = {pesos_saida[0]:.6f}")
print(f"  W21,1 = {pesos_saida[1]:.6f}")
print(f"  W21,2 = {pesos_saida[2]:.6f}")
print(f"\n  Epocas: {num_epocas_convergencia}")
print(f"  EQM final: {eqm_final:.2e}")
print(f"  Taxa de acerto: {taxa_acerto:.1f}%")
print("=" * 60)
print("\nImagens geradas:")
for arquivo in ['item1_clusters_kmeans.png', 'item2_pesos_convergencia.png',
                'item3_pos_processamento.png', 'item4_validacao.png',
                'item5_estrategias.png']:
    print(f"  • {arquivo}")
