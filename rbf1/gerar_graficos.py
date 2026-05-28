"""
RBF1 - Regeneração de gráficos a partir dos arquivos JSON.
Este script lê os resultados salvos e recria todas as imagens.
Não é necessário re-treinar a rede para gerar os gráficos.
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
# Carregar todos os JSONs necessários
# ============================================================
with open('training_data.json') as arquivo_treinamento:
    registros_treinamento = json.load(arquivo_treinamento)

with open('test_data.json') as arquivo_teste:
    registros_teste = json.load(arquivo_teste)

with open('training_results.json') as arquivo_resultados:
    resultados_treinamento = json.load(arquivo_resultados)

with open('validation_results.json') as arquivo_validacao:
    resultados_validacao = json.load(arquivo_validacao)

# Reconstruir arrays numpy a partir dos JSONs
entradas_treinamento = np.array([[r['x1'], r['x2']] for r in registros_treinamento])
saidas_desejadas_treinamento = np.array([r['d'] for r in registros_treinamento])

entradas_teste = np.array([[r['x1'], r['x2']] for r in registros_teste])
saidas_desejadas_teste = np.array([r['d'] for r in registros_teste])

num_amostras_teste = len(entradas_teste)

# Extrair dados do k-means
centros_clusters = np.array([
    [c['centro']['x1'], c['centro']['x2']]
    for c in resultados_treinamento['kmeans']['clusters']
])
variancias_clusters = np.array([
    c['variancia'] for c in resultados_treinamento['kmeans']['clusters']
])

# Extrair pesos da camada de saída
pesos_saida = np.array([
    resultados_treinamento['camada_saida']['pesos']['W21_0_bias'],
    resultados_treinamento['camada_saida']['pesos']['W21_1'],
    resultados_treinamento['camada_saida']['pesos']['W21_2']
])
historico_eqm = resultados_treinamento['camada_saida']['eqm_historico']
num_epocas_convergencia = resultados_treinamento['camada_saida']['num_epocas']
eqm_final = resultados_treinamento['camada_saida']['eqm_final']

# Extrair resultados de validação
taxa_acerto = resultados_validacao['taxa_acerto_pct']
saida_rede_teste = np.array([r['y'] for r in resultados_validacao['resultados_por_amostra']])
saida_pos_processada = np.array([r['y_pos'] for r in resultados_validacao['resultados_por_amostra']])

# Máscaras para os padrões de treinamento
mascara_radiacao = saidas_desejadas_treinamento == 1
mascara_ausencia = saidas_desejadas_treinamento == -1
entradas_com_radiacao = entradas_treinamento[mascara_radiacao]

# Recalcular rótulos dos clusters para o scatter plot
distancias_cluster = np.array([
    np.linalg.norm(entradas_com_radiacao - centro, axis=1)
    for centro in centros_clusters
])
rotulos_clusters = np.argmin(distancias_cluster, axis=0)


# ============================================================
# Funções auxiliares da rede RBF
# ============================================================
def ativacao_gaussiana(entradas, centros, variancias):
    """Calcula ativação Gaussiana (RBF): φ_j(x) = exp(-||x - c_j||² / (2σ²))"""
    num_padroes = len(entradas)
    num_neuronios = len(centros)
    phi = np.zeros((num_padroes, num_neuronios))
    for j in range(num_neuronios):
        diferenca = entradas - centros[j]
        distancia_quadrada = np.sum(diferenca ** 2, axis=1)
        phi[:, j] = np.exp(-distancia_quadrada / (2 * variancias[j]))
    return phi


def funcao_sinal(valores):
    """Função sinal: y>=0 -> 1, y<0 -> -1"""
    return np.where(valores >= 0, 1, -1)


# ============================================================
# Estilos globais (tema escuro)
# ============================================================
PALETA_CORES = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#CE93D8']
FUNDO_ESCURO = '#0D1117'
FUNDO_CARD = '#161B22'
COR_GRADE = '#30363D'
COR_TEXTO = '#E6EDF3'
COR_DESTAQUE = '#58A6FF'
cores_cluster = ['#4FC3F7', '#81C784']


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
# FIGURA 1 — Item 1: Resultados do K-means
# ============================================================
fig_item1, eixos_item1 = plt.subplots(1, 2, figsize=(14, 6))
fig_item1.patch.set_facecolor(FUNDO_ESCURO)
fig_item1.suptitle("Item 1 — Treinamento da Camada Oculta (K-means)",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# Scatter plot dos clusters
eixo_scatter = eixos_item1[0]
aplicar_tema_escuro(eixo_scatter)

eixo_scatter.scatter(
    entradas_treinamento[mascara_ausencia, 0],
    entradas_treinamento[mascara_ausencia, 1],
    c='#E57373', marker='x', s=60, alpha=0.7, label='Ausência (d=-1)', zorder=2
)

for indice_cluster in range(2):
    pontos = entradas_com_radiacao[rotulos_clusters == indice_cluster]
    eixo_scatter.scatter(
        pontos[:, 0], pontos[:, 1],
        c=cores_cluster[indice_cluster], marker='o', s=70, alpha=0.8,
        label=f'Presença - Cluster {indice_cluster + 1}', zorder=3
    )

for indice_cluster in range(2):
    eixo_scatter.scatter(
        centros_clusters[indice_cluster][0], centros_clusters[indice_cluster][1],
        c='white', marker='*', s=300, edgecolors=cores_cluster[indice_cluster],
        linewidths=2, zorder=5, label=f'Centro {indice_cluster + 1}'
    )

eixo_scatter.set_xlabel("x1", fontsize=11)
eixo_scatter.set_ylabel("x2", fontsize=11)
eixo_scatter.set_title("Distribuição dos Padrões e Clusters K-means",
                       color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_scatter.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                    labelcolor=COR_TEXTO, fontsize=8, loc='upper right')

# Tabela com centros e variâncias
eixo_tabela_clusters = eixos_item1[1]
eixo_tabela_clusters.set_facecolor(FUNDO_ESCURO)
eixo_tabela_clusters.axis('off')
eixo_tabela_clusters.set_title("Centros e Variâncias dos Clusters",
                               color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

dados_tabela_clusters = [
    [f"Cluster {k + 1}",
     f"({centros_clusters[k][0]:.4f}, {centros_clusters[k][1]:.4f})",
     f"{variancias_clusters[k]:.6f}"]
    for k in range(2)
]

tabela_clusters = eixo_tabela_clusters.table(
    cellText=dados_tabela_clusters,
    colLabels=['Cluster', 'Centro (x1, x2)', 'Variância (σ²)'],
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
# FIGURA 2 — Item 2: Pesos + Convergência EQM
# ============================================================
fig_item2 = plt.figure(figsize=(14, 6))
fig_item2.patch.set_facecolor(FUNDO_ESCURO)
fig_item2.suptitle("Item 2 — Treinamento da Camada de Saída (Regra Delta Generalizada)",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

gs_item2 = gridspec.GridSpec(1, 2, wspace=0.35)

# Tabela de pesos
eixo_pesos = fig_item2.add_subplot(gs_item2[0, 0])
eixo_pesos.set_facecolor(FUNDO_ESCURO)
eixo_pesos.axis('off')
eixo_pesos.set_title("Pesos do Neurônio de Saída",
                     color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

dados_tabela_pesos = [
    ['W21,0 (bias)', f'{pesos_saida[0]:.6f}'],
    ['W21,1', f'{pesos_saida[1]:.6f}'],
    ['W21,2', f'{pesos_saida[2]:.6f}']
]

tabela_pesos = eixo_pesos.table(
    cellText=dados_tabela_pesos, colLabels=['Peso', 'Valor'],
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

eixo_pesos.text(0.5, 0.08,
                f"Épocas: {num_epocas_convergencia}  |  EQM Final: {eqm_final:.2e}",
                transform=eixo_pesos.transAxes, ha='center', color='#81C784',
                fontsize=10, fontweight='bold')

# Curva EQM
eixo_eqm = fig_item2.add_subplot(gs_item2[0, 1])
aplicar_tema_escuro(eixo_eqm)

epocas_eixo = np.arange(1, len(historico_eqm) + 1)
eixo_eqm.semilogy(epocas_eixo, historico_eqm, color='#4FC3F7', linewidth=1.5, alpha=0.9)
eixo_eqm.fill_between(epocas_eixo, historico_eqm, alpha=0.15, color='#4FC3F7')
eixo_eqm.scatter([epocas_eixo[-1]], [historico_eqm[-1]], color='white',
                 s=60, zorder=5, edgecolors='#4FC3F7', linewidths=2)
eixo_eqm.axhline(y=1e-7, color='#E57373', linewidth=1.2,
                 linestyle='--', alpha=0.8, label='ε = 10⁻⁷')
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

eixo_sinal = eixos_item3[0]
aplicar_tema_escuro(eixo_sinal)

valores_x_sinal = np.linspace(-2, 2, 1000)
valores_y_sinal = np.where(valores_x_sinal >= 0, 1, -1)

eixo_sinal.plot(valores_x_sinal, valores_y_sinal, color='#4FC3F7',
                linewidth=2.5, label='yₚₒₛ = sign(y)')
eixo_sinal.axhline(y=0, color=COR_GRADE, linewidth=0.8)
eixo_sinal.axvline(x=0, color='#FFB74D', linewidth=1.5, linestyle='--',
                   alpha=0.8, label='Limiar (y=0)')

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

eixo_tabela_sinal = eixos_item3[1]
eixo_tabela_sinal.set_facecolor(FUNDO_ESCURO)
eixo_tabela_sinal.axis('off')
eixo_tabela_sinal.set_title("Saídas Antes e Após Pós-processamento",
                            color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

dados_tabela_sinal = [
    [str(resultados_validacao['resultados_por_amostra'][i]['amostra']),
     f'{saida_rede_teste[i]:.4f}',
     f'{saida_pos_processada[i]:+d}',
     f'{saidas_desejadas_teste[i]:+d}']
    for i in range(num_amostras_teste)
]

tabela_sinal = eixo_tabela_sinal.table(
    cellText=dados_tabela_sinal,
    colLabels=['Amostra', 'y (rede)', 'yₚₒₛ (sign)', 'd (desejado)'],
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
# FIGURA 4 — Item 4: Validação
# ============================================================
fig_item4, eixos_item4 = plt.subplots(1, 2, figsize=(14, 6))
fig_item4.patch.set_facecolor(FUNDO_ESCURO)
fig_item4.suptitle("Item 4 — Validação da Rede com Conjunto de Teste",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# Tabela de validação
eixo_tabela_val = eixos_item4[0]
eixo_tabela_val.set_facecolor(FUNDO_ESCURO)
eixo_tabela_val.axis('off')
eixo_tabela_val.set_title("Resultados por Amostra de Teste",
                          color=COR_TEXTO, fontsize=11, fontweight='bold', pad=15)

dados_tabela_validacao = [
    [str(resultados_validacao['resultados_por_amostra'][i]['amostra']),
     f'{entradas_teste[i][0]:.4f}',
     f'{entradas_teste[i][1]:.4f}',
     f'{saidas_desejadas_teste[i]:+d}',
     f'{saida_rede_teste[i]:.4f}',
     f'{saida_pos_processada[i]:+d}']
    for i in range(num_amostras_teste)
]
dados_tabela_validacao.append(['', '', '', '', 'Taxa:', f'{taxa_acerto:.1f}%'])

tabela_validacao = eixo_tabela_val.table(
    cellText=dados_tabela_validacao,
    colLabels=['Amostra', 'x1', 'x2', 'd', 'y', 'yₚₒₛ'],
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

# Fronteira de decisão
eixo_fronteira = eixos_item4[1]
aplicar_tema_escuro(eixo_fronteira)

resolucao_grade = 200
grade_x1 = np.linspace(0.0, 1.0, resolucao_grade)
grade_x2 = np.linspace(0.0, 1.0, resolucao_grade)
malha_x1, malha_x2 = np.meshgrid(grade_x1, grade_x2)
pontos_grade = np.column_stack([malha_x1.ravel(), malha_x2.ravel()])

phi_grade = ativacao_gaussiana(pontos_grade, centros_clusters, variancias_clusters)
phi_grade_bias = np.column_stack([np.ones(len(pontos_grade)), phi_grade])
saida_grade = phi_grade_bias @ pesos_saida
classe_grade = funcao_sinal(saida_grade).reshape(malha_x1.shape)

eixo_fronteira.contourf(malha_x1, malha_x2, classe_grade, levels=[-1.5, 0, 1.5],
                        colors=['#2A1A1A', '#1A2A1A'], alpha=0.5)
eixo_fronteira.contour(malha_x1, malha_x2, classe_grade, levels=[0],
                       colors=['#FFB74D'], linewidths=2, linestyles='--')

for i in range(num_amostras_teste):
    cor = '#81C784' if saida_pos_processada[i] == saidas_desejadas_teste[i] else '#E57373'
    marcador = 'o' if saidas_desejadas_teste[i] == 1 else 'x'
    eixo_fronteira.scatter(entradas_teste[i][0], entradas_teste[i][1],
                           c=cor, marker=marcador, s=100, edgecolors='white',
                           linewidths=1.2, zorder=5)

for indice_cluster in range(2):
    eixo_fronteira.scatter(
        centros_clusters[indice_cluster][0], centros_clusters[indice_cluster][1],
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
# FIGURA 5 — Item 5: Estratégias
# ============================================================
fig_item5, eixos_item5 = plt.subplots(1, 2, figsize=(14, 6))
fig_item5.patch.set_facecolor(FUNDO_ESCURO)
fig_item5.suptitle("Item 5 — Estratégias para Melhorar a Taxa de Acerto",
                   color=COR_TEXTO, fontsize=14, fontweight='bold')

# Dados de treinamento
eixo_dados = eixos_item5[0]
aplicar_tema_escuro(eixo_dados)

eixo_dados.scatter(
    entradas_treinamento[mascara_ausencia, 0],
    entradas_treinamento[mascara_ausencia, 1],
    c='#E57373', marker='x', s=60, alpha=0.7, label='Ausência (d=-1)', zorder=2
)
eixo_dados.scatter(
    entradas_treinamento[mascara_radiacao, 0],
    entradas_treinamento[mascara_radiacao, 1],
    c='#81C784', marker='o', s=60, alpha=0.7, label='Presença (d=1)', zorder=2
)

for indice_cluster in range(2):
    eixo_dados.scatter(
        centros_clusters[indice_cluster][0], centros_clusters[indice_cluster][1],
        c='white', marker='*', s=300, edgecolors='#FFB74D', linewidths=2, zorder=5
    )

eixo_dados.set_xlabel("x1", fontsize=10)
eixo_dados.set_ylabel("x2", fontsize=10)
eixo_dados.set_title("Distribuição dos Dados de Treinamento",
                     color=COR_TEXTO, fontsize=11, fontweight='bold')
eixo_dados.legend(facecolor=FUNDO_CARD, edgecolor=COR_GRADE,
                  labelcolor=COR_TEXTO, fontsize=8)

# Tabela de estratégias
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
# Resumo final
# ============================================================
print("\n" + "=" * 50)
print("  Todos os gráficos regenerados com sucesso!")
print("=" * 50)
for arquivo in ['item1_clusters_kmeans.png', 'item2_pesos_convergencia.png',
                'item3_pos_processamento.png', 'item4_validacao.png',
                'item5_estrategias.png']:
    print(f"  • {arquivo}")
