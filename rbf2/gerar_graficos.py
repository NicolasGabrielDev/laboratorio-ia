"""
RBF2 - Script de geração de gráficos científicos e relatórios em imagem.
Lê os dados salvos em JSON e gera os arquivos PNG correspondentes com um visual
premium escuro (dark scientific theme).
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 1. CARREGAR DADOS DOS JSONs
# ============================================================
with open('training_data.json', 'r', encoding='utf-8') as f:
    train_raw = json.load(f)

with open('test_data.json', 'r', encoding='utf-8') as f:
    test_raw = json.load(f)

with open('training_results.json', 'r', encoding='utf-8') as f:
    training_results = json.load(f)

with open('validation_results.json', 'r', encoding='utf-8') as f:
    validation_results = json.load(f)

# Cores do Tema Científico Premium (Dark/Neon)
BG_CLR = "#0B0F19"      # Fundo ultra escuro
CARD_BG = "#161B2E"     # Fundo dos cards/tabelas
TEXT_CLR = "#F1F5F9"    # Texto principal claro
MUTED_CLR = "#94A3B8"   # Texto secundário
GRID_CLR = "#2A3554"    # Linhas de grade
ACCENT_BLUE = "#38BDF8" # Azul neon
ACCENT_GREEN = "#34D399"# Verde neon
ACCENT_RED = "#F87171"  # Vermelho coral
ACCENT_GOLD = "#FBBF24" # Ouro neon

PALETTE = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_GOLD]

def apply_dark_theme(ax):
    ax.set_facecolor(BG_CLR)
    ax.spines['bottom'].set_color(GRID_CLR)
    ax.spines['top'].set_color(GRID_CLR)
    ax.spines['right'].set_color(GRID_CLR)
    ax.spines['left'].set_color(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR, which='both')
    ax.grid(True, color=GRID_CLR, linestyle='--', alpha=0.3)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)

# ============================================================
# ITEM 1: TABELA DE TREINAMENTOS (item1_tabela_treinamentos.png)
# ============================================================
def gerar_item1_tabela_treinamentos():
    print("Gerando item1_tabela_treinamentos.png...")
    fig = plt.figure(figsize=(10, 5), facecolor=BG_CLR)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(BG_CLR)
    ax.axis('off')
    
    # Prepara dados
    linhas_tabela = []
    headers = ["Rede / Topologia", "Treinamento", "Semente", "Épocas para Convergência", "EQM Final (Ajuste)"]
    
    for r_idx, rede in enumerate(["Rede_5", "Rede_10", "Rede_15"]):
        r_nome = f"Rede {rede.split('_')[1]} (N1={rede.split('_')[1]})"
        for t_idx, t in enumerate(training_results["treinamentos"][rede]):
            # Mostra dados formatados
            linhas_tabela.append([
                r_nome if t_idx == 0 else "",
                t["treinamento"],
                str(t["semente"]),
                f"{t['epocas']:,}",
                f"{t['eqm_final']:.8f}"
            ])
            
    # Desenha a tabela com Matplotlib de forma super limpa
    tabela = ax.table(
        cellText=linhas_tabela,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.15, 0.15, 0.2, 0.2]
    )
    
    # Estilização manual
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    
    for (row, col), cell in tabela.get_celld().items():
        cell.set_text_props(color=TEXT_CLR, fontfamily='sans-serif')
        cell.set_edgecolor(GRID_CLR)
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_text_props(weight='bold', color=BG_CLR)
            cell.set_facecolor(ACCENT_BLUE)
        else:
            # Cores alternadas para visual premium
            if row in [1, 2, 3]: # Rede 5
                cell.set_facecolor("#111827")
            elif row in [4, 5, 6]: # Rede 10
                cell.set_facecolor("#1F2937")
            else: # Rede 15
                cell.set_facecolor("#374151")
                
            # Destaca a melhor configuração (Rede 10 T2 na linha 5)
            if row == 5:
                cell.set_text_props(color=ACCENT_GREEN, weight='bold')
                
    ax.set_title("Resumo de Convergência das Redes RBF", fontsize=14, pad=20, color=ACCENT_BLUE, weight='bold')
    
    plt.tight_layout()
    plt.savefig('item1_tabela_treinamentos.png', dpi=150, facecolor=BG_CLR, bbox_inches='tight')
    plt.close()
    print("Salvo: item1_tabela_treinamentos.png")

# ============================================================
# ITEM 2: CURVAS DE APRENDIZADO (item2_eqm_epocas.png)
# ============================================================
def gerar_item2_eqm_epocas():
    print("Gerando item2_eqm_epocas.png...")
    # Criar subplots 3x1 (uma folha, não superpostos)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), facecolor=BG_CLR)
    
    redes = ["Rede_5", "Rede_10", "Rede_15"]
    nomes_bonitos = ["Rede 1 (N1=5)", "Rede 2 (N1=10)", "Rede 3 (N1=15)"]
    
    for idx, (rede, nome) in enumerate(zip(redes, nomes_bonitos)):
        ax = axes[idx]
        apply_dark_theme(ax)
        
        # Encontra o melhor treinamento baseado no EQM final
        treinos = training_results["treinamentos"][rede]
        best_t = min(treinos, key=lambda t: t["eqm_final"])
        
        hist_eqm = best_t["eqm_historico"]
        epocas = np.arange(1, len(hist_eqm) + 1)
        
        # Plot com gradiente/glow sutil
        ax.plot(epocas, hist_eqm, color=PALETTE[idx], linewidth=2.0, 
                label=f"Melhor Treino ({best_t['treinamento']}): EQM Final = {best_t['eqm_final']:.6f}")
        
        ax.set_yscale('log')
        ax.set_title(f"Curva de Aprendizagem - {nome}", fontsize=11, weight='bold', color=TEXT_CLR, loc='left')
        ax.set_xlabel("Épocas", fontsize=9, color=MUTED_CLR)
        ax.set_ylabel("EQM (Escala Log)", fontsize=9, color=MUTED_CLR)
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
        
        # Destaca o ponto final de parada
        ax.scatter(epocas[-1], hist_eqm[-1], color=ACCENT_RED, s=50, zorder=5)
        ax.annotate(f"Época {best_t['epocas']:,}\nEQM {best_t['eqm_final']:.6f}",
                    xy=(best_t['epocas'], best_t['eqm_final']),
                    xytext=(best_t['epocas'] * 0.5, best_t['eqm_final'] * 3.0),
                    arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.2),
                    color=TEXT_CLR, fontsize=8)

    plt.suptitle("Análise de Queda do Erro Quadrático Médio (EQM) × Épocas", 
                 fontsize=14, color=ACCENT_BLUE, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('item2_eqm_epocas.png', dpi=150, facecolor=BG_CLR, bbox_inches='tight')
    plt.close()
    print("Salvo: item2_eqm_epocas.png")

# ============================================================
# ITEM 4: TABELA DE VALIDAÇÃO (item4_validacao.png)
# ============================================================
def gerar_item4_validacao():
    print("Gerando item4_validacao.png...")
    # Tabela gigante: 15 amostras + entradas/desejados + previsões para as 9 configs
    fig = plt.figure(figsize=(15, 9), facecolor=BG_CLR)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    
    # Headers
    headers = [
        "Amostra", "x1", "x2", "x3", "Desejado (d)", 
        "R1 (T1)", "R1 (T2)", "R1 (T3)", 
        "R2 (T1)", "R2 (T2)", "R2 (T3)", 
        "R3 (T1)", "R3 (T2)", "R3 (T3)"
    ]
    
    # Monta as linhas de dados
    linhas_tabela = []
    for am_idx in range(15):
        # Detalhes básicos da amostra
        am_num = f"{am_idx + 1:02d}"
        x1 = f"{test_raw[am_idx]['x1']:.4f}"
        x2 = f"{test_raw[am_idx]['x2']:.4f}"
        x3 = f"{test_raw[am_idx]['x3']:.4f}"
        d = f"{test_raw[am_idx]['d']:.4f}"
        
        row_cells = [am_num, x1, x2, x3, d]
        
        # Adiciona as predições de cada rede e treino
        for rede in ["Rede_5", "Rede_10", "Rede_15"]:
            for t_idx in range(3):
                y_pred = validation_results[rede][t_idx]["predicoes"][am_idx]["y"]
                row_cells.append(f"{y_pred:.4f}")
                
        linhas_tabela.append(row_cells)
        
    # Adiciona as linhas do Erro Relativo Médio (%)
    row_erm = ["Erro Relativo Médio (%)", "", "", "", ""]
    for rede in ["Rede_5", "Rede_10", "Rede_15"]:
        for t_idx in range(3):
            erm = validation_results[rede][t_idx]["erro_relativo_medio_pct"]
            row_erm.append(f"{erm:.4f}%")
    linhas_tabela.append(row_erm)
    
    # Adiciona as linhas da Variância (%)
    row_var = ["Variância de Erro (%)", "", "", "", ""]
    for rede in ["Rede_5", "Rede_10", "Rede_15"]:
        for t_idx in range(3):
            var = validation_results[rede][t_idx]["variancia_pct"]
            row_var.append(f"{var:.4f}%")
    linhas_tabela.append(row_var)

    # Desenha a tabela com Matplotlib
    tabela = ax.table(
        cellText=linhas_tabela,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8)
    
    # Ajusta alturas e larguras
    for (row, col), cell in tabela.get_celld().items():
        cell.set_text_props(color=TEXT_CLR, fontfamily='monospace')
        cell.set_edgecolor(GRID_CLR)
        cell.set_linewidth(1.0)
        
        if row == 0:
            cell.set_text_props(weight='bold', color=BG_CLR, fontfamily='sans-serif')
            cell.set_facecolor(ACCENT_BLUE)
            if col in [5, 6, 7]:
                cell.set_facecolor("#0284C7")
            elif col in [8, 9, 10]:
                cell.set_facecolor("#059669")
            elif col in [11, 12, 13]:
                cell.set_facecolor("#D97706")
        else:
            # Diferenciação de fundo para colunas e linhas
            if row in [16, 17]:  # Linhas de ERM e Variância
                cell.set_facecolor("#1E293B")
                cell.set_text_props(weight='bold', color=ACCENT_GOLD)
                if col >= 5:
                    # Destaca a melhor configuração global (Rede 2 T2 na coluna 9)
                    if col == 9:
                        cell.set_text_props(color=ACCENT_GREEN, weight='bold')
                        cell.set_facecolor("#064E3B")
            else:
                cell.set_facecolor(CARD_BG if row % 2 == 0 else BG_CLR)
                if col == 9: # Coluna da melhor predição Rede 2 T2
                    cell.set_text_props(color=ACCENT_GREEN, weight='bold')
                    cell.set_facecolor("#064E3B")
                    
    ax.set_title("Matriz Geral de Validação com Conjunto de Teste (15 Amostras)", 
                 fontsize=16, pad=30, color=ACCENT_BLUE, weight='bold')
    
    plt.tight_layout()
    plt.savefig('item4_validacao.png', dpi=150, facecolor=BG_CLR, bbox_inches='tight')
    plt.close()
    print("Salvo: item4_validacao.png")

# ============================================================
# ITEM 5: COMPARAÇÃO DO MELHOR MODELO (item5_melhor_config.png)
# ============================================================
def gerar_item5_melhor_config():
    print("Gerando item5_melhor_config.png...")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_CLR)
    apply_dark_theme(ax)
    
    # Melhor config identificada: Rede 2 (10 neurônios) - Treinamento T2 (semente 107)
    best_rede = "Rede_10"
    best_t_idx = 1 # T2 (índice 1)
    
    dados_best = validation_results[best_rede][best_t_idx]
    predicoes = dados_best["predicoes"]
    
    amostras = np.arange(1, 16)
    valores_desejados = np.array([p["d"] for p in predicoes])
    valores_estimados = np.array([p["y"] for p in predicoes])
    erros_relativos = np.array([p["erro_relativo_pct"] for p in predicoes])
    
    # Plot desejado vs estimado
    ax.plot(amostras, valores_desejados, color=ACCENT_BLUE, linestyle='-', marker='o', 
            linewidth=2.0, markersize=6, label="Valores Desejados (Gasolina Real)")
    ax.plot(amostras, valores_estimados, color=ACCENT_GREEN, linestyle='--', marker='s', 
            linewidth=2.0, markersize=6, label=f"Valores Estimados pela Rede RBF (Rede 2 T2)")
    
    # Preenche o erro sutilmente entre as curvas
    ax.fill_between(amostras, valores_desejados, valores_estimados, color=ACCENT_RED, alpha=0.15,
                    label="Erro de Mapeamento")
    
    ax.set_title("Aproximação Funcional: Consumo Desejado vs Estimado (Melhor Configuração)", 
                 fontsize=12, pad=15, color=TEXT_CLR, weight='bold')
    ax.set_xlabel("Amostra do Conjunto de Teste", fontsize=10)
    ax.set_ylabel("Quantidade de Gasolina Injetada (y)", fontsize=10)
    ax.set_xticks(amostras)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9, loc='upper left')
    
    # Plota no mesmo frame um inset (eixo secundário) com as barras de erro relativo
    ax2 = ax.twinx()
    ax2.bar(amostras, erros_relativos, color=ACCENT_GOLD, alpha=0.25, width=0.4, 
            edgecolor=ACCENT_GOLD, linewidth=0.8, label="Erro Relativo por Amostra (%)")
    ax2.set_ylabel("Erro Relativo (%)", color=ACCENT_GOLD, fontsize=10)
    ax2.tick_params(colors=ACCENT_GOLD)
    ax2.spines['right'].set_color(ACCENT_GOLD)
    
    # Adiciona anotação de texto sutil com a métrica agregada
    stats_text = (f"Erro Médio: {dados_best['erro_relativo_medio_pct']:.4f}%\n"
                  f"Variância: {dados_best['variancia_pct']:.4f}%")
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=9.5, weight='bold',
            color=TEXT_CLR, bbox=dict(facecolor=CARD_BG, edgecolor=GRID_CLR, boxstyle='round,pad=0.8'),
            ha='right', va='bottom')
            
    plt.tight_layout()
    plt.savefig('item5_melhor_config.png', dpi=150, facecolor=BG_CLR, bbox_inches='tight')
    plt.close()
    print("Salvo: item5_melhor_config.png")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    gerar_item1_tabela_treinamentos()
    gerar_item2_eqm_epocas()
    gerar_item4_validacao()
    gerar_item5_melhor_config()
    print("\n--- Todos os gráficos foram gerados e salvos com sucesso! ---")
