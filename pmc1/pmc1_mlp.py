"""
PMC1 - Rede Perceptron Multicamadas com Backpropagation
Topologia: 3 entradas -> 12 neuronios ocultos -> 10 neuronios saida -> 1 saida
Funcao de ativacao: logistica (sigmoid)
Taxa de aprendizado: eta = 0.1
Precisao: epsilon = 1e-6
5 treinamentos com pesos iniciais aleatorios diferentes
"""

import numpy as np
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
LEARNING_RATE = 0.1
EPSILON = 1e-6
MAX_EPOCHS = 100000

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
with open('training_data.json') as f:
    train_raw = json.load(f)

with open('test_data.json') as f:
    test_raw = json.load(f)

with open('metadata.json') as f:
    meta = json.load(f)

X_train = np.array([[r['x1'], r['x2'], r['x3']] for r in train_raw])
d_train = np.array([[r['d']] for r in train_raw])

X_test = np.array([[r['x1'], r['x2'], r['x3']] for r in test_raw])
d_test = np.array([[r['d']] for r in test_raw])

N_train = len(X_train)
N_test  = len(X_test)

# ============================================================
# 2. FUNCOES DA REDE
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(y):   # y = sigmoid(x) ja calculado
    return y * (1.0 - y)

def forward(X, W1, b1, W2, b2):
    """
    X  : (N, 3)
    W1 : (12, 3)  b1: (12,)
    W2 : (10, 12) b2: (10,)
    W3 : (1, 10)  b3: (1,)  -- embutido em W2/b2 para simplificar
    """
    h1 = sigmoid(X @ W1.T + b1)      # (N, 12)
    h2 = sigmoid(h1 @ W2.T + b2)     # (N, 10)
    return h1, h2

def forward_full(X, W1, b1, W2, b2, W3, b3):
    h1 = sigmoid(X @ W1.T + b1)       # (N, 12)
    h2 = sigmoid(h1 @ W2.T + b2)      # (N, 10)
    out = sigmoid(h2 @ W3.T + b3)     # (N, 1)
    return h1, h2, out

def eqm(d, out):
    """Erro Quadratico Medio sobre todos os padroes."""
    return float(np.mean(0.5 * (d - out) ** 2))

# ============================================================
# 3. BACKPROPAGATION
# ============================================================
def train_mlp(X, d, seed, eta=LEARNING_RATE, epsilon=EPSILON, max_epochs=MAX_EPOCHS):
    rng = np.random.default_rng(seed)
    # Pesos aleatorios entre 0 e 1 (conforme enunciado)
    W1 = rng.uniform(0, 1, (12, 3))
    b1 = rng.uniform(0, 1, (12,))
    W2 = rng.uniform(0, 1, (10, 12))
    b2 = rng.uniform(0, 1, (10,))
    W3 = rng.uniform(0, 1, (1, 10))
    b3 = rng.uniform(0, 1, (1,))

    history = []
    epoch = 0

    while epoch < max_epochs:
        # Forward
        h1, h2, out = forward_full(X, W1, b1, W2, b2, W3, b3)
        eq = eqm(d, out)
        history.append(eq)

        if eq < epsilon:
            stop_reason = "precision_reached"
            break

        # --- Backprop ---
        # Delta camada saida (1 neuronio)
        delta3 = (d - out) * sigmoid_deriv(out)          # (N, 1)

        # Delta camada oculta 2
        delta2 = (delta3 @ W3) * sigmoid_deriv(h2)       # (N, 10)

        # Delta camada oculta 1
        delta1 = (delta2 @ W2) * sigmoid_deriv(h1)       # (N, 12)

        # Atualiza pesos (batch)
        W3 += eta * (delta3.T @ h2) / N_train
        b3 += eta * delta3.mean(axis=0)

        W2 += eta * (delta2.T @ h1) / N_train
        b2 += eta * delta2.mean(axis=0)

        W1 += eta * (delta1.T @ X) / N_train
        b1 += eta * delta1.mean(axis=0)

        epoch += 1
    else:
        stop_reason = "max_epochs_reached"

    return {
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3,
        'epochs': epoch,
        'eqm_final': history[-1],
        'history': history,
        'stop_reason': stop_reason
    }

# ============================================================
# 4. EXECUTAR 5 TREINAMENTOS
# ============================================================
print("Iniciando 5 treinamentos...")
seeds = [42, 7, 2024, 314, 99]
results = []

for i, seed in enumerate(seeds):
    print(f"  Treinamento T{i+1} (seed={seed})...", end=' ', flush=True)
    r = train_mlp(X_train, d_train, seed=seed, eta=LEARNING_RATE, epsilon=EPSILON, max_epochs=MAX_EPOCHS)
    results.append(r)
    print(f"EQM={r['eqm_final']:.6f}  Epocas={r['epochs']}  Parada={r['stop_reason']}")

# ============================================================
# 5. SALVAR RESULTADOS EM JSON
# ============================================================
summary = []
for i, r in enumerate(results):
    summary.append({
        "treinamento": f"T{i+1}",
        "seed": seeds[i],
        "eqm_final": r['eqm_final'],
        "num_epocas": r['epochs'],
        "stop_reason": r['stop_reason'],
        "epsilon": EPSILON,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE
    })

with open('training_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nResultados salvos -> training_results.json")

# ============================================================
# 6. VALIDACAO COM CONJUNTO DE TESTE
# ============================================================
validation = []
for i, r in enumerate(results):
    _, _, out = forward_full(X_test, r['W1'], r['b1'], r['W2'], r['b2'], r['W3'], r['b3'])
    out_flat = out.flatten()
    d_flat   = d_test.flatten()

    # Erro relativo por amostra (%)
    rel_err = np.abs(d_flat - out_flat) / (np.abs(d_flat) + 1e-12) * 100.0
    mean_rel_err = float(np.mean(rel_err))
    variance     = float(np.var(rel_err))

    validation.append({
        "treinamento": f"T{i+1}",
        "y_rede": out_flat.tolist(),
        "erro_relativo_por_amostra": rel_err.tolist(),
        "erro_relativo_medio_pct": mean_rel_err,
        "variancia_pct": variance
    })

with open('validation_results.json', 'w') as f:
    json.dump(validation, f, indent=2)
print("Resultados de validacao salvos -> validation_results.json")

with open("results_data.js", "w", encoding="utf-8") as f:
    json_content = json.dumps({
        "config": {
            "algorithm": "PMC1 MLP Backpropagation",
            "learning_rate": LEARNING_RATE,
            "epsilon": EPSILON,
            "max_epochs": MAX_EPOCHS,
            "topology": "3-12-10-1",
        },
        "trainingResults": summary,
        "validationResults": validation,
    }, ensure_ascii=False, indent=2)
    f.write(f"window.PMC1_RESULTS = {json_content};\n")
print("Dados para HTML salvos -> results_data.js")

# ============================================================
# 7. ESTILO GLOBAL DOS GRAFICOS
# ============================================================
PALETTE = ['#4FC3F7','#81C784','#FFB74D','#E57373','#CE93D8']
DARK_BG  = '#0D1117'
CARD_BG  = '#161B22'
GRID_CLR = '#30363D'
TEXT_CLR = '#E6EDF3'
ACCENT   = '#58A6FF'

def apply_dark(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    ax.spines['bottom'].set_color(GRID_CLR)
    ax.spines['left'].set_color(GRID_CLR)
    ax.spines['top'].set_color(GRID_CLR)
    ax.spines['right'].set_color(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)

# ============================================================
# FIGURA 1 - ITEM 2: Tabela dos 5 treinamentos
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')

ax.set_title("Item 1 — Resultados dos 5 Treinamentos (Backpropagation)",
             color=TEXT_CLR, fontsize=14, fontweight='bold', pad=20)

col_labels = ['Treinamento', 'Erro Quadrático Médio (EQM)', 'Número de Épocas']
table_data = [[f"T{i+1}", f"{r['eqm_final']:.6f}", f"{r['epochs']}"]
              for i, r in enumerate(results)]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#1C2128')
        cell.set_text_props(color=TEXT_CLR)
    else:
        cell.set_facecolor(CARD_BG)
        cell.set_text_props(color=TEXT_CLR)
    cell.set_edgecolor(GRID_CLR)

plt.tight_layout(pad=2.0)
plt.savefig('item1_tabela_treinamentos.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("Salvo: item1_tabela_treinamentos.png")

# ============================================================
# FIGURA 2 - ITEM 2: EQM x Epocas dos 2 treinamentos com mais epocas
# ============================================================
epochs_count = [(r['epochs'], i) for i, r in enumerate(results)]
epochs_count.sort(reverse=True)
top2_idx = sorted([epochs_count[0][1], epochs_count[1][1]])

fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Item 2 — EQM por Época: 2 Treinamentos com Maior Número de Épocas",
             color=TEXT_CLR, fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 1, hspace=0.5)

for plot_num, idx in enumerate(top2_idx):
    ax = fig.add_subplot(gs[plot_num])
    apply_dark(ax)

    hist = results[idx]['history']
    ep   = np.arange(1, len(hist)+1)
    color = PALETTE[idx]

    ax.semilogy(ep, hist, color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(ep, hist, alpha=0.15, color=color)

    # Marca ponto final
    ax.scatter([ep[-1]], [hist[-1]], color='white', s=60, zorder=5,
               edgecolors=color, linewidths=2)

    ax.set_title(f"Treinamento T{idx+1}  |  Épocas: {results[idx]['epochs']}  |  EQM Final: {results[idx]['eqm_final']:.6f}",
                 color=TEXT_CLR, fontsize=11, fontweight='bold')
    ax.set_xlabel("Época", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("EQM (escala log)", color=TEXT_CLR, fontsize=10)

    ax.annotate(f"EQM={hist[-1]:.6f}", xy=(ep[-1], hist[-1]),
                xytext=(-60, 15), textcoords='offset points',
                color='white', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

plt.savefig('item2_eqm_epocas.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("Salvo: item2_eqm_epocas.png")

# item3: resposta textual movida para README.md — sem geração de imagem

# ============================================================
# FIGURA 4 - ITEM 4: Tabela de Validacao (erro relativo e variancia)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Item 4 — Validação da Rede com Conjunto de Teste",
             color=TEXT_CLR, fontsize=13, fontweight='bold')

# Subplot 1: Tabela resumo
ax = axes[0]
ax.set_facecolor(DARK_BG)
ax.axis('off')
ax.set_title("Erro Relativo Médio e Variância por Treinamento",
             color=TEXT_CLR, fontsize=10, fontweight='bold', pad=10)

col_labels = ['Treinamento', 'Erro Relativo\nMédio (%)', 'Variância (%)']
table_data = [
    [v['treinamento'],
     f"{v['erro_relativo_medio_pct']:.4f}",
     f"{v['variancia_pct']:.4f}"]
    for v in validation
]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.4, 2.5)

best_idx = int(np.argmin([v['erro_relativo_medio_pct'] for v in validation]))
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID_CLR)
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold')
    elif row - 1 == best_idx:
        cell.set_facecolor('#1A3A1A')
        cell.set_text_props(color='#81C784', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#1C2128')
        cell.set_text_props(color=TEXT_CLR)
    else:
        cell.set_facecolor(CARD_BG)
        cell.set_text_props(color=TEXT_CLR)

ax.text(0.5, 0.02, f"★ Melhor: T{best_idx+1} (menor erro relativo médio)",
        transform=ax.transAxes, ha='center', color='#81C784',
        fontsize=9, style='italic')

# Subplot 2: Grafico de barras comparativo
ax2 = axes[1]
ax2.set_facecolor(CARD_BG)
apply_dark(ax2)
ax2.set_title("Erro Relativo Médio por Treinamento (%)",
              color=TEXT_CLR, fontsize=10, fontweight='bold')

labels = [v['treinamento'] for v in validation]
errs   = [v['erro_relativo_medio_pct'] for v in validation]
vars_  = [v['variancia_pct'] for v in validation]

colors_bar = [('#81C784' if i == best_idx else PALETTE[i]) for i in range(5)]
bars = ax2.bar(labels, errs, color=colors_bar, edgecolor=GRID_CLR,
               linewidth=1.2, width=0.55, alpha=0.9)

# Adicionar erro de variancia como linha
ax2_twin = ax2.twinx()
ax2_twin.plot(labels, vars_, 'o--', color='#FFB74D', linewidth=1.5,
              markersize=6, label='Variância (%)')
ax2_twin.set_ylabel('Variância (%)', color='#FFB74D', fontsize=9)
ax2_twin.tick_params(colors='#FFB74D', labelsize=8)
ax2_twin.spines['right'].set_color('#FFB74D')

for bar, val in zip(bars, errs):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom',
             color=TEXT_CLR, fontsize=8, fontweight='bold')

ax2.set_xlabel("Treinamento", color=TEXT_CLR, fontsize=10)
ax2.set_ylabel("Erro Relativo Médio (%)", color=TEXT_CLR, fontsize=10)

lines, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines, labels2, loc='upper right',
           facecolor=CARD_BG, edgecolor=GRID_CLR,
           labelcolor='#FFB74D', fontsize=8)

plt.tight_layout()
plt.savefig('item4_validacao.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("Salvo: item4_validacao.png")

# ============================================================
# FIGURA 5 - ITEM 5: Melhor configuracao + saidas da rede vs desejado
# ============================================================
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(f"Item 5 — Melhor Configuração: T{best_idx+1}  |  Análise de Generalização",
             color=TEXT_CLR, fontsize=13, fontweight='bold')

gs5 = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# --- Sub-grafico 1: Saidas rede vs desejado (melhor treinamento) ---
ax1 = fig.add_subplot(gs5[0, :])
apply_dark(ax1)

d_flat = d_test.flatten()
best_out = np.array(validation[best_idx]['y_rede'])
amostras = np.arange(1, N_test+1)

ax1.plot(amostras, d_flat, 'o-', color='#4FC3F7', linewidth=2,
         markersize=7, label='Saída Desejada (d)', zorder=3)
ax1.plot(amostras, best_out, 's--', color='#FFB74D', linewidth=2,
         markersize=7, label=f'Saída da Rede T{best_idx+1}', zorder=3)

ax1.fill_between(amostras, d_flat, best_out, alpha=0.15, color='#E57373')
ax1.set_xlabel("Amostra de Teste", color=TEXT_CLR)
ax1.set_ylabel("Valor de Saída", color=TEXT_CLR)
ax1.set_title(f"Comparação: Saída Desejada vs Rede T{best_idx+1} (melhor generalização)",
              color=TEXT_CLR, fontsize=10)
ax1.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
ax1.set_xticks(amostras)

# --- Sub-grafico 2: Scatter desejado vs obtido ---
ax2 = fig.add_subplot(gs5[1, 0])
apply_dark(ax2)

ax2.scatter(d_flat, best_out, color=PALETTE[best_idx], s=80, alpha=0.85,
            edgecolors='white', linewidths=0.8, zorder=3)
min_v = min(d_flat.min(), best_out.min()) - 0.02
max_v = max(d_flat.max(), best_out.max()) + 0.02
ax2.plot([min_v, max_v], [min_v, max_v], '--', color='#58A6FF',
         linewidth=1.5, label='Linha ideal (y=x)')
ax2.set_xlabel("Saída Desejada (d)", color=TEXT_CLR)
ax2.set_ylabel(f"Saída da Rede T{best_idx+1}", color=TEXT_CLR)
ax2.set_title("Dispersão: Desejado vs Obtido", color=TEXT_CLR, fontsize=10)
ax2.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

# --- Sub-grafico 3: Erro relativo por amostra (melhor treinamento) ---
ax3 = fig.add_subplot(gs5[1, 1])
apply_dark(ax3)
rel_errs = np.array(validation[best_idx]['erro_relativo_por_amostra'])
amostras_test = np.arange(1, N_test + 1)
ax3.bar(amostras_test, rel_errs, color=PALETTE[best_idx], edgecolor=GRID_CLR,
        linewidth=0.8, alpha=0.85)
ax3.axhline(y=float(np.mean(rel_errs)), color='#FF6B6B', linewidth=1.5,
            linestyle='--', label=f'Média = {np.mean(rel_errs):.2f}%')
ax3.set_xlabel("Amostra de Teste", color=TEXT_CLR)
ax3.set_ylabel("Erro Relativo (%)", color=TEXT_CLR)
ax3.set_title(f"Erro Relativo por Amostra — T{best_idx+1}", color=TEXT_CLR, fontsize=10)
ax3.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)
ax3.set_xticks(amostras_test)

plt.savefig('item5_melhor_config.png', dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("Salvo: item5_melhor_config.png")

# ============================================================
# RESUMO FINAL
# ============================================================
print("\n" + "="*55)
print("  RESUMO DOS RESULTADOS")
print("="*55)
print(f"{'Treino':<10} {'EQM Final':<18} {'Épocas':<12} {'Erro Rel.Médio %'}")
print("-"*55)
for i, (r, v) in enumerate(zip(results, validation)):
    mark = " ★" if i == best_idx else ""
    print(f"  T{i+1:<7} {r['eqm_final']:<18.6f} {r['epochs']:<12} {v['erro_relativo_medio_pct']:.4f}%{mark}")
print("="*55)
print(f"\n→ Melhor generalização: T{best_idx+1}")
print("\nArquivos gerados:")
for f in ['training_data.json','test_data.json','metadata.json',
          'training_results.json','validation_results.json',
          'results_data.js',
          'item1_tabela_treinamentos.png','item2_eqm_epocas.png',
          'item4_validacao.png','item5_melhor_config.png',
          'README.md']:
    print(f"  • {f}")
