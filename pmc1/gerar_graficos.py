"""
PMC1 - Geração de gráficos (sem texto explicativo — respostas textuais no README.md)
Todas as figuras usam apenas dados numéricos plotados com matplotlib/numpy.
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
# Carregar JSONs
# ============================================================
with open('training_data.json') as f:
    train_raw = json.load(f)
with open('test_data.json') as f:
    test_raw = json.load(f)
with open('training_results.json') as f:
    summary = json.load(f)
with open('validation_results.json') as f:
    validation = json.load(f)

X_test = np.array([[r['x1'], r['x2'], r['x3']] for r in test_raw])
d_test = np.array([[r['d']] for r in test_raw])
N_test = len(X_test)

# ============================================================
# Reconstruir históricos de treinamento
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(y):
    return y * (1.0 - y)

def forward_full(X, W1, b1, W2, b2, W3, b3):
    h1 = sigmoid(X @ W1.T + b1)
    h2 = sigmoid(h1 @ W2.T + b2)
    out = sigmoid(h2 @ W3.T + b3)
    return h1, h2, out

def eqm_fn(d, out):
    return float(np.mean(0.5 * (d - out) ** 2))

X_train = np.array([[r['x1'], r['x2'], r['x3']] for r in train_raw])
d_train = np.array([[r['d']] for r in train_raw])
N_train = len(X_train)

def train_mlp(X, d, seed, eta=0.1, epsilon=1e-6, max_epochs=50000):
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(0, 1, (12, 3)); b1 = rng.uniform(0, 1, (12,))
    W2 = rng.uniform(0, 1, (10, 12)); b2 = rng.uniform(0, 1, (10,))
    W3 = rng.uniform(0, 1, (1, 10)); b3 = rng.uniform(0, 1, (1,))
    history = []
    epoch = 0
    while epoch < max_epochs:
        h1, h2, out = forward_full(X, W1, b1, W2, b2, W3, b3)
        eq = eqm_fn(d, out)
        history.append(eq)
        if eq < epsilon:
            break
        delta3 = (d - out) * sigmoid_deriv(out)
        delta2 = (delta3 @ W3) * sigmoid_deriv(h2)
        delta1 = (delta2 @ W2) * sigmoid_deriv(h1)
        W3 += eta * (delta3.T @ h2) / N_train; b3 += eta * delta3.mean(axis=0)
        W2 += eta * (delta2.T @ h1) / N_train; b2 += eta * delta2.mean(axis=0)
        W1 += eta * (delta1.T @ X) / N_train;  b1 += eta * delta1.mean(axis=0)
        epoch += 1
    return {'history': history, 'epochs': epoch, 'eqm_final': history[-1]}

seeds = [42, 7, 2024, 314, 99]
print("Reconstruindo históricos de treinamento...")
histories = []
for i, seed in enumerate(seeds):
    print(f"  T{i+1}...", end=' ', flush=True)
    r = train_mlp(X_train, d_train, seed=seed)
    histories.append(r)
    print(f"EQM={r['eqm_final']:.6f}  Épocas={r['epochs']}")

# ============================================================
# Estilos
# ============================================================
PALETTE  = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#CE93D8']
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
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.7)

# ============================================================
# FIGURA 1 — Item 1: Tabela de resultados dos 5 treinamentos
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')
ax.set_title("Item 1 — Resultados dos 5 Treinamentos",
             color=TEXT_CLR, fontsize=14, fontweight='bold', pad=20)

col_labels = ['Treinamento', 'Erro Quadrático Médio (EQM)', 'Número de Épocas']
table_data = [[s['treinamento'], f"{s['eqm_final']:.6f}", str(s['num_epocas'])]
              for s in summary]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID_CLR)
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#1C2128')
        cell.set_text_props(color=TEXT_CLR)
    else:
        cell.set_facecolor(CARD_BG)
        cell.set_text_props(color=TEXT_CLR)

plt.tight_layout(pad=2.0)
plt.savefig('item1_tabela_treinamentos.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Salvo: item1_tabela_treinamentos.png")

# ============================================================
# FIGURA 2 — Item 2: EQM × Época (2 treinamentos com mais épocas)
# ============================================================
epochs_count = [(h['epochs'], i) for i, h in enumerate(histories)]
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
    hist  = histories[idx]['history']
    ep    = np.arange(1, len(hist) + 1)
    color = PALETTE[idx]

    ax.semilogy(ep, hist, color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(ep, hist, alpha=0.15, color=color)
    ax.scatter([ep[-1]], [hist[-1]], color='white', s=60, zorder=5,
               edgecolors=color, linewidths=2)

    ax.set_title(
        f"T{idx+1}  —  Épocas: {histories[idx]['epochs']}  |  EQM Final: {histories[idx]['eqm_final']:.6f}",
        color=TEXT_CLR, fontsize=11, fontweight='bold')
    ax.set_xlabel("Época", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("EQM (escala log)", color=TEXT_CLR, fontsize=10)

plt.savefig('item2_eqm_epocas.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Salvo: item2_eqm_epocas.png")

# ============================================================
# FIGURA 3 — Item 4: Erro relativo médio e variância (validação)
# ============================================================
best_idx = int(np.argmin([v['erro_relativo_medio_pct'] for v in validation]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Item 4 — Validação da Rede com Conjunto de Teste",
             color=TEXT_CLR, fontsize=13, fontweight='bold')

# --- Subplot esquerdo: tabela de resultados ---
ax = axes[0]
ax.set_facecolor(DARK_BG)
ax.axis('off')
ax.set_title("Erro Relativo Médio e Variância por Treinamento",
             color=TEXT_CLR, fontsize=10, fontweight='bold', pad=10)

tbl_data = [[v['treinamento'],
             f"{v['erro_relativo_medio_pct']:.4f} %",
             f"{v['variancia_pct']:.4f} %"] for v in validation]
tbl = ax.table(cellText=tbl_data,
               colLabels=['Treinamento', 'Erro Rel. Médio (%)', 'Variância (%)'],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.4, 2.5)

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

# --- Subplot direito: barras de erro relativo ---
ax2 = axes[1]
apply_dark(ax2)
ax2.set_title("Erro Relativo Médio (%) e Variância por Treinamento",
              color=TEXT_CLR, fontsize=10, fontweight='bold')

labels    = [v['treinamento'] for v in validation]
errs      = [v['erro_relativo_medio_pct'] for v in validation]
vars_     = [v['variancia_pct'] for v in validation]
colors_bar = [('#81C784' if i == best_idx else PALETTE[i]) for i in range(5)]

bars = ax2.bar(labels, errs, color=colors_bar, edgecolor=GRID_CLR,
               linewidth=1.2, width=0.55, alpha=0.9)

ax2_twin = ax2.twinx()
ax2_twin.plot(labels, vars_, 'o--', color='#FFB74D', linewidth=1.5,
              markersize=6, label='Variância (%)')
ax2_twin.set_ylabel('Variância (%)', color='#FFB74D', fontsize=9)
ax2_twin.tick_params(colors='#FFB74D', labelsize=8)
for spine_name, spine in ax2_twin.spines.items():
    spine.set_color('#FFB74D' if spine_name == 'right' else GRID_CLR)

for bar, val in zip(bars, errs):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom',
             color=TEXT_CLR, fontsize=8, fontweight='bold')

ax2.set_xlabel("Treinamento", color=TEXT_CLR, fontsize=10)
ax2.set_ylabel("Erro Relativo Médio (%)", color=TEXT_CLR, fontsize=10)
lines, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines, labels2, loc='upper right',
           facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor='#FFB74D', fontsize=8)

plt.tight_layout()
plt.savefig('item4_validacao.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Salvo: item4_validacao.png")

# ============================================================
# FIGURA 4 — Item 5: Comparação saída desejada vs melhor rede
# ============================================================
d_flat   = d_test.flatten()
best_out = np.array(validation[best_idx]['y_rede'])
amostras = np.arange(1, N_test + 1)

fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(f"Item 5 — Melhor Configuração: T{best_idx+1}  |  Análise de Generalização",
             color=TEXT_CLR, fontsize=13, fontweight='bold')

gs5 = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# --- Linha superior: Saída desejada vs rede ---
ax1 = fig.add_subplot(gs5[0, :])
apply_dark(ax1)
ax1.plot(amostras, d_flat, 'o-', color='#4FC3F7', linewidth=2,
         markersize=7, label='Saída Desejada (d)', zorder=3)
ax1.plot(amostras, best_out, 's--', color='#FFB74D', linewidth=2,
         markersize=7, label=f'Saída da Rede T{best_idx+1}', zorder=3)
ax1.fill_between(amostras, d_flat, best_out, alpha=0.15, color='#E57373')
ax1.set_xlabel("Amostra de Teste", color=TEXT_CLR)
ax1.set_ylabel("Valor de Saída", color=TEXT_CLR)
ax1.set_title(f"Comparação: Saída Desejada vs T{best_idx+1}",
              color=TEXT_CLR, fontsize=11)
ax1.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
ax1.set_xticks(amostras)

# --- Esquerda inferior: Dispersão desejado vs obtido ---
ax2 = fig.add_subplot(gs5[1, 0])
apply_dark(ax2)
ax2.scatter(d_flat, best_out, color=PALETTE[best_idx], s=80, alpha=0.85,
            edgecolors='white', linewidths=0.8, zorder=3)
min_v = min(d_flat.min(), best_out.min()) - 0.02
max_v = max(d_flat.max(), best_out.max()) + 0.02
ax2.plot([min_v, max_v], [min_v, max_v], '--', color='#58A6FF',
         linewidth=1.5, label='Ideal (y = x)')
ax2.set_xlabel("Saída Desejada (d)", color=TEXT_CLR)
ax2.set_ylabel(f"Saída da Rede T{best_idx+1}", color=TEXT_CLR)
ax2.set_title("Dispersão: Desejado vs Obtido", color=TEXT_CLR, fontsize=10)
ax2.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)

# --- Direita inferior: Erro relativo por amostra (melhor treino) ---
ax3 = fig.add_subplot(gs5[1, 1])
apply_dark(ax3)
rel_errs = np.array(validation[best_idx]['erro_relativo_por_amostra'])
ax3.bar(amostras, rel_errs, color=PALETTE[best_idx], edgecolor=GRID_CLR,
        linewidth=0.8, alpha=0.85)
ax3.axhline(y=float(np.mean(rel_errs)), color='#FF6B6B', linewidth=1.5,
            linestyle='--', label=f'Média = {np.mean(rel_errs):.2f}%')
ax3.set_xlabel("Amostra de Teste", color=TEXT_CLR)
ax3.set_ylabel("Erro Relativo (%)", color=TEXT_CLR)
ax3.set_title(f"Erro Relativo por Amostra — T{best_idx+1}", color=TEXT_CLR, fontsize=10)
ax3.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)
ax3.set_xticks(amostras)

plt.savefig('item5_melhor_config.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Salvo: item5_melhor_config.png")

# ============================================================
# Resumo final no terminal
# ============================================================
print("\n" + "=" * 60)
print("  RESUMO DOS RESULTADOS")
print("=" * 60)
print(f"{'Treino':<10} {'EQM Final':<18} {'Épocas':<12} {'Erro Rel. Médio %'}")
print("-" * 60)
for i, (s, v) in enumerate(zip(summary, validation)):
    mark = " ★" if i == best_idx else ""
    print(f"  T{i+1:<7} {s['eqm_final']:<18.6f} {s['num_epocas']:<12} "
          f"{v['erro_relativo_medio_pct']:.4f}%{mark}")
print("=" * 60)
print(f"\n→ Melhor generalização: T{best_idx+1}")
print("→ Respostas textuais (Itens 3 e 5) disponíveis em README.md")
print("\nImagens geradas:")
for f in ['item1_tabela_treinamentos.png', 'item2_eqm_epocas.png',
          'item4_validacao.png', 'item5_melhor_config.png']:
    print(f"  • {f}")
