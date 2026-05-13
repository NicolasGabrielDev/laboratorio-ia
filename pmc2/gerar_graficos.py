"""
gerar_graficos.py
Gera todos os gráficos do projeto PMC2 a partir dos arquivos JSON.
Itens gerados:
  item1_tabela_treinamentos.png  — tabela resumo dos dois treinamentos
  item2_eqm_epocas.png           — EQM x Épocas (padrão e momentum)
  item3_explicacao.png           — explicação do pós-processamento
  item4_validacao.png            — resultados de validação no teste
  item5_melhor_config.png        — comparativo / melhor configuração
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sem display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# Carrega JSONs necessários
# ─────────────────────────────────────────────
with open("training_results.json") as f:
    tr = json.load(f)

with open("validation_results.json") as f:
    vr = json.load(f)

with open("metadata.json") as f:
    meta = json.load(f)

eqm_std = np.array(tr["standard"]["eqm_per_epoch"])
eqm_mom = np.array(tr["momentum"]["eqm_per_epoch"])

# ══════════════════════════════════════════════
# ITEM 1 — Tabela resumo dos treinamentos
# ══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")
ax.set_title("Item 1 — Tabela Resumo dos Treinamentos", fontsize=13,
             fontweight="bold", pad=14)

col_labels = ["Método", "Épocas", "EQM Final", "Tempo (s)", "Taxa de Acerto (%)"]
row_data = [
    ["Backpropagation Padrão",
     f"{tr['standard']['epochs']}",
     f"{tr['standard']['final_eqm']:.2e}",
     f"{tr['standard']['time_seconds']:.4f}",
     f"{vr['standard']['accuracy_pct']:.1f}%"],
    ["Backpropagation + Momentum",
     f"{tr['momentum']['epochs']}",
     f"{tr['momentum']['final_eqm']:.2e}",
     f"{tr['momentum']['time_seconds']:.4f}",
     f"{vr['momentum']['accuracy_pct']:.1f}%"],
]

# Parâmetros da rede abaixo da tabela
param_text = (f"Arquitetura: {meta['arquitetura_rede']['entradas']} entradas → "
              f"{meta['arquitetura_rede']['camadas_ocultas'][0]} neurônios ocultos → "
              f"{meta['arquitetura_rede']['saidas']} saídas  |  "
              f"η = {meta['hiperparametros']['taxa_aprendizado']}  |  "
              f"β = {meta['hiperparametros']['momentum']}  |  "
              f"ε = {meta['hiperparametros']['precisao']}")

tbl = ax.table(cellText=row_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)

# Destaca cabeçalho
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c7bb6")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Cores alternadas nas linhas
colors = ["#d9edf7", "#ffffff"]
for i in range(1, 3):
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(colors[i - 1])

ax.text(0.5, 0.05, param_text, ha="center", va="bottom",
        transform=ax.transAxes, fontsize=8, color="#555555",
        style="italic")

plt.tight_layout()
plt.savefig("item1_tabela_treinamentos.png", dpi=150, bbox_inches="tight")
plt.close()
print("item1_tabela_treinamentos.png gerado")

# ══════════════════════════════════════════════
# ITEM 2 — EQM x Épocas (dois gráficos na mesma folha, não sobrepostos)
# ══════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Item 2 — EQM por Época de Treinamento", fontsize=13,
             fontweight="bold")

epocas_std = np.arange(1, len(eqm_std) + 1)
epocas_mom = np.arange(1, len(eqm_mom) + 1)

# ── Subplot 1: Backpropagation Padrão ─────────
ax1.plot(epocas_std, eqm_std, color="#e74c3c", linewidth=0.8, label="BP Padrão")
ax1.set_xlabel("Época", fontsize=10)
ax1.set_ylabel("EQM", fontsize=10)
ax1.set_title(f"Backpropagation Padrão  "
              f"(épocas={len(eqm_std)}, EQM final={eqm_std[-1]:.2e})",
              fontsize=10)
ax1.set_yscale("log")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=9)

# ── Subplot 2: Backpropagation + Momentum ─────
ax2.plot(epocas_mom, eqm_mom, color="#2980b9", linewidth=0.8, label="BP + Momentum")
ax2.set_xlabel("Época", fontsize=10)
ax2.set_ylabel("EQM", fontsize=10)
ax2.set_title(f"Backpropagation com Momentum  "
              f"(épocas={len(eqm_mom)}, EQM final={eqm_mom[-1]:.2e})",
              fontsize=10)
ax2.set_yscale("log")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("item2_eqm_epocas.png", dpi=150, bbox_inches="tight")
plt.close()
print("item2_eqm_epocas.png gerado")

# ══════════════════════════════════════════════
# ITEM 3 — Explicação do pós-processamento
# ══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")
ax.set_title("Item 3 — Pós-processamento das Saídas (Arredondamento Simétrico)",
             fontsize=13, fontweight="bold", pad=14)

# Ilustra a regra com exemplos reais do conjunto de teste
examples = vr["standard"]["details"][:6]

col_labels = ["Amostra", "y1 (real)", "y2 (real)", "y3 (real)",
              "y1 (int)", "y2 (int)", "y3 (int)", "Classe Prevista"]

rows = []
for ex in examples:
    raw = ex["raw_output"]
    pred = ex["predicted"]
    cls_map = {(1,0,0):"A", (0,1,0):"B", (0,0,1):"C"}
    cls = cls_map.get(tuple(pred), "?")
    rows.append([f"#{ex['sample']}",
                 f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{raw[2]:.4f}",
                 str(pred[0]), str(pred[1]), str(pred[2]), cls])

tbl = ax.table(cellText=rows, colLabels=col_labels,
               loc="upper center", cellLoc="center",
               bbox=[0.0, 0.35, 1.0, 0.55])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#27ae60")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Texto explicativo da regra
rule_text = (
    "Regra de Arredondamento Simétrico:\n\n"
    "   se  yₖ ≥ 0.5  →  yₖ = 1\n"
    "   se  yₖ  < 0.5  →  yₖ = 0\n\n"
    "Após o pós-processamento, a saída com valor 1 indica o conservante:\n"
    "   y1=1 → Tipo A  |  y2=1 → Tipo B  |  y3=1 → Tipo C"
)
ax.text(0.5, 0.20, rule_text, ha="center", va="center",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf6ff",
                  edgecolor="#2980b9", linewidth=1.5),
        family="monospace")

ax.text(0.5, -0.02,
        "* Tabela mostra as 6 primeiras amostras do conjunto de teste (método padrão)",
        ha="center", va="bottom", transform=ax.transAxes,
        fontsize=8, color="#777777", style="italic")

plt.tight_layout()
plt.savefig("item3_explicacao.png", dpi=150, bbox_inches="tight")
plt.close()
print("item3_explicacao.png gerado")

# ══════════════════════════════════════════════
# ITEM 4 — Validação: resultados no conjunto de teste
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(14, 7))
fig.suptitle("Item 4 — Validação no Conjunto de Teste", fontsize=13,
             fontweight="bold")

gs = GridSpec(1, 2, figure=fig, wspace=0.35)

def plot_validation_table(ax, details, title, acc):
    ax.axis("off")
    ax.set_title(f"{title}\nTaxa de Acerto: {acc:.1f}%", fontsize=10,
                 fontweight="bold")
    col_labels = ["Amostra", "Desejado", "Previsto", "✓"]
    cls_map = {(1,0,0):"A", (0,1,0):"B", (0,0,1):"C"}
    rows = []
    for ex in details:
        d = tuple(ex["desired"])
        p = tuple(ex["predicted"])
        rows.append([f"#{ex['sample']}",
                     cls_map.get(d, "?"),
                     cls_map.get(p, "?"),
                     "✓" if ex["correct"] else "✗"])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for j in range(4):
        tbl[0, j].set_facecolor("#34495e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, ex in enumerate(details, 1):
        color = "#d5f5e3" if ex["correct"] else "#fadbd8"
        for j in range(4):
            tbl[i, j].set_facecolor(color)

ax_left  = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])

plot_validation_table(ax_left,  vr["standard"]["details"],
                      "Backpropagation Padrão",   vr["standard"]["accuracy_pct"])
plot_validation_table(ax_right, vr["momentum"]["details"],
                      "Backpropagation + Momentum", vr["momentum"]["accuracy_pct"])

plt.savefig("item4_validacao.png", dpi=150, bbox_inches="tight")
plt.close()
print("item4_validacao.png gerado")

# ══════════════════════════════════════════════
# ITEM 5 — Comparativo / Melhor Configuração
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Item 5 — Comparativo: Melhor Configuração", fontsize=13,
             fontweight="bold")

# ── Gráfico 1: EQM comparativo na mesma escala ─
ax = axes[0]
ax.plot(np.arange(1, len(eqm_std)+1), eqm_std,
        color="#e74c3c", linewidth=1.0, alpha=0.85, label="BP Padrão")
ax.plot(np.arange(1, len(eqm_mom)+1), eqm_mom,
        color="#2980b9", linewidth=1.0, alpha=0.85, label="BP + Momentum")
ax.set_xlabel("Época", fontsize=10)
ax.set_ylabel("EQM (log)", fontsize=10)
ax.set_title("EQM — Comparativo", fontsize=10)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=9)

# ── Gráfico 2: Barras de comparação ────────────
ax2 = axes[1]
metrics   = ["Épocas", "Tempo (s)", "Acerto (%)"]
val_std   = [tr["standard"]["epochs"],
             tr["standard"]["time_seconds"],
             vr["standard"]["accuracy_pct"]]
val_mom   = [tr["momentum"]["epochs"],
             tr["momentum"]["time_seconds"],
             vr["momentum"]["accuracy_pct"]]

# Normaliza para exibição comparativa (eixo único)
# Mostra como barras side-by-side com escala independente por métrica
x   = np.arange(len(metrics))
w   = 0.35
b1  = ax2.bar(x - w/2, val_std, w, color="#e74c3c", alpha=0.8, label="BP Padrão")
b2  = ax2.bar(x + w/2, val_mom, w, color="#2980b9", alpha=0.8, label="BP + Momentum")

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_title("Comparativo de Métricas", fontsize=10)
ax2.legend(fontsize=9)
ax2.set_yscale("log")
ax2.set_ylabel("Valor (escala log)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

# Rótulos nas barras
for bar in list(b1) + list(b2):
    h = bar.get_height()
    label = f"{h:.1f}" if h < 1000 else f"{int(h)}"
    ax2.text(bar.get_x() + bar.get_width()/2., h * 1.05,
             label, ha="center", va="bottom", fontsize=8)

# Destaca o vencedor
acc_std_v = vr["standard"]["accuracy_pct"]
acc_mom_v = vr["momentum"]["accuracy_pct"]
winner = "Momentum" if tr["momentum"]["epochs"] < tr["standard"]["epochs"] else "Padrão"
note = (f"✦ Método com momentum convergiu em {tr['momentum']['epochs']} épocas\n"
        f"  vs {tr['standard']['epochs']} épocas do método padrão.\n"
        f"  Acerto: Padrão={acc_std_v:.1f}%  Momentum={acc_mom_v:.1f}%")
fig.text(0.5, -0.04, note, ha="center", fontsize=9, color="#333333",
         style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7",
                   edgecolor="#f39c12", linewidth=1.2))

plt.tight_layout()
plt.savefig("item5_melhor_config.png", dpi=150, bbox_inches="tight")
plt.close()
print("item5_melhor_config.png gerado")

print("\nTodos os gráficos gerados com sucesso.")