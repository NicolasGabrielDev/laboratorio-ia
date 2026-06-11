import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# ─── DADOS ────────────────────────────────────────────────────────────────────

treinamento = [
    [-0.6508, 0.1097, 4.0009, -1.0],
    [-1.4492, 0.8896, 4.4005, -1.0],
    [2.0850,  0.6876, 12.0710, -1.0],
    [0.2626,  1.1476, 7.7985,   1.0],
    [0.6418,  1.0234, 7.0427,   1.0],
    [0.2569,  0.6730, 8.3265,  -1.0],
    [1.1155,  0.6043, 7.4446,   1.0],
    [0.0914,  0.3399, 7.0677,  -1.0],
    [0.0121,  0.5256, 4.6316,   1.0],
    [-0.0429, 0.4660, 5.4323,   1.0],
    [0.4340,  0.6870, 8.2287,  -1.0],
    [0.2735,  1.0287, 7.1934,   1.0],
    [0.4839,  0.4851, 7.4850,  -1.0],
    [0.4089, -0.1267, 5.5019,  -1.0],
    [1.4391,  0.1614, 8.5843,  -1.0],
    [-0.9115,-0.1973, 2.1962,  -1.0],
    [0.3654,  1.0475, 7.4858,   1.0],
    [0.2144,  0.7515, 7.1699,   1.0],
    [0.2013,  1.0014, 6.5489,   1.0],
    [0.6483,  0.2183, 5.8991,   1.0],
    [-0.1147, 0.2242, 7.2435,  -1.0],
    [-0.7970, 0.8795, 3.8762,   1.0],
    [-1.0625, 0.6366, 2.4707,   1.0],
    [0.5307,  0.1285, 5.6883,   1.0],
    [-1.2200, 0.7777, 1.7252,   1.0],
    [0.3957,  0.1076, 5.6623,  -1.0],
    [-0.1013, 0.5989, 7.1812,  -1.0],
    [2.4482,  0.9455, 11.2095,  1.0],
    [2.0149,  0.6192, 10.9263, -1.0],
    [0.2012,  0.2611, 5.4631,   1.0],
]

teste = [
    [-0.3565, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012,  0.5611, 5.8234],
    [0.7757,  1.0648, 8.0677],
    [0.1570,  0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748,  0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842,-0.2805, 1.2548],
]

# ─── PERCEPTRON ───────────────────────────────────────────────────────────────

def ativacao(v):
    return 1.0 if v >= 0 else -1.0

def treinar(pesos, dados, eta=0.01):
    epocas = 0
    erro = True
    while erro:
        erro = False
        for p in dados:
            x = [-1.0, p[0], p[1], p[2]]
            d = p[3]
            v = sum(x[i] * pesos[i] for i in range(4))
            y = ativacao(v)
            if y != d:
                for i in range(4):
                    pesos[i] += eta * (d - y) * x[i] / 2.0
                erro = True
        epocas += 1
    return epocas

def classificar(pesos, x_in):
    x = [-1.0, x_in[0], x_in[1], x_in[2]]
    return ativacao(sum(x[i] * pesos[i] for i in range(4)))

# ─── EXECUTAR 5 TREINAMENTOS ──────────────────────────────────────────────────

seeds = [42, 7, 123, 999, 2024]
resultados = []
pesos_finais = []

for i, seed in enumerate(seeds):
    random.seed(seed)
    pesos = [random.uniform(0, 1) for _ in range(4)]
    wi = pesos.copy()
    ep = treinar(pesos, treinamento)
    resultados.append({'wi': wi, 'wf': pesos.copy(), 'epocas': ep})
    pesos_finais.append(pesos)

# ─── CLASSIFICAR AMOSTRAS DE TESTE ───────────────────────────────────────────

classificacoes = [[classificar(pesos_finais[t], am) for t in range(5)] for am in teste]

training_data = [
    {
        "sample": index + 1,
        "features": sample[:3],
        "class": sample[3],
    }
    for index, sample in enumerate(treinamento)
]
test_data = [
    {
        "sample": index + 1,
        "features": sample,
    }
    for index, sample in enumerate(teste)
]
training_results = [
    {
        "training": f"T{index + 1}",
        "initial_weights": result["wi"],
        "final_weights": result["wf"],
        "epochs": result["epocas"],
    }
    for index, result in enumerate(resultados)
]
classification_results = [
    {
        "sample": index + 1,
        "features": sample,
        "predictions": {
            f"T{training_index + 1}": int(prediction)
            for training_index, prediction in enumerate(predictions)
        },
    }
    for index, (sample, predictions) in enumerate(zip(teste, classificacoes))
]

with open("training_data.json", "w", encoding="utf-8") as file:
    json.dump(training_data, file, ensure_ascii=False, indent=2)

with open("test_data.json", "w", encoding="utf-8") as file:
    json.dump(test_data, file, ensure_ascii=False, indent=2)

with open("training_results.json", "w", encoding="utf-8") as file:
    json.dump(training_results, file, ensure_ascii=False, indent=2)

with open("classification_results.json", "w", encoding="utf-8") as file:
    json.dump(classification_results, file, ensure_ascii=False, indent=2)

with open("results_data.js", "w", encoding="utf-8") as file:
    json_content = json.dumps({
        "trainingData": training_data,
        "testData": test_data,
        "trainingResults": training_results,
        "classificationResults": classification_results,
    }, ensure_ascii=False, indent=2)
    file.write(f"window.PERCEPTRON_RESULTS = {json_content};\n")

# ─── QUESTÕES 1 & 2: Tabela de treinamento ────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis('off')
ax.set_title('Questões 1 & 2 — Resultados dos Treinamentos (η = 0.01)', fontsize=13, pad=12)

cols = ['Trein.', 'w0 ini', 'w1 ini', 'w2 ini', 'w3 ini',
                  'w0 fin', 'w1 fin', 'w2 fin', 'w3 fin', 'Épocas']
rows = []
for i, r in enumerate(resultados):
    wi, wf = r['wi'], r['wf']
    rows.append([f"T{i+1}",
                 f"{wi[0]:.4f}", f"{wi[1]:.4f}", f"{wi[2]:.4f}", f"{wi[3]:.4f}",
                 f"{wf[0]:.4f}", f"{wf[1]:.4f}", f"{wf[2]:.4f}", f"{wf[3]:.4f}",
                 str(r['epocas'])])

table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

plt.tight_layout()
plt.savefig('q1_q2_treinamento.png', dpi=130, bbox_inches='tight')
plt.close()

# ─── QUESTÃO 3: Tabela de classificação ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
ax.set_title('Questão 3 — Classificação das Amostras de Teste', fontsize=13, pad=12)

cols3 = ['Amostra', 'x1', 'x2', 'x3', 'y(T1)', 'y(T2)', 'y(T3)', 'y(T4)', 'y(T5)']
rows3 = []
for i, (am, cls) in enumerate(zip(teste, classificacoes)):
    rows3.append([f"A{i+1}",
                  f"{am[0]:.4f}", f"{am[1]:.4f}", f"{am[2]:.4f}",
                  f"{int(cls[0])}", f"{int(cls[1])}", f"{int(cls[2])}",
                  f"{int(cls[3])}", f"{int(cls[4])}"])

table3 = ax.table(cellText=rows3, colLabels=cols3, loc='center', cellLoc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(9)
table3.scale(1, 1.6)

plt.tight_layout()
plt.savefig('q3_classificacao.png', dpi=130, bbox_inches='tight')
plt.close()

# ─── QUESTÃO 4: Gráfico de épocas + texto explicativo ────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Questão 4 — Por que o número de épocas varia?', fontsize=13)

epocas = [r['epocas'] for r in resultados]
ax1.bar([f"T{i+1}" for i in range(5)], epocas, color='steelblue')
ax1.axhline(np.mean(epocas), color='red', linestyle='--', label=f'Média: {np.mean(epocas):.0f}')
ax1.set_xlabel('Treinamento')
ax1.set_ylabel('Épocas')
ax1.set_title('Épocas por treinamento')
ax1.legend()
for i, ep in enumerate(epocas):
    ax1.text(i, ep + 2, str(ep), ha='center', fontsize=9)

ax2.axis('off')
texto = (
    "Por que as épocas variam?\n\n"
    "1. Pesos iniciais aleatórios\n"
    "   A cada treinamento, os pesos w0..w3\n"
    "   são sorteados aleatoriamente em [0,1],\n"
    "   gerando um hiperplano de partida diferente.\n\n"
    "2. Distância até a solução\n"
    "   Pesos mais distantes do hiperplano\n"
    "   separador ideal exigem mais correções\n"
    "   e, portanto, mais épocas.\n\n"
    "3. Caminho no espaço de pesos\n"
    "   Cada ponto de partida define um caminho\n"
    "   diferente de ajuste até convergir.\n\n"
    "Conclusão: todos convergem para um\n"
    "hiperplano correto, mas em velocidades\n"
    "diferentes dependendo do início."
)
ax2.text(0.05, 0.95, texto, transform=ax2.transAxes,
         fontsize=10, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))

plt.tight_layout()
plt.savefig('q4_epocas.png', dpi=130, bbox_inches='tight')
plt.close()

# ─── QUESTÃO 5: Limitação do perceptron ──────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Questão 5 — Principal Limitação do Perceptron', fontsize=13)

# Separável linearmente
np.random.seed(1)
c1 = np.random.randn(20, 2) * 0.5 + [-1.5, -1.5]
c2 = np.random.randn(20, 2) * 0.5 + [1.5, 1.5]
ax1.scatter(c1[:,0], c1[:,1], label='C1 (−1)', marker='x', color='blue')
ax1.scatter(c2[:,0], c2[:,1], label='C2 (+1)', marker='o', color='orange')
x_line = np.linspace(-3, 3, 100)
ax1.plot(x_line, -x_line, 'k--', label='Hiperplano separador')
ax1.set_title('Linearmente separável\n✓ Perceptron converge')
ax1.legend(fontsize=8)
ax1.set_xlabel('x1'); ax1.set_ylabel('x2')
ax1.grid(True, alpha=0.3)

# Não separável (XOR)
xor_c1 = np.array([[-1,-1],[1,1]])
xor_c2 = np.array([[-1,1],[1,-1]])
ax2.scatter(xor_c1[:,0], xor_c1[:,1], s=150, marker='x', color='blue', label='C1 (−1)')
ax2.scatter(xor_c2[:,0], xor_c2[:,1], s=150, marker='o', color='orange', label='C2 (+1)')
ax2.set_title('Não linearmente separável (XOR)\n✗ Perceptron NÃO converge')
ax2.legend(fontsize=8)
ax2.set_xlabel('x1'); ax2.set_ylabel('x2')
ax2.grid(True, alpha=0.3)
ax2.text(0, 0, 'Nenhuma reta\nsepara as classes', ha='center', va='center',
         fontsize=10, bbox=dict(facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout()
plt.savefig('q5_limitacao.png', dpi=130, bbox_inches='tight')
plt.close()

print("Arquivos gerados: training_data.json, test_data.json, training_results.json, classification_results.json, results_data.js")
print("Imagens geradas: q1_q2_treinamento.png, q3_classificacao.png, q4_epocas.png, q5_limitacao.png")
