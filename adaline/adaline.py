import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# ─── DADOS ────────────────────────────────────────────────────────────────────

treinamento = np.array([
    [0.4329, -1.3719, 0.7022, -0.8535,  1.0],
    [0.3024,  0.2286, 0.8630,  2.7909, -1.0],
    [0.1349, -0.6445, 1.0530,  0.5687, -1.0],
    [0.3374, -1.7163, 0.3670, -0.6283, -1.0],
    [1.1434, -0.0485, 0.6637,  1.2606,  1.0],
    [1.3749, -0.5071, 0.4464,  1.3009,  1.0],
    [0.7221, -0.7587, 0.7681, -0.5592,  1.0],
    [0.4403, -0.8072, 0.5154, -0.3129,  1.0],
    [-0.5231, 0.3548, 0.2538,  1.5776, -1.0],
    [0.3255, -2.0000, 0.7112, -1.1209,  1.0],
    [0.5824,  1.3915,-0.2291,  4.1735, -1.0],
    [0.1340,  0.6081, 0.4450,  3.2230, -1.0],
    [0.1480, -0.2988, 0.4778,  0.8649,  1.0],
    [0.7359,  0.1869,-0.0872,  2.3584,  1.0],
    [0.7115, -1.1469, 0.3394,  0.9573, -1.0],
    [0.8251, -1.2840, 0.8452,  1.2382, -1.0],
    [0.1569,  0.3712, 0.8825,  1.7633,  1.0],
    [0.0033,  0.6835, 0.5389,  2.8249, -1.0],
    [0.4243,  0.8313, 0.2634,  3.5855, -1.0],
    [1.0490,  0.1326, 0.9138,  1.9792,  1.0],
    [1.4276,  0.5331,-0.0145,  3.7286,  1.0],
    [0.5971,  1.4865, 0.2904,  4.6069, -1.0],
    [0.8475,  2.1479, 0.3179,  5.8235, -1.0],
    [1.3967, -0.4171, 0.6443,  1.3927,  1.0],
    [0.0044,  1.5378, 0.6099,  4.7755, -1.0],
    [0.2201, -0.5668, 0.0515,  0.7829,  1.0],
    [0.6300, -1.2480, 0.8591,  0.8093, -1.0],
    [-0.2479, 0.8960, 0.0547,  1.7381,  1.0],
    [-0.3088,-0.0929, 0.8659,  1.5483, -1.0],
    [-0.5180, 1.4974, 0.5453,  2.3993,  1.0],
    [0.6833,  0.8266, 0.0829,  2.8864,  1.0],
    [0.4353, -1.4066, 0.4207, -0.4879,  1.0],
    [-0.1069,-3.2329, 0.1856, -2.4572, -1.0],
    [0.4662,  0.6261, 0.7304,  3.4370, -1.0],
    [0.8298, -1.4089, 0.3119,  1.3235, -1.0],
])

classificacao = np.array([
    [0.9694,  0.6909, 0.4334, 3.4965],
    [0.5427,  1.3832, 0.6390, 4.0352],
    [0.6081, -0.9196, 0.5925, 0.1016],
    [-0.1618, 0.4694, 0.2030, 3.0117],
    [0.1870, -0.2578, 0.6124, 1.7749],
    [0.4891, -0.5276, 0.4378, 0.6439],
    [0.3777,  2.0149, 0.7423, 3.3932],
    [1.1498, -0.4067, 0.2469, 1.5866],
    [0.9325,  1.0950, 1.0359, 3.3591],
    [0.5060,  1.3317, 0.9222, 3.7174],
    [0.0497, -2.0656, 0.6124,-0.6585],
    [0.4004,  3.5369, 0.9766, 5.3532],
    [-0.1874, 1.3343, 0.5374, 3.2189],
    [0.5060,  1.3317, 0.9222, 3.7174],
    [1.6375, -0.7911, 0.7537, 0.5515],
])

# Adicionando x0 = -1
X_train = np.hstack([-np.ones((len(treinamento), 1)), treinamento[:, :4]])
d_train = treinamento[:, 4]

X_class = np.hstack([-np.ones((len(classificacao), 1)), classificacao])

# ─── ADALINE ──────────────────────────────────────────────────────────────────

def eqm(w, X, d):
    return np.mean((d - X @ w) ** 2)

def treinar(eta=0.0025, precisao=1e-6):
    w = np.random.rand(5)
    wi = w.copy()
    eqm_hist = [eqm(w, X_train, d_train)]
    epocas = 0

    while True:
        epocas += 1
        for i in range(len(X_train)):
            u = w @ X_train[i]
            w += eta * (d_train[i] - u) * X_train[i]

        eqm_atual = eqm(w, X_train, d_train)
        eqm_hist.append(eqm_atual)

        if abs(eqm_hist[-1] - eqm_hist[-2]) <= precisao:
            break

    return wi, w, epocas, eqm_hist

# ─── 5 TREINAMENTOS ───────────────────────────────────────────────────────────

np.random.seed(None)
resultados = []
historicos = []

for t in range(5):
    wi, wf, ep, hist = treinar()
    resultados.append({'wi': wi, 'wf': wf, 'epocas': ep})
    historicos.append(hist)

training_data = [
    {
        "sample": index + 1,
        "features": row[:4].tolist(),
        "class": float(row[4]),
    }
    for index, row in enumerate(treinamento)
]
test_data = [
    {
        "sample": index + 1,
        "features": row.tolist(),
    }
    for index, row in enumerate(classificacao)
]
training_results = [
    {
        "training": f"T{index + 1}",
        "initial_weights": result["wi"].tolist(),
        "final_weights": result["wf"].tolist(),
        "epochs": result["epocas"],
        "eqm_history": [float(value) for value in historicos[index]],
        "final_eqm": float(historicos[index][-1]),
    }
    for index, result in enumerate(resultados)
]

# ─── Q1: Tabela de treinamentos ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(15, 3.5))
ax.axis('off')
ax.set_title('Questão 1 — Resultados dos Treinamentos ADALINE (η = 0.0025)', fontsize=13, pad=12)

cols = ['Trein.',
        'w0 ini', 'w1 ini', 'w2 ini', 'w3 ini', 'w4 ini',
        'w0 fin', 'w1 fin', 'w2 fin', 'w3 fin', 'w4 fin',
        'Épocas']
rows = []
for i, r in enumerate(resultados):
    rows.append([f"T{i+1}"]
                + [f"{v:.4f}" for v in r['wi']]
                + [f"{v:.4f}" for v in r['wf']]
                + [str(r['epocas'])])

table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.6)

plt.tight_layout()
plt.savefig('q1_treinamentos.png', dpi=130, bbox_inches='tight')
plt.close()

# ─── Q2: Curva EQM x Épocas (T1 e T2) ────────────────────────────────────────

plt.figure(figsize=(9, 5))
plt.plot(historicos[0], label='T1')
plt.plot(historicos[1], label='T2')
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.title('Questão 3 — Curva de Aprendizagem: EQM × Épocas')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('q3_eqm.png', dpi=130, bbox_inches='tight')
plt.close()

# ─── Q3: Classificação das amostras ───────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis('off')
ax.set_title('Questão 4 — Classificação das Amostras (A = −1, B = +1)', fontsize=13, pad=12)

cols3 = ['Amostra', 'y(T1)', 'y(T2)', 'y(T3)', 'y(T4)', 'y(T5)']
rows3 = []
for i in range(len(classificacao)):
    row = [f"A{i+1}"]
    for r in resultados:
        u = r['wf'] @ X_class[i]
        y = 1 if u >= 0 else -1
        row.append('B (+1)' if y == 1 else 'A (−1)')
    rows3.append(row)

table3 = ax.table(cellText=rows3, colLabels=cols3, loc='center', cellLoc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(9)
table3.scale(1, 1.6)

plt.tight_layout()
plt.savefig('q4_classificacao.png', dpi=130, bbox_inches='tight')
plt.close()

classification_results = []
for sample_index, sample in enumerate(classificacao):
    predictions = {}

    for training_index, result in enumerate(resultados):
        activation_value = float(result["wf"] @ X_class[sample_index])
        prediction = 1 if activation_value >= 0 else -1
        predictions[f"T{training_index + 1}"] = {
            "activation": activation_value,
            "class": "B" if prediction == 1 else "A",
            "numeric_class": prediction,
        }

    classification_results.append({
        "sample": sample_index + 1,
        "features": sample.tolist(),
        "predictions": predictions,
    })

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
        "config": {
            "algorithm": "ADALINE",
            "learning_rate": 0.0025,
            "precision": 1e-6,
        },
        "trainingData": training_data,
        "testData": test_data,
        "trainingResults": training_results,
        "classificationResults": classification_results,
    }, ensure_ascii=False, indent=2)
    file.write(f"window.ADALINE_RESULTS = {json_content};\n")

# ─── Q4: Explicação pesos finais ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.set_title('Questão 5 — Por que os pesos finais são praticamente iguais?', fontsize=13, pad=12)

texto = (
    "Embora o número de épocas varie entre os treinamentos, os pesos finais\n"
    "convergem para valores praticamente idênticos. Isso ocorre porque:\n\n"
    "1. Superfície de erro convexa\n"
    "   O ADALINE minimiza o EQM, cuja superfície de erro é um parabolóide\n"
    "   com um único mínimo global. O Gradiente Descendente (Regra Delta)\n"
    "   sempre converge para esse mesmo ponto, independente da partida.\n\n"
    "2. Pesos iniciais diferentes → caminhos diferentes, mesmo destino\n"
    "   Cada treinamento começa em um ponto diferente da superfície de erro.\n"
    "   Alguns caminhos são mais curtos (menos épocas), outros mais longos,\n"
    "   mas todos chegam ao mesmo mínimo global.\n\n"
    "Conclusão: a variação no número de épocas reflete apenas a distância\n"
    "do ponto inicial até o mínimo, não o resultado final do aprendizado."
)

ax.text(0.02, 0.95, texto, transform=ax.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))

plt.tight_layout()
plt.savefig('q5_explicacao.png', dpi=130, bbox_inches='tight')
plt.close()

print("Arquivos gerados: training_data.json, test_data.json, training_results.json, classification_results.json, results_data.js")
print("Imagens geradas: q1_treinamentos.png, q3_eqm.png, q4_classificacao.png, q5_explicacao.png")
