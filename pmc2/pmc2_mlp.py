"""
pmc2_mlp.py
Treina a rede Perceptron Multicamadas (MLP) com:
  - Backpropagation padrão
  - Backpropagation com momentum
Salva os resultados em training_results.json e validation_results.json.
Arquitetura: 4 entradas → 15 neurônios ocultos → 3 saídas
"""

import json
import time
import numpy as np

# ─────────────────────────────────────────────
# Semente para reprodutibilidade dos pesos iniciais
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Carrega dados dos arquivos JSON
# ─────────────────────────────────────────────
with open("training_data.json") as f:
    train_records = json.load(f)

with open("test_data.json") as f:
    test_records = json.load(f)

# Converte para arrays numpy
def records_to_arrays(records):
    X = np.array([[r["inputs"]["x1"], r["inputs"]["x2"],
                   r["inputs"]["x3"], r["inputs"]["x4"]] for r in records])
    D = np.array([[r["targets"]["d1"], r["targets"]["d2"],
                   r["targets"]["d3"]] for r in records])
    return X, D

X_train, D_train = records_to_arrays(train_records)
X_test,  D_test  = records_to_arrays(test_records)

N_TRAIN = X_train.shape[0]   # 130 (total lido no anexo)
N_IN    = 4                   # entradas
N_HID   = 15                  # neurônios na camada oculta (conforme figura)
N_OUT   = 3                   # saídas
LR      = 0.1                 # taxa de aprendizado η
BETA    = 0.9                 # fator de momentum
EPS     = 1e-6                # precisão ε
MAX_EP  = 50000               # limite de épocas para evitar loop infinito

# ─────────────────────────────────────────────
# Função sigmoid e sua derivada
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    """Derivada em função da saída já ativada: a*(1-a)"""
    return a * (1.0 - a)

# ─────────────────────────────────────────────
# Inicialização dos pesos (0 a 1) — mesmos para os dois treinamentos
# ─────────────────────────────────────────────
# W1: (N_HID x N_IN+1)  — inclui bias como última coluna de entradas
# W2: (N_OUT x N_HID+1) — inclui bias da camada oculta
W1_init = np.random.uniform(0, 1, (N_HID, N_IN + 1))
W2_init = np.random.uniform(0, 1, (N_OUT, N_HID + 1))

# ─────────────────────────────────────────────
# Função de treinamento genérica
# ─────────────────────────────────────────────
def train(W1_0, W2_0, use_momentum=False):
    """
    Executa o treinamento backpropagation padrão ou com momentum.
    Retorna (eqm_por_epoca, W1_final, W2_final, tempo_segundos)
    """
    W1 = W1_0.copy()
    W2 = W2_0.copy()

    # Incrementos anteriores para momentum (zerados no início)
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)

    eqm_history = []
    t_start = time.time()

    for epoch in range(MAX_EP):
        eqm_epoch = 0.0

        # Itera sobre cada amostra (modo padrão/online)
        for n in range(N_TRAIN):
            x = X_train[n]          # vetor de entrada (4,)
            d = D_train[n]          # vetor desejado (3,)

            # Acrescenta bias (+1) às entradas
            x_b = np.append(x, 1.0)          # (5,)

            # ── Forward pass ──────────────────────
            z1 = W1 @ x_b                    # (15,)
            a1 = sigmoid(z1)                 # (15,) — saída camada oculta
            a1_b = np.append(a1, 1.0)        # (16,) — adiciona bias

            z2 = W2 @ a1_b                   # (3,)
            a2 = sigmoid(z2)                 # (3,) — saída final

            # ── Erro ──────────────────────────────
            e = d - a2                        # (3,)
            eqm_epoch += 0.5 * np.sum(e ** 2)

            # ── Backward pass ─────────────────────
            # Delta camada de saída
            delta2 = e * sigmoid_deriv(a2)    # (3,)

            # Delta camada oculta (sem o nó de bias)
            delta1 = (W2[:, :-1].T @ delta2) * sigmoid_deriv(a1)  # (15,)

            # ── Gradientes ────────────────────────
            dW2 = LR * np.outer(delta2, a1_b)  # (3,16)
            dW1 = LR * np.outer(delta1, x_b)   # (15,5)

            # ── Aplica momentum se solicitado ──────
            if use_momentum:
                W2 += dW2 + BETA * dW2_prev
                W1 += dW1 + BETA * dW1_prev
                dW2_prev = dW2.copy()   # salva gradiente puro, sem momentum
                dW1_prev = dW1.copy()
            else:
                W2 += dW2
                W1 += dW1

        # EQM médio da época
        eqm_mean = eqm_epoch / N_TRAIN
        eqm_history.append(float(eqm_mean))

        # Critério de parada
        if eqm_mean < EPS:
            break

    elapsed = time.time() - t_start
    return eqm_history, W1, W2, elapsed

# ─────────────────────────────────────────────
# Pós-processamento: arredondamento simétrico
# ─────────────────────────────────────────────
def postprocess(output):
    """Arredondamento simétrico: valores >= 0.5 → 1, < 0.5 → 0"""
    return (output >= 0.5).astype(int)

# ─────────────────────────────────────────────
# Função de inferência (forward pass único)
# ─────────────────────────────────────────────
def predict(W1, W2, X):
    results = []
    for x in X:
        x_b = np.append(x, 1.0)
        a1  = sigmoid(W1 @ x_b)
        a1_b = np.append(a1, 1.0)
        a2  = sigmoid(W2 @ a1_b)
        results.append(a2)
    return np.array(results)

# ─────────────────────────────────────────────
# Treinamento 1: Backpropagation Padrão
# ─────────────────────────────────────────────
print("Treinando com Backpropagation Padrão...")
eqm_std, W1_std, W2_std, t_std = train(W1_init, W2_init, use_momentum=False)
print(f"  Épocas: {len(eqm_std)} | EQM final: {eqm_std[-1]:.2e} | Tempo: {t_std:.2f}s")

# ─────────────────────────────────────────────
# Treinamento 2: Backpropagation com Momentum
# ─────────────────────────────────────────────
print("Treinando com Backpropagation + Momentum...")
eqm_mom, W1_mom, W2_mom, t_mom = train(W1_init, W2_init, use_momentum=True)
print(f"  Épocas: {len(eqm_mom)} | EQM final: {eqm_mom[-1]:.2e} | Tempo: {t_mom:.2f}s")

# ─────────────────────────────────────────────
# Validação no conjunto de teste
# ─────────────────────────────────────────────
def validate(W1, W2, X_test, D_test, test_records):
    raw_out = predict(W1, W2, X_test)
    pred    = postprocess(raw_out)
    correct = int(np.all(pred == D_test, axis=1).sum())
    accuracy = correct / len(D_test) * 100

    details = []
    for i, rec in enumerate(test_records):
        details.append({
            "sample": rec["sample"],
            "desired": D_test[i].tolist(),
            "raw_output": [round(float(v), 4) for v in raw_out[i]],
            "predicted": pred[i].tolist(),
            "correct": bool(np.all(pred[i] == D_test[i]))
        })
    return accuracy, details

acc_std, details_std = validate(W1_std, W2_std, X_test, D_test, test_records)
acc_mom, details_mom = validate(W1_mom, W2_mom, X_test, D_test, test_records)

print(f"\nTaxa de Acerto (Padrão):   {acc_std:.1f}%")
print(f"Taxa de Acerto (Momentum): {acc_mom:.1f}%")

# ─────────────────────────────────────────────
# Salva training_results.json
# ─────────────────────────────────────────────
training_results = {
    "standard": {
        "method": "Backpropagation Padrao",
        "epochs": len(eqm_std),
        "final_eqm": float(eqm_std[-1]),
        "time_seconds": round(t_std, 4),
        "eqm_per_epoch": eqm_std
    },
    "momentum": {
        "method": "Backpropagation com Momentum",
        "epochs": len(eqm_mom),
        "final_eqm": float(eqm_mom[-1]),
        "time_seconds": round(t_mom, 4),
        "eqm_per_epoch": eqm_mom
    }
}

with open("training_results.json", "w") as f:
    json.dump(training_results, f, indent=2)
print("training_results.json salvo")

# ─────────────────────────────────────────────
# Salva validation_results.json
# ─────────────────────────────────────────────
validation_results = {
    "standard": {
        "accuracy_pct": round(acc_std, 2),
        "details": details_std
    },
    "momentum": {
        "accuracy_pct": round(acc_mom, 2),
        "details": details_mom
    }
}

with open("validation_results.json", "w") as f:
    json.dump(validation_results, f, indent=2)
print("validation_results.json salvo")