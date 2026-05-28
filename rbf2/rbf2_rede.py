"""
RBF2 - Rede de Função de Base Radial para Aproximação Funcional de Injeção Eletrônica
Topologias: 
  - Rede 1: 3 entradas -> 5 neurônios RBF -> 1 saída
  - Rede 2: 3 entradas -> 10 neurônios RBF -> 1 saída
  - Rede 3: 3 entradas -> 15 neurônios RBF -> 1 saída

Treinamento camada oculta: K-means (K = 5, 10, 15) sobre as 150 amostras
Treinamento camada saída: Regra Delta (Modo Batch) com η = 0.01 e ε = 10⁻⁷
3 Treinamentos (T1, T2, T3) por topologia, inicializando pesos entre 0 e 1.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CARREGAR DADOS DOS ARQUIVOS JSON
# ============================================================
with open('training_data.json', 'r', encoding='utf-8') as f:
    registros_treinamento = json.load(f)

with open('test_data.json', 'r', encoding='utf-8') as f:
    registros_teste = json.load(f)

# Converte para arrays numpy
X_train = np.array([[r['x1'], r['x2'], r['x3']] for r in registros_treinamento])
d_train = np.array([r['d'] for r in registros_treinamento])

X_test = np.array([[r['x1'], r['x2'], r['x3']] for r in registros_teste])
d_test = np.array([r['d'] for r in registros_teste])

num_amostras_treinamento = len(X_train)
num_amostras_teste = len(X_test)

print(f"Total de amostras de treinamento: {num_amostras_treinamento}")
print(f"Total de amostras de teste (validação): {num_amostras_teste}")

# ============================================================
# 2. IMPLEMENTAÇÃO DO K-MEANS
# ============================================================
def kmeans(dados, num_clusters, max_iteracoes=1000, semente=42):
    """
    Algoritmo K-means para encontrar os centros da camada oculta.
    """
    rng = np.random.default_rng(semente)
    # Escolhe K pontos aleatórios como centros iniciais
    indices_iniciais = rng.choice(len(dados), num_clusters, replace=False)
    centros = dados[indices_iniciais].copy()
    
    for iteracao in range(max_iteracoes):
        # Distância Euclidiana de cada ponto a cada centro
        distancias = np.array([
            np.linalg.norm(dados - centro, axis=1)
            for centro in centros
        ])
        # Atribui rótulos baseado na menor distância
        rotulos = np.argmin(distancias, axis=0)
        
        # Recalcula centros
        novos_centros = np.zeros_like(centros)
        for k in range(num_clusters):
            pontos_k = dados[rotulos == k]
            if len(pontos_k) > 0:
                novos_centros[k] = pontos_k.mean(axis=0)
            else:
                # Caso um cluster fique vazio, mantém o centro anterior
                novos_centros[k] = centros[k]
                
        # Verifica convergência
        if np.allclose(centros, novos_centros, atol=1e-6):
            break
        centros = novos_centros
        
    return centros, rotulos

# ============================================================
# 3. ATIVAÇÃO GAUSSIANA RBF
# ============================================================
def ativacao_gaussiana(entradas, centros, variancias):
    """
    Calcula as ativações das funções gaussianas para cada entrada.
    """
    N = len(entradas)
    K = len(centros)
    phi = np.zeros((N, K))
    for j in range(K):
        distancia_quadrada = np.sum((entradas - centros[j])**2, axis=1)
        phi[:, j] = np.exp(-distancia_quadrada / (2 * variancias[j]))
    return phi

# ============================================================
# 4. CONFIGURAÇÕES E DICIONÁRIOS DE RESULTADOS
# ============================================================
topologias = [5, 10, 15]
taxa_aprendizado = 0.01
precisao_parada = 1e-7
max_epocas = 100000

# Sementes fixas para reprodutibilidade das 3 inicializações de pesos
sementes_pesos = [42, 107, 2026]

training_results = {
    "kmeans_por_rede": {},
    "treinamentos": {
        "Rede_5": [],
        "Rede_10": [],
        "Rede_15": []
    }
}

validation_results = {
    "Rede_5": [],
    "Rede_10": [],
    "Rede_15": []
}

# ============================================================
# 5. LOOP PRINCIPAL DE TREINAMENTO E VALIDAÇÃO
# ============================================================
for n1 in topologias:
    rede_nome = f"Rede_{n1}"
    print(f"\n============================================================")
    print(f" TREINANDO A {rede_nome.upper()} (N1 = {n1})")
    print(f"============================================================")
    
    # --- Passo A: Treinar Camada Oculta com K-means ---
    # Usamos semente 42 para K-means fixo por topologia
    centros_clusters, rotulos_clusters = kmeans(X_train, num_clusters=n1, semente=42)
    
    # Calcular variâncias
    variancias_clusters = []
    for k in range(n1):
        pontos_k = X_train[rotulos_clusters == k]
        if len(pontos_k) > 0:
            var_k = np.mean(np.sum((pontos_k - centros_clusters[k])**2, axis=1))
            # Proteção contra variância nula
            if var_k < 1e-4:
                var_k = 1e-4
        else:
            var_k = 0.1
        variancias_clusters.append(var_k)
    variancias_clusters = np.array(variancias_clusters)
    
    # Salvar metadados do K-means
    training_results["kmeans_por_rede"][rede_nome] = {
        "centros": [{"x1": float(c[0]), "x2": float(c[1]), "x3": float(c[2])} for c in centros_clusters],
        "variancias": [float(v) for v in variancias_clusters]
    }
    
    # --- Passo B: Calcular Ativações RBF ---
    phi_train = ativacao_gaussiana(X_train, centros_clusters, variancias_clusters)
    phi_train_bias = np.column_stack([np.ones(num_amostras_treinamento), phi_train])
    
    phi_test = ativacao_gaussiana(X_test, centros_clusters, variancias_clusters)
    phi_test_bias = np.column_stack([np.ones(num_amostras_teste), phi_test])
    
    # --- Passo C: Executar 3 Treinamentos (T1, T2, T3) ---
    for t_idx, semente in enumerate(sementes_pesos):
        t_nome = f"T{t_idx+1}"
        print(f"\n  -> Inicializando Treinamento {t_nome} (Semente={semente})")
        
        # Inicializa pesos aleatórios entre 0 e 1 (bias incluso no índice 0)
        rng = np.random.default_rng(semente)
        w = rng.uniform(0, 1, n1 + 1)
        
        historico_eqm = []
        
        # Loop de Épocas (Batch Delta Rule)
        for epoca in range(max_epocas):
            # Forward
            y_train = phi_train_bias @ w
            erros_train = d_train - y_train
            
            # EQM = 0.5 * mean(erros^2)
            eqm = float(np.mean(0.5 * erros_train**2))
            historico_eqm.append(eqm)
            
            # Critério de parada: convergência consecutivo do EQM
            if epoca > 0 and abs(historico_eqm[-1] - historico_eqm[-2]) < precisao_parada:
                break
                
            # Atualização em lote (batch):
            # grad_w = η * (1/N) * Φ.T @ erros
            gradiente = taxa_aprendizado * (phi_train_bias.T @ erros_train) / num_amostras_treinamento
            w += gradiente
            
        final_epochs = len(historico_eqm)
        final_eqm = historico_eqm[-1]
        print(f"     Convergido em {final_epochs} épocas. EQM final: {final_eqm:.8f}")
        
        # --- Passo D: Validação (Conjunto de Teste) ---
        y_test = phi_test_bias @ w
        erros_relativos = np.abs((d_test - y_test) / d_test) * 100
        erm = float(np.mean(erros_relativos))
        variancia_erros = float(np.var(erros_relativos))
        
        print(f"     Validação -> Erro Relativo Médio: {erm:.4f}%, Variância: {variancia_erros:.4f}%")
        
        # Salva resultados do treinamento
        training_results["treinamentos"][rede_nome].append({
            "treinamento": t_nome,
            "semente": semente,
            "epocas": final_epochs,
            "eqm_final": final_eqm,
            "pesos": [float(val) for val in w],
            "eqm_historico": [float(val) for val in historico_eqm]
        })
        
        # Salva predições de validação para as 15 amostras
        validation_results[rede_nome].append({
            "treinamento": t_nome,
            "erro_relativo_medio_pct": erm,
            "variancia_pct": variancia_erros,
            "predicoes": [
                {
                    "amostra": int(d['amostra']),
                    "x1": float(d['x1']),
                    "x2": float(d['x2']),
                    "x3": float(d['x3']),
                    "d": float(d['d']),
                    "y": float(y_test[idx]),
                    "erro_relativo_pct": float(erros_relativos[idx])
                }
                for idx, d in enumerate(registros_teste)
            ]
        })

# ============================================================
# 6. SALVAR ARQUIVOS JSON DE SAÍDA
# ============================================================
with open('training_results.json', 'w', encoding='utf-8') as f:
    json.dump(training_results, f, indent=2, ensure_ascii=False)
print("\nSalvo: training_results.json")

with open('validation_results.json', 'w', encoding='utf-8') as f:
    json.dump(validation_results, f, indent=2, ensure_ascii=False)
print("Salvo: validation_results.json")

print("\n--- Treinamento Concluído com Sucesso! ---")
