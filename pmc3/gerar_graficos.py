import json
import matplotlib.pyplot as plt
import numpy as np
from pmc3_mlp import MLP_TDNN, prepare_data, get_test_features

# Load data
with open('training_data.json', 'r') as f:
    train_data = json.load(f)

with open('test_data.json', 'r') as f:
    test_data = json.load(f)

# Topologies
topologies = {
    'Rede 1': {'p': 5, 'N1': 10},
    'Rede 2': {'p': 10, 'N1': 15},
    'Rede 3': {'p': 15, 'N1': 25}
}

# Run trainings
results = {}
for rede, config in topologies.items():
    print(f"Training {rede}...")
    results[rede] = {}
    p = config['p']
    N1 = config['N1']
    
    X_train, y_train = prepare_data(train_data, p)
    
    for i in range(1, 4): # 3 trainings (T1, T2, T3)
        print(f"  Training {i}...")
        # Random seed to ensure different weights but reproducible if needed. Here we just don't set a seed, or use i for variation
        mlp = MLP_TDNN(p, N1, seed=None)
        
        eqm_history, epochs = mlp.train(X_train, y_train)
        
        results[rede][f'T{i}'] = {
            'eqm': eqm_history[-1],
            'epochs': epochs,
            'eqm_history': eqm_history,
            'model': mlp
        }

# Identify best training for each network based on lowest EQM
best_trainings = {}
for rede in topologies.keys():
    best_t = min(results[rede].keys(), key=lambda t: results[rede][t]['eqm'])
    best_trainings[rede] = best_t
    print(f"Best training for {rede}: {best_t}")

# Generate Item 1: Tabela de treinamentos (EQM, Épocas)
def save_table_item1():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Treinamento', 'Rede 1 (EQM)', 'Rede 1 (Épocas)', 
               'Rede 2 (EQM)', 'Rede 2 (Épocas)', 
               'Rede 3 (EQM)', 'Rede 3 (Épocas)']
    
    cell_text = []
    for t in ['T1', 'T2', 'T3']:
        row = [t]
        for rede in ['Rede 1', 'Rede 2', 'Rede 3']:
            row.append(f"{results[rede][t]['eqm']:.6f}")
            row.append(str(results[rede][t]['epochs']))
        cell_text.append(row)
        
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Resultados Finais dos Treinamentos")
    plt.savefig('item1_tabela_treinamentos.png', bbox_inches='tight')
    plt.close()

# Generate Item 2: Gráfico EQM x Épocas
def save_plot_item2():
    plt.figure(figsize=(10, 6))
    for rede in topologies.keys():
        best_t = best_trainings[rede]
        history = results[rede][best_t]['eqm_history']
        plt.plot(history, label=f"{rede} ({best_t})")
        
    plt.title("Erro Quadrático Médio x Épocas (Melhores Treinamentos)")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.legend()
    plt.grid(True)
    plt.savefig('item2_eqm_epocas.png')
    plt.close()

# Generate Item 4: Tabela e cálculos de validação
def save_table_item4():
    # Evaluate models on test data
    validation_results = {}
    
    for rede, config in topologies.items():
        X_test, y_test = get_test_features(train_data, test_data, config['p'])
        validation_results[rede] = {}
        for t in ['T1', 'T2', 'T3']:
            model = results[rede][t]['model']
            preds = model.predict(X_test).flatten()
            validation_results[rede][t] = preds
            
    times = sorted([int(k) for k in test_data.keys()])
    y_true = np.array([test_data[str(t)] for t in times])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ['Amostra t', 'Desejado f(t)', 
               'R1(T1)', 'R1(T2)', 'R1(T3)',
               'R2(T1)', 'R2(T2)', 'R2(T3)',
               'R3(T1)', 'R3(T2)', 'R3(T3)']
               
    cell_text = []
    
    # Calculate Mean Relative Error and Variance
    metrics = {'R1': {}, 'R2': {}, 'R3': {}}
    for idx, t in enumerate(times):
        row = [str(t), f"{y_true[idx]:.4f}"]
        for r_idx, rede in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
            r_key = f'R{r_idx+1}'
            for tr in ['T1', 'T2', 'T3']:
                pred = validation_results[rede][tr][idx]
                row.append(f"{pred:.4f}")
        cell_text.append(row)
        
    for r_idx, rede in enumerate(['Rede 1', 'Rede 2', 'Rede 3']):
        r_key = f'R{r_idx+1}'
        for tr in ['T1', 'T2', 'T3']:
            preds = validation_results[rede][tr]
            # Relative Error: abs(true - pred) / true (or just mean error if f(t) is small)
            # Standard mean relative error: mean(|true - pred| / |true|)
            # Protect against div by zero
            rel_error = np.abs(y_true - preds) / (np.abs(y_true) + 1e-8)
            mean_rel_error = np.mean(rel_error)
            variance = np.var(preds) # or var of error? The text says: "Obtenha também a respectiva variância."
            # "Erro relativo médio ... e a respectiva variância." (Variance of relative error or variance of predictions? Usually variance of the error). Let's calculate variance of the relative error.
            var_rel_error = np.var(rel_error)
            metrics[r_key][tr] = {'mean': mean_rel_error, 'var': var_rel_error}
            
    # Add metric rows
    row_mean = ["Erro Rel. Médio", "-"]
    row_var = ["Variância", "-"]
    for r_idx in ['R1', 'R2', 'R3']:
        for tr in ['T1', 'T2', 'T3']:
            row_mean.append(f"{metrics[r_idx][tr]['mean']:.4f}")
            row_var.append(f"{metrics[r_idx][tr]['var']:.4f}")
            
    cell_text.append(row_mean)
    cell_text.append(row_var)
    
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    
    plt.title("Resultados da Validação")
    plt.savefig('item4_validacao.png', bbox_inches='tight')
    plt.close()

    # Save to json as well
    out_json = {}
    for rede in ['Rede 1', 'Rede 2', 'Rede 3']:
        out_json[rede] = {}
        for tr in ['T1', 'T2', 'T3']:
            out_json[rede][tr] = {
                'predictions': validation_results[rede][tr].tolist(),
                'mean_relative_error': metrics[f"R{rede[-1]}"][tr]['mean'],
                'variance_relative_error': metrics[f"R{rede[-1]}"][tr]['var']
            }
    with open('validation_results.json', 'w') as f:
        json.dump(out_json, f, indent=4)
        
    return validation_results

# Generate Item 5: Plot outputs vs desired for test data
def save_plot_item5(validation_results):
    times = sorted([int(k) for k in test_data.keys()])
    y_true = np.array([test_data[str(t)] for t in times])
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, y_true, label='Desejado f(t)', color='black', linewidth=2, marker='o')
    
    for rede in topologies.keys():
        best_t = best_trainings[rede]
        preds = validation_results[rede][best_t]
        plt.plot(times, preds, label=f"Estimado {rede} ({best_t})", linestyle='dashed', marker='x')
        
    plt.title("Valores Desejados vs Estimados (Melhores Treinamentos)")
    plt.xlabel("Amostra (t)")
    plt.ylabel("f(t)")
    plt.xticks(times)
    plt.legend()
    plt.grid(True)
    plt.savefig('item5_melhor_config.png') # Using this name to map appropriately or to visually show the best config
    plt.close()

# To save training details
def save_training_results():
    t_res = {}
    for rede in topologies.keys():
        t_res[rede] = {}
        for t in ['T1', 'T2', 'T3']:
            t_res[rede][t] = {
                'eqm': results[rede][t]['eqm'],
                'epochs': results[rede][t]['epochs']
            }
    with open('training_results.json', 'w') as f:
        json.dump(t_res, f, indent=4)

if __name__ == '__main__':
    save_table_item1()
    save_plot_item2()
    val_res = save_table_item4()
    save_plot_item5(val_res)
    save_training_results()
    
    print("Graphs and JSONs generated successfully.")
