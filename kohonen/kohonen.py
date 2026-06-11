import json
import math
import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

GRID_ROWS  = 4
GRID_COLS  = 4
N1         = GRID_ROWS * GRID_COLS
INPUT_DIM  = 3
LR         = 0.001
RADIUS     = 1.0
EPOCHS     = 1000
SEED       = 42

def neuron_rc(idx):
    return divmod(idx, GRID_COLS)

def grid_dist(i, j):
    ri, ci = neuron_rc(i)
    rj, cj = neuron_rc(j)
    return math.sqrt((ri - rj) ** 2 + (ci - cj) ** 2)

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def find_bmu(x, weights):
    return min(range(N1), key=lambda j: euclidean(x, weights[j]))

def neighborhood(j, bmu):
    return 1.0 if grid_dist(j, bmu) <= RADIUS else 0.0

def train(data, weights):
    errors = []
    for epoch in range(EPOCHS):
        random.shuffle(data)
        epoch_err = 0.0
        for sample in data:
            x = sample["features"]
            bmu = find_bmu(x, weights)
            epoch_err += euclidean(x, weights[bmu]) ** 2
            for j in range(N1):
                h = neighborhood(j, bmu)
                if h > 0:
                    for k in range(INPUT_DIM):
                        weights[j][k] += LR * h * (x[k] - weights[j][k])
        errors.append(epoch_err / len(data))
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}  MSE={errors[-1]:.6f}")
    return errors

def classify_dataset(data, weights):
    return [{**s, "bmu": find_bmu(s["features"], weights) + 1} for s in data]

def dominant_class_per_neuron(classified):
    counts = [{} for _ in range(N1)]
    for s in classified:
        c = s["class"]
        counts[s["bmu"] - 1][c] = counts[s["bmu"] - 1].get(c, 0) + 1
    dom = [max(c, key=c.get) if c else None for c in counts]
    return dom, counts

def main():
    with open("training_data.json") as f:
        training_data = json.load(f)
    with open("test_data.json") as f:
        test_data = json.load(f)

    random.seed(SEED)
    weights = [[random.uniform(0, 1) for _ in range(INPUT_DIM)] for _ in range(N1)]

    print("Training Kohonen SOM ...")
    errors = train(training_data, weights)
    print(f"Done. Final MSE = {errors[-1]:.6f}")

    classified_train = classify_dataset(training_data, weights)
    dom_class, counts = dominant_class_per_neuron(classified_train)

    classified_test = []
    for s in test_data:
        bmu = find_bmu(s["features"], weights) + 1
        classified_test.append({**s, "bmu": bmu, "predicted_class": dom_class[bmu - 1] or "?"})

    results = {
        "config": {"N1": N1, "grid": "4x4", "lr": LR, "radius": RADIUS, "epochs": EPOCHS},
        "final_mse": errors[-1],
        "weights": weights,
        "neuron_dominant_class": dom_class,
        "neuron_counts": counts,
        "error_history": errors,
        "classified_training": classified_train,
        "classified_test": classified_test,
    }

    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved training_results.json")

    with open("results.js", "w") as f:
        f.write("const RESULTS = ")
        json.dump(results, f)
        f.write(";")
    print("Saved results.js")

    with open("results_data.js", "w", encoding="utf-8") as f:
        f.write("window.KOHONEN_RESULTS = ")
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write(";\n")
    print("Saved results_data.js")

    print("\n── Test Classification ──")
    for s in classified_test:
        print(f"  Sample {s['sample']:2d}: BMU=N{s['bmu']:2d}  Class={s['predicted_class']}")

    print("\n── Neurons by Class ──")
    for cls in ["A", "B", "C"]:
        ns = [i + 1 for i, d in enumerate(dom_class) if d == cls]
        print(f"  Class {cls}: {ns}")

if __name__ == "__main__":
    main()
