import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
LEARNING_RATE = 0.05
EPOCHS = 100
HOURS = ["7h", "8h", "9h", "10h", "11h", "12h"]

# Cada vetor possui 6 entradas, uma para cada hora medida no dia.
# A classe representa o perfil de demanda elétrica daquele conjunto de medições.
TRAINING_ROWS = [
    (1, [2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649], 1),
    (2, [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110], 1),
    (3, [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099], 1),
    (4, [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744], 1),
    (5, [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276], 2),
    (6, [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331], 2),
    (7, [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383], 2),
    (8, [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193], 2),
    (9, [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850], 3),
    (10, [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645], 3),
    (11, [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470], 3),
    (12, [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876], 3),
    (13, [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749], 4),
    (14, [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142], 4),
    (15, [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966], 4),
    (16, [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336], 4),
]

TEST_ROWS = [
    (1, [2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718]),
    (2, [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958]),
    (3, [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675]),
    (4, [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695]),
    (5, [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868]),
    (6, [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839]),
    (7, [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640]),
    (8, [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808]),
]


def save_json(filename, data):
    path = BASE_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_results_data(training_results, classification_results):
    data = {
        "trainingResults": training_results,
        "classificationResults": classification_results,
    }
    json_content = json.dumps(data, ensure_ascii=False, indent=2)
    path = BASE_DIR / "results_data.js"

    with path.open("w", encoding="utf-8") as file:
        file.write(f"window.LVQ1_RESULTS = {json_content};\n")


def build_training_data():
    return [
        {"sample": sample, "features": features, "class": label}
        for sample, features, label in TRAINING_ROWS
    ]


def build_test_data():
    return [
        {"day": day, "features": features}
        for day, features in TEST_ROWS
    ]


def euclidean_distance(first_vector, second_vector):
    # A distância euclidiana mede o quanto dois perfis de potência são parecidos.
    # Quanto menor a distância, mais similar o vetor é ao protótipo da classe.
    return math.sqrt(sum((first - second) ** 2 for first, second in zip(first_vector, second_vector)))


def class_centroids(training_data):
    # Na LVQ, cada classe é representada por um ou mais protótipos.
    # Aqui usamos um protótipo por classe, iniciado no centróide das amostras daquela classe.
    centroids = []

    for class_label in sorted({sample["class"] for sample in training_data}):
        class_samples = [
            sample["features"]
            for sample in training_data
            if sample["class"] == class_label
        ]
        centroid = [
            sum(values) / len(values)
            for values in zip(*class_samples)
        ]
        centroids.append({"class": class_label, "weights": centroid})

    return centroids


def closest_prototype(features, prototypes):
    # O protótipo vencedor é aquele com menor distância até a amostra analisada.
    # A classe prevista pela rede será a classe desse protótipo vencedor.
    distances = [
        {
            "class": prototype["class"],
            "distance": euclidean_distance(features, prototype["weights"]),
            "prototype": prototype,
        }
        for prototype in prototypes
    ]
    distances.sort(key=lambda item: item["distance"])

    return distances[0], distances


def train_lvq1(training_data):
    # O treinamento ajusta os protótipos para que cada um fique mais representativo
    # da região ocupada pela sua própria classe no espaço de características.
    prototypes = class_centroids(training_data)
    history = []

    for epoch in range(EPOCHS):
        # O decaimento da taxa de aprendizagem reduz o tamanho dos ajustes ao longo do treino.
        # Isso permite mudanças maiores no início e refinamentos menores no final.
        learning_rate = LEARNING_RATE * (1 - epoch / EPOCHS)
        squared_error = 0.0
        hits = 0

        for sample in training_data:
            winner, _ = closest_prototype(sample["features"], prototypes)
            prototype = winner["prototype"]
            # Regra principal da LVQ-1:
            # se o protótipo vencedor tem a classe correta, ele se aproxima da amostra;
            # se tem a classe errada, ele se afasta da amostra.
            direction = 1 if prototype["class"] == sample["class"] else -1

            if direction == 1:
                hits += 1

            squared_error += winner["distance"] ** 2
            # Fórmula: w_novo = w_atual + direção * alfa * (x - w_atual)
            # w é o protótipo, x é a amostra, alfa é a taxa de aprendizagem.
            prototype["weights"] = [
                weight + direction * learning_rate * (feature - weight)
                for weight, feature in zip(prototype["weights"], sample["features"])
            ]

        history.append({
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "mean_squared_error": squared_error / len(training_data),
            "accuracy": hits / len(training_data),
        })

    return prototypes, history


def classify_samples(samples, prototypes):
    # Depois do treinamento, os pesos dos protótipos ficam fixos.
    # Para classificar uma nova amostra, basta encontrar o protótipo mais próximo.
    classified_samples = []

    for sample in samples:
        winner, distances = closest_prototype(sample["features"], prototypes)
        classified_samples.append({
            **sample,
            "predicted_class": winner["class"],
            "winner_distance": winner["distance"],
            "distances": [
                {"class": item["class"], "distance": item["distance"]}
                for item in distances
            ],
        })

    return classified_samples


def generate_training_chart(history):
    # O gráfico do erro mostra se os protótipos estabilizaram durante o treinamento.
    path = BASE_DIR / "lvq1_training_error.png"
    epochs = [item["epoch"] for item in history]
    errors = [item["mean_squared_error"] for item in history]

    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, errors, color="#1f6feb", linewidth=2)
    plt.title("Erro médio quadrático durante o treinamento LVQ-1")
    plt.xlabel("Época")
    plt.ylabel("Erro médio quadrático")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_profiles_chart(test_results):
    # Este gráfico ajuda a comparar visualmente os perfis de potência classificados.
    path = BASE_DIR / "lvq1_test_profiles.png"

    plt.figure(figsize=(9, 5.2))

    for result in test_results:
        plt.plot(
            HOURS,
            result["features"],
            marker="o",
            linewidth=1.7,
            label=f"Dia {result['day']} - Classe {result['predicted_class']}",
        )

    plt.title("Perfis de potência classificados")
    plt.xlabel("Horário")
    plt.ylabel("Potência")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    # Fluxo completo pedido no enunciado:
    # dados do DOCX -> JSON -> treinamento LVQ-1 -> classificação -> JSON, JS de dados e imagens.
    training_data = build_training_data()
    test_data = build_test_data()

    save_json("training_data.json", training_data)
    save_json("test_data.json", test_data)

    prototypes, history = train_lvq1(training_data)
    classified_training = classify_samples(training_data, prototypes)
    classified_test = classify_samples(test_data, prototypes)
    training_hits = sum(
        1
        for sample in classified_training
        if sample["predicted_class"] == sample["class"]
    )

    results = {
        "config": {
            "algorithm": "LVQ-1",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "features": HOURS,
            "prototype_initialization": "class_centroids",
        },
        "summary": {
            "final_mean_squared_error": history[-1]["mean_squared_error"],
            "training_accuracy": training_hits / len(classified_training),
        },
        "prototypes": prototypes,
        "history": history,
        "classified_training": classified_training,
        "classified_test": classified_test,
    }

    save_json("training_results.json", results)
    save_json("classification_results.json", classified_test)
    save_results_data(results, classified_test)
    generate_training_chart(history)
    generate_profiles_chart(classified_test)

    print("Arquivos gerados em rnalvq1:")
    print("- training_data.json")
    print("- test_data.json")
    print("- training_results.json")
    print("- classification_results.json")
    print("- results_data.js")
    print("- lvq1_training_error.png")
    print("- lvq1_test_profiles.png")
    print("Classificações:", [item["predicted_class"] for item in classified_test])


if __name__ == "__main__":
    main()
