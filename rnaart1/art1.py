import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
VARIABLES = [f"x{index}" for index in range(1, 17)]
VIGILANCE_VALUES = [0.5, 0.8, 0.9, 0.99]
CHOICE_ALPHA = 0.001

# Cada situação é um vetor binário com 16 variáveis de status do processo.
# O valor 1 indica presença/ativação daquela característica; o valor 0 indica ausência.
SITUATION_ROWS = [
    (1, [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]),
    (2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]),
    (3, [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
    (4, [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0]),
    (5, [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]),
    (6, [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]),
    (7, [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]),
    (8, [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
    (9, [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]),
    (10, [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]),
]


def save_json(filename, data):
    path = BASE_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_results_data(results):
    json_content = json.dumps(results, ensure_ascii=False, indent=2)
    path = BASE_DIR / "results_data.js"

    with path.open("w", encoding="utf-8") as file:
        file.write(f"window.ART1_RESULTS = {json_content};\n")


def build_training_data():
    return [
        {
            "situation": situation,
            "features": features,
            "active_features": sum(features),
        }
        for situation, features in SITUATION_ROWS
    ]


def vector_intersection(first_vector, second_vector):
    # Na ART-1, a comparação entre entrada e protótipo usa a interseção lógica.
    # A interseção preserva apenas as características ativas nos dois vetores.
    return [
        first_value & second_value
        for first_value, second_value in zip(first_vector, second_vector)
    ]


def vector_magnitude(vector):
    # Como os vetores são binários, a magnitude é a quantidade de bits ativos.
    return sum(vector)


def choice_score(features, prototype):
    # A função de escolha mede o quanto uma categoria é candidata para receber a entrada.
    # O alpha evita divisão por zero e reduz empates artificiais em protótipos muito pequenos.
    intersection = vector_intersection(features, prototype)

    return vector_magnitude(intersection) / (CHOICE_ALPHA + vector_magnitude(prototype))


def vigilance_match(features, prototype):
    # O teste de vigilância mede quanto da entrada está coberto pelo protótipo vencedor.
    # Se esse valor for menor que rho, a categoria é rejeitada para essa entrada.
    intersection = vector_intersection(features, prototype)
    feature_count = vector_magnitude(features)

    if feature_count == 0:
        return 0.0

    return vector_magnitude(intersection) / feature_count


def sorted_category_candidates(features, categories, rejected_categories):
    candidates = []

    for index, category in enumerate(categories):
        if index in rejected_categories:
            continue

        candidates.append({
            "index": index,
            "choice_score": choice_score(features, category["prototype"]),
            "match": vigilance_match(features, category["prototype"]),
        })

    return sorted(candidates, key=lambda item: item["choice_score"], reverse=True)


def classify_with_art1(training_data, vigilance):
    # A ART-1 cria categorias de forma incremental.
    # Se nenhuma categoria existente atende ao grau de vigilância, nasce uma nova classe.
    categories = []
    assignments = []

    for sample in training_data:
        rejected_categories = set()
        selected_category = None
        selected_match = 1.0
        selected_choice_score = 1.0
        created_category = False

        while selected_category is None:
            candidates = sorted_category_candidates(
                sample["features"],
                categories,
                rejected_categories,
            )

            if not candidates:
                categories.append({
                    "category": len(categories) + 1,
                    "prototype": list(sample["features"]),
                })
                selected_category = categories[-1]
                created_category = True
                break

            candidate = candidates[0]

            if candidate["match"] >= vigilance:
                selected_category = categories[candidate["index"]]
                selected_match = candidate["match"]
                selected_choice_score = candidate["choice_score"]
                selected_category["prototype"] = vector_intersection(
                    selected_category["prototype"],
                    sample["features"],
                )
            else:
                rejected_categories.add(candidate["index"])

        assignments.append({
            "situation": sample["situation"],
            "category": selected_category["category"],
            "created_category": created_category,
            "match": selected_match,
            "choice_score": selected_choice_score,
        })

    groups = []

    for category in categories:
        situations = [
            assignment["situation"]
            for assignment in assignments
            if assignment["category"] == category["category"]
        ]
        groups.append({
            "category": category["category"],
            "situations": situations,
            "prototype": category["prototype"],
            "active_features": vector_magnitude(category["prototype"]),
        })

    return {
        "vigilance": vigilance,
        "active_category_count": len(categories),
        "groups": groups,
        "assignments": assignments,
    }


def run_all_simulations(training_data):
    return [
        classify_with_art1(training_data, vigilance)
        for vigilance in VIGILANCE_VALUES
    ]


def generate_category_chart(simulations):
    path = BASE_DIR / "art1_active_categories.png"
    labels = [str(simulation["vigilance"]) for simulation in simulations]
    counts = [simulation["active_category_count"] for simulation in simulations]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, counts, color="#185a9d")
    plt.title("Classes ativas por grau de vigilância")
    plt.xlabel("Grau de vigilância")
    plt.ylabel("Quantidade de classes ativas")
    plt.ylim(0, max(counts) + 1)
    plt.grid(axis="y", alpha=0.25)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_group_chart(simulations):
    path = BASE_DIR / "art1_group_sizes.png"
    labels = []
    sizes = []

    for simulation in simulations:
        for group in simulation["groups"]:
            labels.append(f"rho {simulation['vigilance']} C{group['category']}")
            sizes.append(len(group["situations"]))

    plt.figure(figsize=(11, 5.6))
    plt.bar(labels, sizes, color="#2f9e44")
    plt.title("Tamanho dos agrupamentos formados pela ART-1")
    plt.xlabel("Simulação e classe")
    plt.ylabel("Quantidade de situações")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_results(training_data, simulations):
    return {
        "config": {
            "algorithm": "ART-1",
            "variables": VARIABLES,
            "vigilance_values": VIGILANCE_VALUES,
            "choice_alpha": CHOICE_ALPHA,
            "learning": "fast_learning_binary_intersection",
        },
        "summary": [
            {
                "vigilance": simulation["vigilance"],
                "active_category_count": simulation["active_category_count"],
                "groups": [
                    {
                        "category": group["category"],
                        "situations": group["situations"],
                    }
                    for group in simulation["groups"]
                ],
            }
            for simulation in simulations
        ],
        "training_data": training_data,
        "simulations": simulations,
    }


def main():
    # Fluxo completo: dados do DOCX -> JSON -> simulações ART-1 -> JSON, JS de dados e imagens.
    training_data = build_training_data()
    simulations = run_all_simulations(training_data)
    results = build_results(training_data, simulations)

    save_json("training_data.json", training_data)
    save_json("training_results.json", results)
    save_json("classification_results.json", simulations)
    save_results_data(results)
    generate_category_chart(simulations)
    generate_group_chart(simulations)

    print("Arquivos gerados em rnaart1:")
    print("- training_data.json")
    print("- training_results.json")
    print("- classification_results.json")
    print("- results_data.js")
    print("- art1_active_categories.png")
    print("- art1_group_sizes.png")

    for simulation in simulations:
        groups = [
            f"C{group['category']}={group['situations']}"
            for group in simulation["groups"]
        ]
        print(
            f"rho={simulation['vigilance']}: "
            f"{simulation['active_category_count']} classes; "
            + "; ".join(groups)
        )


if __name__ == "__main__":
    main()
