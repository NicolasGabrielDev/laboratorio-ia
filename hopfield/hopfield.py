import json
import math
import random

TRAINING_DATA_PATH = "training_data.json"
TEST_DATA_PATH = "test_data.json"
RESULTS_PATH = "results.js"
MAX_ITERATIONS = 30
BETA = 1000.0
NOISE_EXPERIMENT_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
NOISE_EXPERIMENT_TRIALS = 40
NOISE_EXPERIMENT_SEED = 2026


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results_js(path, data):
    json_content = json.dumps(data, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"var RESULTS = {json_content};\n")


def sign(value):
    return 1 if value >= 0 else -1


def activation(value):
    if value > 20:
        return 1

    if value < -20:
        return -1

    return sign(math.tanh(BETA * value))


def train_hopfield(patterns):
    neuron_count = len(patterns[0])
    weights = [[0.0 for _ in range(neuron_count)] for _ in range(neuron_count)]

    for pattern in patterns:
        for row_index in range(neuron_count):
            for col_index in range(neuron_count):
                if row_index != col_index:
                    weights[row_index][col_index] += pattern[row_index] * pattern[col_index]

    for row_index in range(neuron_count):
        for col_index in range(neuron_count):
            weights[row_index][col_index] /= neuron_count

    return weights


def energy(state, weights):
    total = 0.0

    for row_index in range(len(state)):
        for col_index in range(len(state)):
            total += weights[row_index][col_index] * state[row_index] * state[col_index]

    return -0.5 * total


def recover_pattern(initial_state, weights, max_iterations=MAX_ITERATIONS):
    state = list(initial_state)
    energy_history = [energy(state, weights)]

    for iteration in range(1, max_iterations + 1):
        previous_state = list(state)

        for neuron_index in range(len(state)):
            local_field = 0.0

            for source_index in range(len(state)):
                local_field += weights[neuron_index][source_index] * state[source_index]

            state[neuron_index] = activation(local_field)

        energy_history.append(energy(state, weights))

        if state == previous_state:
            return state, iteration, True, energy_history

    return state, max_iterations, False, energy_history


def hamming_distance(first_vector, second_vector):
    return sum(1 for first, second in zip(first_vector, second_vector) if first != second)


def closest_pattern_label(vector, pattern_records):
    distances = []

    for pattern in pattern_records:
        distances.append({
            "label": pattern["label"],
            "distance": hamming_distance(vector, pattern["vector"]),
        })

    distances.sort(key=lambda item: item["distance"])

    return distances[0]["label"], distances[0]["distance"], distances


def vector_to_grid(vector, cols):
    return [vector[index:index + cols] for index in range(0, len(vector), cols)]


def vector_to_text_grid(vector, cols):
    grid = vector_to_grid(vector, cols)

    return ["".join("#" if value == 1 else "." for value in row) for row in grid]


def flip_random_pixels(vector, noise_level, rng):
    noisy_vector = list(vector)
    flip_count = round(len(noisy_vector) * noise_level)
    indexes = rng.sample(range(len(noisy_vector)), flip_count)

    for index in indexes:
        noisy_vector[index] *= -1

    return noisy_vector, indexes


def run_transmission_tests(test_data, pattern_records, weights, cols):
    results = []

    for sample in test_data["samples"]:
        recovered_vector, iterations, converged, energy_history = recover_pattern(
            sample["distorted_vector"],
            weights
        )

        recovered_label, recovered_distance, distances = closest_pattern_label(
            recovered_vector,
            pattern_records
        )

        success = recovered_label == sample["source_label"] and recovered_distance == 0

        results.append({
            "id": sample["id"],
            "source_label": sample["source_label"],
            "noise_level": sample["noise_level"],
            "flipped_pixels": len(sample["flipped_indexes"]),
            "initial_distance": hamming_distance(sample["original_vector"], sample["distorted_vector"]),
            "final_distance": recovered_distance,
            "recovered_label": recovered_label,
            "success": success,
            "iterations": iterations,
            "converged": converged,
            "energy_history": energy_history,
            "transmitted_grid": vector_to_text_grid(sample["original_vector"], cols),
            "distorted_grid": vector_to_text_grid(sample["distorted_vector"], cols),
            "recovered_grid": vector_to_text_grid(recovered_vector, cols),
            "distances_to_patterns": distances,
        })

    return results


def run_noise_experiment(pattern_records, weights):
    rng = random.Random(NOISE_EXPERIMENT_SEED)
    rows = []

    for noise_level in NOISE_EXPERIMENT_LEVELS:
        total_tests = 0
        successful_tests = 0
        total_final_distance = 0
        total_iterations = 0

        for pattern in pattern_records:
            for _ in range(NOISE_EXPERIMENT_TRIALS):
                noisy_vector, flipped_indexes = flip_random_pixels(pattern["vector"], noise_level, rng)

                recovered_vector, iterations, converged, energy_history = recover_pattern(
                    noisy_vector,
                    weights
                )

                recovered_label, recovered_distance, distances = closest_pattern_label(
                    recovered_vector,
                    pattern_records
                )

                success = recovered_label == pattern["label"] and recovered_distance == 0

                total_tests += 1
                successful_tests += 1 if success else 0
                total_final_distance += recovered_distance
                total_iterations += iterations

        rows.append({
            "noise_level": noise_level,
            "noise_percent": round(noise_level * 100),
            "tests": total_tests,
            "successes": successful_tests,
            "accuracy": round(successful_tests / total_tests, 4),
            "average_final_distance": round(total_final_distance / total_tests, 4),
            "average_iterations": round(total_iterations / total_tests, 4),
        })

    return rows


def build_summary(transmission_results, noise_results):
    total_tests = len(transmission_results)
    successful_tests = sum(1 for result in transmission_results if result["success"])
    average_initial_distance = sum(result["initial_distance"] for result in transmission_results) / total_tests
    average_final_distance = sum(result["final_distance"] for result in transmission_results) / total_tests

    return {
        "transmission_tests": total_tests,
        "successful_transmission_tests": successful_tests,
        "transmission_accuracy": round(successful_tests / total_tests, 4),
        "average_initial_distance": round(average_initial_distance, 4),
        "average_final_distance": round(average_final_distance, 4),
        "noise_experiment": noise_results,
    }


def main():
    training_data = load_json(TRAINING_DATA_PATH)
    test_data = load_json(TEST_DATA_PATH)

    pattern_records = training_data["patterns"]
    patterns = [record["vector"] for record in pattern_records]
    cols = pattern_records[0]["cols"]

    weights = train_hopfield(patterns)
    transmission_results = run_transmission_tests(test_data, pattern_records, weights, cols)
    noise_results = run_noise_experiment(pattern_records, weights)

    results = {
        "title": "Hopfield Associative Memory Results",
        "network": {
            "neurons": len(patterns[0]),
            "stored_patterns": len(patterns),
            "activation": "Hyperbolic tangent with very large beta, approximated as bipolar sign activation.",
            "training_rule": "Outer product rule with zero diagonal.",
            "max_iterations": MAX_ITERATIONS,
        },
        "training": {
            "patterns": pattern_records,
            "weights": weights,
        },
        "transmission_results": transmission_results,
        "summary": build_summary(transmission_results, noise_results),
        "answers": {
            "question_1": "A simulação gerou doze situações de transmissão: três versões com ruído para cada um dos quatro padrões armazenados.",
            "question_2": "Cada resultado compara a imagem transmitida, a imagem distorcida e a imagem recuperada pela rede. A recuperação é considerada correta quando o vetor final coincide exatamente com o padrão original armazenado.",
            "question_3": "Quando o nível de ruído se torna excessivo, o vetor distorcido pode sair da bacia de atração do padrão correto. Nesse caso, a rede de Hopfield pode convergir para outro padrão armazenado, para um estado espúrio ou não recuperar perfeitamente a imagem original.",
        },
    }

    save_results_js(RESULTS_PATH, results)
    print(f"{RESULTS_PATH} was generated.")


if __name__ == "__main__":
    main()