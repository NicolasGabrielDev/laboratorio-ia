import json
import random

ROWS = 9
COLS = 5
PIXEL_ON = 1
PIXEL_OFF = -1
NOISE_LEVEL = 0.20
SAMPLES_PER_PATTERN = 3
RANDOM_SEED = 42

PATTERN_GRIDS = {
    "pattern_1": [
        "..##.",
        ".###.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
    ],
    "pattern_2": [
        "#####",
        "#####",
        "...##",
        "...##",
        "#####",
        "##...",
        "##...",
        "#####",
        "#####",
    ],
    "pattern_3": [
        "#####",
        "#####",
        "...##",
        "...##",
        "#####",
        "...##",
        "...##",
        "#####",
        "#####",
    ],
    "pattern_4": [
        "##.##",
        "##.##",
        "##.##",
        "#####",
        "#####",
        "...##",
        "...##",
        "...##",
        "...##",
    ],
}


def grid_to_vector(grid):
    return [PIXEL_ON if char == "#" else PIXEL_OFF for row in grid for char in row]


def vector_to_grid(vector):
    return [vector[index:index + COLS] for index in range(0, len(vector), COLS)]


def flip_random_pixels(vector, noise_level, rng):
    noisy_vector = list(vector)
    flip_count = round(len(noisy_vector) * noise_level)
    indexes = rng.sample(range(len(noisy_vector)), flip_count)

    for index in indexes:
        noisy_vector[index] *= -1

    return noisy_vector, indexes


def build_training_data():
    patterns = []

    for label, grid in PATTERN_GRIDS.items():
        vector = grid_to_vector(grid)
        patterns.append({
            "label": label,
            "rows": ROWS,
            "cols": COLS,
            "grid_text": grid,
            "vector": vector,
        })

    return {
        "description": "Training patterns for a 45-neuron Hopfield network.",
        "encoding": {
            "white_pixel": PIXEL_OFF,
            "dark_pixel": PIXEL_ON,
        },
        "patterns": patterns,
    }


def build_test_data(training_data):
    rng = random.Random(RANDOM_SEED)
    samples = []

    for pattern in training_data["patterns"]:
        for sample_number in range(1, SAMPLES_PER_PATTERN + 1):
            noisy_vector, flipped_indexes = flip_random_pixels(pattern["vector"], NOISE_LEVEL, rng)
            samples.append({
                "id": f"{pattern['label']}_sample_{sample_number}",
                "source_label": pattern["label"],
                "noise_level": NOISE_LEVEL,
                "flipped_indexes": flipped_indexes,
                "original_vector": pattern["vector"],
                "distorted_vector": noisy_vector,
            })

    return {
        "description": "Distorted samples generated from the training patterns.",
        "random_seed": RANDOM_SEED,
        "samples_per_pattern": SAMPLES_PER_PATTERN,
        "samples": samples,
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    training_data = build_training_data()
    test_data = build_test_data(training_data)

    save_json("training_data.json", training_data)
    save_json("test_data.json", test_data)

    print("training_data.json and test_data.json were generated.")


if __name__ == "__main__":
    main()
