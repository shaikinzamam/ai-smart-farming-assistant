import json
from pathlib import Path


def load_solutions(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_solution_bundle(solutions: dict, disease: str) -> dict:
    default_bundle = solutions.get(
        "Unknown",
        {
            "organic_solution": "Inspect the crop closely and isolate the affected plant.",
            "chemical_solution": "Consult a local agronomist before applying chemicals.",
            "confidence_hint": "Low confidence",
        },
    )
    return solutions.get(disease, default_bundle)
