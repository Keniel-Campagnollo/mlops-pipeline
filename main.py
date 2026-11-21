"""Executa o exercício de modularização de um pipeline de clustering geográfico."""

import json
from pathlib import Path

from generate_data import generate_data
from src.data import load_data, prepare_features
from src.evaluate import evaluate_model
from src.predict import predict
from src.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> dict:
    """Executa o fluxo completo e grava métricas e previsão em um arquivo JSON.

    Returns:
        Dicionário serializável com as chaves ``metrics`` e
        ``sample_prediction``, conforme o enunciado do exercício.
    """
    data_path = PROJECT_ROOT / "data" / "raw" / "points.csv"
    output_path = PROJECT_ROOT / "outputs" / "metrics.json"

    # O gerador é executado somente quando o CSV ainda não está presente.
    if not data_path.is_file():
        generate_data(str(data_path))

    points = load_data(str(data_path))
    features, scaler = prepare_features(points)
    model = train_model(features, n_clusters=5, random_state=42)
    metrics = evaluate_model(model, features)
    prediction = predict(model, scaler, lat=-23.55, lng=-46.63)

    output = {"metrics": metrics, "sample_prediction": prediction}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    print("Pipeline executado com sucesso!")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    main()
