"""Testes automatizados dos módulos e do fluxo completo do exercício 054."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_data import generate_data
from src.data import load_data, prepare_features
from src.evaluate import evaluate_model
from src.predict import predict
from src.train import train_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trained_pipeline(tmp_path: Path):
    file = generate_data(str(tmp_path / "points.csv"), n=300)
    points = load_data(str(file))
    features, scaler = prepare_features(points)
    model = train_model(features, n_clusters=5, random_state=42)
    return features, scaler, model


def test_generated_data_are_reproducible(tmp_path: Path):
    first = generate_data(str(tmp_path / "first.csv"))
    second = generate_data(str(tmp_path / "second.csv"))
    assert first.read_bytes() == second.read_bytes()
    points = load_data(str(first))
    assert points.shape == (300, 2)
    assert np.isfinite(points).all()


def test_missing_file_raises_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Arquivo CSV"):
        load_data(str(tmp_path / "missing.csv"))


def test_missing_csv_columns_raises_error(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"latitude": [-23.5, -23.6], "longitude": [-46.6, -46.7]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="colunas"):
        load_data(str(path))


@pytest.mark.parametrize("coordinates", [
    [[-23.5, np.nan], [-23.6, -46.7]],
    [[91, -46.6], [-23.6, -46.7]],
    [[-23.5, -181], [-23.6, -46.7]],
    [["invalid", -46.6], [-23.6, -46.7]],
    [[-23.5, -46.6]],
])
def test_invalid_coordinates_are_rejected(coordinates):
    with pytest.raises(ValueError):
        prepare_features(np.asarray(coordinates))


def test_features_are_standardized(trained_pipeline):
    features, scaler, _ = trained_pipeline
    assert features.shape == (300, 2)
    assert np.allclose(features.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(features.std(axis=0), 1, atol=1e-12)
    assert scaler.n_features_in_ == 2


@pytest.mark.parametrize("n_clusters", [0, 1, 300, 301, 2.5, True])
def test_invalid_cluster_count_raises_error(trained_pipeline, n_clusters):
    features, _, _ = trained_pipeline
    with pytest.raises(ValueError, match="n_clusters"):
        train_model(features, n_clusters=n_clusters)


def test_model_metrics_match_reference(trained_pipeline):
    features, _, model = trained_pipeline
    metrics = evaluate_model(model, features)
    assert set(metrics) == {"inertia", "silhouette_score", "n_clusters"}
    assert metrics["n_clusters"] == 5
    assert metrics["inertia"] == pytest.approx(189.224, abs=0.0001)
    assert metrics["silhouette_score"] == pytest.approx(0.291, abs=0.0001)


def test_predictions_match_reference(trained_pipeline):
    _, scaler, model = trained_pipeline
    output = predict(model, scaler, lat=-23.55, lng=-46.63)
    assert set(output) == {"cluster", "distance_km"}
    assert output["cluster"] == 4
    assert output["distance_km"] == pytest.approx(12.3773, abs=0.0001)


@pytest.mark.parametrize("latitude,longitude", [
    (91, -46.6), (-91, -46.6), (0, 181), (0, -181),
    (np.nan, -46.6), (-23.5, np.inf), ("x", -46.6),
])
def test_invalid_prediction_coordinates_raise_error(trained_pipeline, latitude, longitude):
    _, scaler, model = trained_pipeline
    with pytest.raises(ValueError):
        predict(model, scaler, lat=latitude, lng=longitude)


def test_full_pipeline_from_another_working_directory(tmp_path: Path):
    """Verifica que main.py não depende do diretório atual do terminal."""
    command = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Pipeline executado com sucesso!" in command.stdout

    result = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    assert set(result) == {"metrics", "sample_prediction"}
    assert result["metrics"]["n_clusters"] == 5
    assert result["sample_prediction"]["cluster"] == 4
    assert not (tmp_path / "outputs" / "metrics.json").exists()


def test_metrics_json_is_versioned_and_complete():
    """O arquivo esperado pelo professor precisa existir já no repositório."""
    file = ROOT / "outputs" / "metrics.json"
    assert file.is_file()
    data = json.loads(file.read_text(encoding="utf-8"))
    assert set(data) == {"metrics", "sample_prediction"}
    assert set(data["metrics"]) == {"inertia", "silhouette_score", "n_clusters"}
    assert set(data["sample_prediction"]) == {"cluster", "distance_km"}
    assert isinstance(data["metrics"]["inertia"], float)
    assert isinstance(data["metrics"]["silhouette_score"], float)
    assert isinstance(data["metrics"]["n_clusters"], int)
    assert isinstance(data["sample_prediction"]["cluster"], int)
    assert isinstance(data["sample_prediction"]["distance_km"], float)
