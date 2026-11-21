"""Contrato do template Cookiecutter: assinaturas e pipeline de saída."""

import inspect

from src.data import load_data, prepare_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict


def test_expected_public_signatures():
    expected = {
        load_data: ["filepath"],
        prepare_features: ["points"],
        train_model: ["features", "n_clusters", "random_state"],
        evaluate_model: ["model", "features"],
        predict: ["model", "scaler", "lat", "lng"],
    }
    for function, parameters in expected.items():
        assert list(inspect.signature(function).parameters) == parameters


def test_training_defaults():
    signature = inspect.signature(train_model)
    assert signature.parameters["n_clusters"].default == 5
    assert signature.parameters["random_state"].default == 42
