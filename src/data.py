"""Funções de leitura e pré-processamento de dados geográficos."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _check_points(points: np.ndarray) -> np.ndarray:
    """Valida duas colunas numéricas de latitude e longitude."""
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordenadas precisam ser numéricas") from exc

    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError("São necessários pelo menos 2 pontos em um array (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError("Coordenadas não podem conter valores ausentes ou infinitos")
    if not np.all((-90 <= array[:, 0]) & (array[:, 0] <= 90)):
        raise ValueError("Latitude deve estar entre -90 e 90")
    if not np.all((-180 <= array[:, 1]) & (array[:, 1] <= 180)):
        raise ValueError("Longitude deve estar entre -180 e 180")
    return array


def load_data(filepath: str) -> np.ndarray:
    """Lê um CSV com colunas ``lat`` e ``lng`` e retorna pontos (n, 2).

    Args:
        filepath: Caminho do arquivo CSV.

    Returns:
        Array NumPy de coordenadas geográficas, em graus decimais.
    """
    file = Path(filepath)
    if not file.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {file}")
    frame = pd.read_csv(file)
    if not {"lat", "lng"}.issubset(frame.columns):
        raise ValueError("O CSV deve ter as colunas 'lat' e 'lng'")
    return _check_points(frame.loc[:, ["lat", "lng"]].to_numpy())


def prepare_features(points: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Padroniza latitude/longitude com StandardScaler.

    Args:
        points: Array de coordenadas em graus, com shape (n, 2).

    Returns:
        Tupla com os dados normalizados e o scaler ajustado.
    """
    array = _check_points(points)
    scaler = StandardScaler()
    features = scaler.fit_transform(array)
    return features, scaler
