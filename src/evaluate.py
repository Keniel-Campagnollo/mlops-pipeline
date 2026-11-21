"""Métricas do agrupamento K-Means."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_model(model: KMeans, features: np.ndarray) -> dict:
    """Calcula inércia, Silhouette Score e quantidade de clusters.

    Args:
        model: Modelo KMeans treinado.
        features: Mesma matriz de dados utilizada no ajuste.

    Returns:
        Dicionário com ``inertia``, ``silhouette_score`` e ``n_clusters``.
    """
    values = np.asarray(features, dtype=float)
    labels = model.labels_
    if values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("As features devem corresponder às observações do treinamento")
    unique_clusters = np.unique(labels)
    if not 2 <= len(unique_clusters) < len(values):
        raise ValueError("Silhouette Score exige de 2 até n-1 grupos distintos")

    return {
        "inertia": round(float(model.inertia_), 4),
        "silhouette_score": round(float(silhouette_score(values, labels)), 4),
        "n_clusters": int(model.n_clusters),
    }
