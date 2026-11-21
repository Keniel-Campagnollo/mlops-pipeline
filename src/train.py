"""Treinamento do agrupador K-Means."""

import numpy as np
from sklearn.cluster import KMeans


def train_model(features: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> KMeans:
    """Ajusta um K-Means nas features e devolve o modelo treinado.

    Args:
        features: Matriz numérica padronizada (n, 2).
        n_clusters: Quantidade de agrupamentos (menor que n).
        random_state: Semente para treinamento reproduzível.

    Returns:
        Instância treinada de sklearn.cluster.KMeans.
    """
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("features deve ser uma matriz finita de shape (n, 2)")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters deve ser um número inteiro")
    if not 2 <= n_clusters < len(values):
        raise ValueError("n_clusters deve ser >= 2 e menor que o número de observações")
    model = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
    model.fit(values)
    return model
