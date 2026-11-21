"""Inferência de cluster e distância geodésica ao centroide."""

import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def predict(model: KMeans, scaler: StandardScaler, lat: float, lng: float) -> dict:
    """Prevê o cluster de um ponto e calcula sua distância ao centroide.

    Args:
        model: KMeans ajustado nas coordenadas padronizadas.
        scaler: StandardScaler utilizado durante o treinamento.
        lat: Latitude do ponto, em graus decimais.
        lng: Longitude do ponto, em graus decimais.

    Returns:
        Dicionário com ``cluster`` (inteiro) e ``distance_km`` (float).
    """
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError) as exc:
        raise ValueError("Latitude e longitude precisam ser numéricas") from exc
    if not np.isfinite([lat, lng]).all() or not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        raise ValueError("Latitude ou longitude inválida")

    point_scaled = scaler.transform(np.array([[lat, lng]]))
    cluster = int(model.predict(point_scaled)[0])
    centroid_scaled = model.cluster_centers_[cluster].reshape(1, -1)
    centroid = scaler.inverse_transform(centroid_scaled)[0]
    distance_km = geodesic((lat, lng), (float(centroid[0]), float(centroid[1]))).km
    return {"cluster": cluster, "distance_km": round(float(distance_km), 4)}
