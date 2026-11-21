"""Gera dados sintéticos de coordenadas para o exercício."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_data(output_path: str = "data/raw/points.csv", n: int = 300) -> Path:
    """Gera n coordenadas sintéticas próximas de São Paulo, com semente fixa.

    Parâmetros:
        output_path: Arquivo CSV de destino.
        n: Quantidade de pontos a gerar.

    Retorna:
        Path do arquivo salvo, com colunas lat e lng.
    """
    if n < 1:
        raise ValueError("n deve ser positivo")

    # Gerador compatível com o exemplo monolítico apresentado na atividade.
    rng = np.random.RandomState(42)
    lats = rng.normal(-23.55, 0.15, n)
    lngs = rng.normal(-46.63, 0.15, n)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lat": lats, "lng": lngs}).to_csv(destination, index=False)
    return destination


if __name__ == "__main__":
    path = generate_data()
    print(f"Dados gerados: {path}")
