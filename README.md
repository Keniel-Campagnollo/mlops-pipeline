# mlops-pipeline — Exercício de Modularização do Pipeline de ML

**Autor:** KenielCampagnollo  
**GitHub:** [Keniel-Campagnollo](https://github.com/Keniel-Campagnollo)  
**Atividade:** Avaliação de Desempenho / Modularização do Pipeline de ML

Este projeto foi integrado à estrutura **gerada pelo Cookiecutter oficial**
`cookiecutter gh:weslleymoura/mlops-template`, executado pelo próprio aluno.
O objetivo é completar os módulos `src/` fornecidos no template e executar um
pipeline de agrupamento de pontos geográficos usando **K-Means**.

## O que o projeto faz

1. Gera 300 pontos geográficos **sintéticos**, próximos de São Paulo (`generate_data.py`), usando a semente 42.
2. Lê as colunas `lat` e `lng` de um CSV (`load_data`).
3. Padroniza as duas variáveis com `StandardScaler` (`prepare_features`).
4. Ajusta `KMeans` para 5 clusters, com `random_state=42` (`train_model`).
5. Calcula inércia, Silhouette Score e número de clusters (`evaluate_model`).
6. Prevê o cluster do ponto fixo (-23.55, -46.63) e calcula a distância
   **geodésica** até seu centroide, após inverter a normalização (`predict`).
7. Salva o resultado em `outputs/metrics.json`, no formato fornecido pelo template.

Não são utilizados dados pessoais nem o conjunto LoggiBUD discutido em outras aulas.

## Estrutura

```text
mlops-pipeline/
├── data/
│   └── raw/
│       ├── .gitkeep
│       └── points.csv
├── outputs/
│   ├── .gitkeep
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── generate_data.py
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Executar no Windows (PowerShell)

Recomendado: **Python 3.13** (versão usada na validação), com ambiente virtual.

Abra a pasta `mlops-pipeline` no VS Code e execute:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe main.py
```

> Usar o executável da `.venv` diretamente dispensa ativar o ambiente no PowerShell.
> O `main.py` também gera o CSV automaticamente caso ainda não exista.
> Os caminhos são resolvidos a partir da localização do projeto.

## Testes automatizados

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes cobrem geração reproduzível, validação de coordenadas, arquivo CSV,
normalização, treinamento, métricas, previsão, integração e estrutura do JSON.

## Exemplo de saída (validada)

```json
{
  "metrics": {
    "inertia": 189.224,
    "silhouette_score": 0.291,
    "n_clusters": 5
  },
  "sample_prediction": {
    "cluster": 4,
    "distance_km": 12.3773
  }
}
```

**Interpretação:** a inércia mede a soma de distâncias quadráticas nos dados
padronizados; o Silhouette Score mede a coesão/separação dos clusters e não é
uma acurácia de classificação. Como os pontos são aleatórios, o valor 0,291
não implica erro do código. O rótulo `4` é um identificador sem ordenação.
A distância de 12,3773 km foi calculada por `geopy.distance.geodesic` com
coordenadas originais (não padronizadas).

## Contrato das funções do template

| Módulo | Assinatura original mantida | Retorno |
|---|---|---|
| `src/data.py` | `load_data(filepath)` | `numpy.ndarray` `(n, 2)` |
| `src/data.py` | `prepare_features(points)` | `(features, scaler)` |
| `src/train.py` | `train_model(features, n_clusters=5, random_state=42)` | KMeans treinado |
| `src/evaluate.py` | `evaluate_model(model, features)` | dict com as três métricas |
| `src/predict.py` | `predict(model, scaler, lat, lng)` | dict com cluster e distância em km |

A implementação mantém as assinaturas e o formato de saída previstos no
Cookiecutter enviado pelo aluno. As validações adicionadas procuram retornar
erros claros para dados inválidos, sem alterar os parâmetros do exemplo.

## Entrega no GitHub

O repositório deve ser **público** e se chamar exatamente `mlops-pipeline`,
na branch `main`, com `outputs/metrics.json` já publicado.

```powershell
git init
git add .
git commit -m "Implementa pipeline de ML modularizado"
git branch -M main
git remote add origin https://github.com/Keniel-Campagnollo/mlops-pipeline.git
git push -u origin main
```

Crie antes um repositório **vazio** no GitHub, sem README adicional.
Após o push, confira a URL pública:

`https://raw.githubusercontent.com/Keniel-Campagnollo/mlops-pipeline/main/outputs/metrics.json`

Se houver erro de autenticação ou de `git push`, confira o acesso à conta
GitHub sem expor credenciais no terminal ou nos arquivos. Não envie `.venv`,
segredos, tokens nem arquivos pessoais.
