# Tarea 2 IAA - Dataset Clickbait, Prensa Chilena e Internacional

Este proyecto arma el dataset pedido en `Tarea2-IAA.pdf` para la Entrega 1:

- Minimo 1.000 titulares de prensa chilena.
- Minimo 1.000 titulares de prensa extranjera.
- Curacion y etiquetado debil como `informativo` o `clickbait`.
- EDA por portal y autor.
- Bonus opcional: tercera clase `fake_news` con 1.000 claims/titulares desde fuentes de fact-checking.

La estructura esta separada en scripts para que el proceso sea demostrable en Canvas:

```text
scripts/01_web_scraping.py       # obtiene titulares nacionales e internacionales
scripts/02_classify_clickbait.py # limpia, deduplica, etiqueta y genera EDA
scripts/03_bonus_fake_news.py    # obtiene fake news y crea dataset multiclase
scripts/04_eda_entrega1.py       # regenera reportes EDA si cambia el dataset
src/                            # funciones reutilizables
data/raw/                       # salidas crudas
data/processed/                 # datasets curados
reports/                        # tablas para el informe
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Google Colab puedes usar:

```python
!pip install -r requirements.txt
```

## 1. Web scraping

```bash
python scripts/01_web_scraping.py --target-per-group 1000 --max-per-source 350 --delay 0.8
```

Salida principal:

```text
data/raw/headlines_raw.csv
```

El scraper usa tres niveles: feeds RSS/Atom, autodeteccion de feeds en la portada y lectura de sitemaps para llegar a mas titulares historicos. El `delay` ayuda a mantener scraping academico y respetuoso.

Para una prueba rapida:

```bash
python scripts/01_web_scraping.py --target-per-group 25 --max-per-source 20 --rss-only
```

## 2. Clasificacion y curacion

```bash
python scripts/02_classify_clickbait.py
```

Salidas:

```text
data/processed/dataset_clickbait.csv
data/processed/manual_review_candidates.csv
reports/clickbait_by_source.csv
reports/clickbait_by_author.csv
reports/eda_entrega1_summary.md
```

El etiquetado es debil y explicable: asigna `clickbait_score`, `label`, `clickbait_reasons` y `needs_review`. Para la entrega, conviene revisar manualmente los casos marcados en `manual_review_candidates.csv`, porque los patrones linguisticos no reemplazan la curacion humana.

## 3. Bonus Fake News

```bash
python scripts/03_bonus_fake_news.py --target-fake 1000 --max-per-source 300 --delay 0.8
```

Salidas:

```text
data/raw/fake_news_headlines.csv
data/processed/dataset_multiclase_bonus.csv
```

El bonus busca `ClaimReview` en sitios de fact-checking. Cuando encuentra una afirmacion revisada, usa el campo `claimReviewed` como titular/claim y conserva el rating en `fact_check_rating`.

Para hacerlo mas estricto:

```bash
python scripts/03_bonus_fake_news.py --strict-rating
```

## Columnas principales

- `headline`: titular o claim.
- `label`: `informativo`, `clickbait` o `fake_news`.
- `source_type`: `nacional`, `internacional` o `fake_news`.
- `source`: portal.
- `author`: autor si el sitio lo publica.
- `published_at`: fecha detectada.
- `url`: trazabilidad de la fuente.
- `clickbait_score`: score de reglas entre 0 y 1.
- `clickbait_reasons`: señales usadas por el clasificador debil.
- `needs_review`: casos cercanos al umbral que conviene revisar manualmente.
- `split`: particion inicial para modelado posterior.

## Relacion con el notebook de ayudantia

`IAA_collab_ayun04_2026.ipynb` sirve como referencia para la etapa de modelado y XAI: carga de datos con `pandas`, comparacion de metricas, uso de Transformers, LIME, `transformers_interpret` y SHAP. Este repositorio prepara el insumo que despues se puede usar en esa logica.

