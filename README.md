# Tarea 2 IAA - Dataset Clickbait, Prensa Chilena e Internacional

Este proyecto arma el dataset pedido en `Tarea2-IAA.pdf` para la Entrega 1:

- Minimo 1.000 titulares de prensa chilena.
- Minimo 1.000 titulares de prensa extranjera.
- Curacion y etiquetado debil como `informativo` o `clickbait`.
- Dataset final con 3.000 titulares: 1.500 nacionales y 1.500 internacionales. La seleccion automatica conserva 250 candidatos `clickbait`; despues de la revision manual quedan 231 `clickbait` y 2.769 `informativo`.
- EDA por portal y autor.
- Bonus opcional: tercera clase `fake_news` con 1.000 claims/titulares desde fuentes de fact-checking y The Onion como fuente satirica.

La estructura esta separada en scripts para que el proceso sea demostrable en Canvas:

```text
scripts/01_web_scraping.py       # obtiene titulares nacionales e internacionales
scripts/02_classify_clickbait.py # limpia, deduplica, etiqueta y genera EDA
scripts/03_bonus_fake_news.py    # obtiene fake news y crea dataset multiclase
scripts/04_eda_entrega1.py       # regenera reportes EDA si cambia el dataset
scripts/05_translate_headlines_es.py # traduce headline al espanol y conserva headline_original
scripts/06_apply_manual_review.py # aplica labels revisados manualmente al dataset base
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
python scripts/01_web_scraping.py --target-per-group 12000 --max-per-source 1200 --delay 0 --no-article-pages
```

Salida principal:

```text
data/raw/headlines_raw.csv
```

El scraper usa tres niveles: feeds RSS/Atom, autodeteccion de feeds en la portada y lectura de sitemaps para llegar a mas titulares historicos. El modo `--no-article-pages` evita descargar cada pagina completa y permite juntar suficientes candidatos para conservar el umbral `0.35`. Si se omite ese modo, el scraper intenta leer metadatos desde cada articulo, pero puede tardar bastante mas.

Para una prueba rapida:

```bash
python scripts/01_web_scraping.py --target-per-group 25 --max-per-source 20 --rss-only
```

## 2. Clasificacion y curacion

```bash
python scripts/02_classify_clickbait.py
```

Por defecto, este paso usa `--threshold 0.35` y genera exactamente 3.000 filas: 1.500 titulares nacionales, 1.500 titulares internacionales, 250 candidatos `clickbait` con `clickbait_score >= threshold` y 2.750 casos `informativo`.

Si necesitas ajustar la distribucion:

```bash
python scripts/02_classify_clickbait.py --target-per-source-type 1500 --target-clickbait 250 --threshold 0.35
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

Para aplicar la revision manual guardada en `data/hand_processed/manual_review_candidates_REVIEWED.csv`:

```bash
python scripts/06_apply_manual_review.py
python scripts/04_eda_entrega1.py
```

En la version actual, la revision manual deja 231 titulares `clickbait` y 2.769 `informativo` en el dataset base. Los 250 casos revisados quedan marcados con `labeling_method = manual-review-v1.0`.

## 3. Bonus Fake News

```bash
python scripts/03_bonus_fake_news.py --target-fake 1000 --max-per-source 400 --delay 0 --no-article-pages
```

Salidas:

```text
data/raw/fake_news_headlines.csv
data/processed/dataset_multiclase_bonus.csv
```

El bonus mezcla fuentes de fact-checking con una fuente satirica: The Onion. Con `--no-article-pages`, usa RSS y sitemaps para acelerar la recoleccion; si una fuente expone `ClaimReview`, puede conservar el rating en `fact_check_rating`. The Onion queda marcada como `satire` y con `labeling_method = satire-source-v1.0`.

Para hacerlo mas estricto, omite `--no-article-pages` y usa:

```bash
python scripts/03_bonus_fake_news.py --strict-rating
```

## 4. Titulares en espanol

```bash
python scripts/05_translate_headlines_es.py
```

Este paso deja `headline` en espanol para los datasets principales y conserva el texto original en `headline_original`. La URL y la noticia fuente no se modifican; solo se traduce el titular usado como entrada textual para el modelado.

## Columnas principales

- `headline`: titular o claim en espanol.
- `headline_original`: titular o claim original antes de traducir.
- `label`: `informativo` (noticia real), `clickbait` o `fake_news`.
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
