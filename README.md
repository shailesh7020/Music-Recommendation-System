# Music Recommendation System

This repository now has a production-ready foundation for a hybrid music recommender. The Streamlit app still provides the user interface, but the recommendation logic, artifact loading, validation, and health checks now live in a reusable Python package.

## What changed

- Recommendation logic moved out of the Streamlit script into `music_recommender/`.
- Artifact loading now validates file presence, row alignment, and collaborative-filtering keys.
- Data normalization is explicit instead of being spread across the UI file.
- Automated tests cover search, mood filtering, recommendation ranking, and artifact validation.
- Project metadata and setup now live in `pyproject.toml`.

## Project layout

```text
music_recommender/
  artifacts.py
  healthcheck.py
  log.py
  recommender.py
  service.py
music_recommender_hybrid_app.py
tests/
```

## Run locally

```powershell
py -3 -m pip install -e ".[dev]"
streamlit run .\music_recommender_hybrid_app.py
```

## Health check

Use this before deployment to confirm the serialized artifacts are present and internally consistent.

```powershell
py -3 -m music_recommender.healthcheck --base-dir .
```

## Tests

```powershell
py -3 -m pytest -q
```

If you prefer the standard-library test runner:

```powershell
py -3 -m unittest discover -s tests -v
```

## Industrialization roadmap

This refactor improves the codebase substantially, but a truly industrial deployment would still benefit from:

1. CI/CD for tests, linting, packaging, and deployment.
2. Artifact versioning and a dedicated data/model registry instead of large binaries in the repo root.
3. Observability for latency, error rates, recommendation quality, and user behavior.
4. A service boundary for online inference if the app needs to support multiple clients.
5. Security and secrets management for any future Spotify or backend integrations.
