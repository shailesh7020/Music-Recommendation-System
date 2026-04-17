# Pulsewave Monorepo

Pulsewave is a full-stack music recommendation platform inspired by modern streaming products. It keeps the original Python recommendation work in the repository, but now adds a production-style monorepo with:

- `apps/web`: Next.js + React + Tailwind + Framer Motion + Zustand + React Query
- `apps/api`: FastAPI backend with auth, catalog, playlists, and hybrid recommendations
- `apps/api/db/schema.sql`: PostgreSQL schema for a persistent production database
- `docker-compose.yml`: local orchestration for web, API, Postgres, and Redis

The legacy Streamlit prototype still exists in `music_recommender_hybrid_app.py`, but the new primary product direction is the `apps/web` + `apps/api` stack.

## Product Highlights

- Spotify-inspired dark UI with a branded Pulsewave visual system
- Landing page, auth screens, home, search, recommendations, song, artist, album, playlist, library, and liked songs pages
- Persistent bottom player with queue, seek, volume, shuffle, repeat, and keyboard shortcuts
- Hybrid recommendation engine covering content-based, collaborative, mood-based, and genre-based strategies
- Playlist CRUD endpoints, JWT auth scaffolding, sample content, and responsive layout patterns

## Repository Layout

```text
apps/
  api/
    app/
      api/routes/
      core/
      data/
      models/
      services/
    db/schema.sql
    Dockerfile
    requirements.txt
  web/
    src/app/
    src/components/
    src/lib/
    src/providers/
    src/store/
    Dockerfile
music_recommender/
tests/
docker-compose.yml
```

## Backend APIs

Main routes live under `/api`.

- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/catalog/home`
- `GET /api/catalog/search?q=...`
- `GET /api/catalog/songs/{song_id}`
- `GET /api/catalog/artists/{artist_id}`
- `GET /api/catalog/albums/{album_id}`
- `GET /api/catalog/playlists/{playlist_id}`
- `GET /api/recommendations`
- `GET /api/recommendations/songs/{song_id}/similar`
- `GET /api/recommendations/moods/{mood}`
- `GET /api/recommendations/genres/{genre}`
- `POST /api/playlists`
- `PATCH /api/playlists/{playlist_id}`
- `POST /api/playlists/{playlist_id}/songs/{song_id}`
- `POST /api/playlists/{playlist_id}/reorder`
- `POST /api/playlists/{playlist_id}/duplicate`

## Local Development

### API

```powershell
py -3 -m pip install -r .\apps\api\requirements.txt
py -3 -m uvicorn apps.api.app.main:app --reload
```

### Web

```powershell
cd .\apps\web
npm install
npm run dev
```

### Docker

```powershell
docker compose up --build
```

## Recommendation Engine

The backend recommendation engine combines:

1. Content-based similarity using audio feature vectors and metadata overlap
2. Collaborative filtering from shared listener profiles
3. Mood-based and genre-based ranking overlays
4. Hybrid scoring with human-readable reasons for why each song appears

The core logic lives in [recommendation_service.py](</E:/Music-Recommendation-System/apps/api/app/services/recommendation_service.py:1>).

## Tests

Verified locally in this environment:

```powershell
py -3 -m unittest discover -s tests -v
```

That covers the original recommender tests plus the new backend recommendation and token helpers.

## Notes

- This workspace did not have Node.js available during implementation, so the Next.js app was scaffolded but not executed locally here.
- The FastAPI routes are scaffolded for production-style usage, but because FastAPI dependencies were not installed in this workspace session, route boot verification was limited to static/syntax checks and service-layer tests.
- The sample data in `apps/api/app/data/sample_seed.py` is designed to make the full product demonstrable immediately.
