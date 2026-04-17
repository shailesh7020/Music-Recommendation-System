CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    profile_image TEXT,
    bio TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE artists (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    bio TEXT,
    image TEXT,
    monthly_listeners INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE albums (
    id UUID PRIMARY KEY,
    artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    cover_image TEXT,
    release_date DATE,
    genre VARCHAR(100)
);

CREATE TABLE songs (
    id UUID PRIMARY KEY,
    artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    album_id UUID REFERENCES albums(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    genre VARCHAR(100),
    duration_ms INTEGER NOT NULL,
    cover_image TEXT,
    audio_url TEXT,
    mood VARCHAR(50),
    popularity INTEGER NOT NULL DEFAULT 0,
    danceability REAL NOT NULL DEFAULT 0,
    energy REAL NOT NULL DEFAULT 0,
    valence REAL NOT NULL DEFAULT 0,
    acousticness REAL NOT NULL DEFAULT 0,
    tempo REAL NOT NULL DEFAULT 0
);

CREATE TABLE playlists (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    cover_image TEXT,
    visibility VARCHAR(16) NOT NULL DEFAULT 'public',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE playlist_songs (
    playlist_id UUID NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (playlist_id, song_id)
);

CREATE TABLE liked_songs (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    liked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, song_id)
);

CREATE TABLE listening_history (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    played_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    progress_seconds INTEGER NOT NULL DEFAULT 0
);
