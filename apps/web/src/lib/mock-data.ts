import type {
  Album,
  AlbumDetailResponse,
  Artist,
  ArtistDetailResponse,
  HomeResponse,
  Playlist,
  PlaylistDetailResponse,
  RecommendationPageResponse,
  SearchResponse,
  Song,
  SongDetailResponse,
  UserProfile,
} from "@/lib/types";

export const demoUser: UserProfile = {
  id: "user-ava",
  username: "ava",
  email: "ava@pulsewave.fm",
  profileImage: "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
  bio: "Night-drive collector and synth-pop obsessive.",
};

export const artists: Artist[] = [
  {
    id: "artist-aurora-neon",
    name: "Aurora Neon",
    bio: "Synth-pop producer known for skyline choruses and midnight energy.",
    image: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e",
    genres: ["Synthwave", "Synth-pop"],
    monthlyListeners: 2840000,
  },
  {
    id: "artist-velvet-atlas",
    name: "Velvet Atlas",
    bio: "A duo balancing glossy hooks with club-ready percussion.",
    image: "https://images.unsplash.com/photo-1521572267360-ee0c2909d518",
    genres: ["Dance-pop", "Electro-pop"],
    monthlyListeners: 1910000,
  },
  {
    id: "artist-luna-harbor",
    name: "Luna Harbor",
    bio: "Dream-pop songwriter with cinematic late-night arrangements.",
    image: "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df",
    genres: ["Dream-pop", "Indie-pop"],
    monthlyListeners: 1320000,
  },
  {
    id: "artist-static-bloom",
    name: "Static Bloom",
    bio: "Low-lit electronics for deep work and soft glitches.",
    image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d",
    genres: ["Lo-fi", "Alt-electronic"],
    monthlyListeners: 980000,
  },
  {
    id: "artist-midnight-circuit",
    name: "Midnight Circuit",
    bio: "Retro-futurist producer making romantic synths and after-hours grooves.",
    image: "https://images.unsplash.com/photo-1517841905240-472988babdf9",
    genres: ["Synthwave", "Nu-disco"],
    monthlyListeners: 1560000,
  },
  {
    id: "artist-solis-runclub",
    name: "Solis Runclub",
    bio: "House duo with sunrise tempos and bright, kinetic drops.",
    image: "https://images.unsplash.com/photo-1504593811423-6dd665756598",
    genres: ["House", "Dance"],
    monthlyListeners: 2100000,
  },
];

export const albums: Album[] = [
  { id: "album-city-afterglow", artistId: "artist-aurora-neon", title: "City Afterglow", coverImage: "/covers/city-afterglow.jpg", releaseDate: "2025-02-14", genre: "Synth-pop" },
  { id: "album-velvet-drive", artistId: "artist-velvet-atlas", title: "Velvet Drive", coverImage: "/covers/velvet-drive.jpg", releaseDate: "2025-05-09", genre: "Dance-pop" },
  { id: "album-moonwater", artistId: "artist-luna-harbor", title: "Moonwater", coverImage: "/covers/moonwater.jpg", releaseDate: "2024-11-02", genre: "Dream-pop" },
  { id: "album-slow-signal", artistId: "artist-static-bloom", title: "Slow Signal", coverImage: "/covers/slow-signal.jpg", releaseDate: "2025-01-22", genre: "Alt-electronic" },
  { id: "album-midnight-protocol", artistId: "artist-midnight-circuit", title: "Midnight Protocol", coverImage: "/covers/midnight-protocol.jpg", releaseDate: "2024-08-30", genre: "Synthwave" },
  { id: "album-electric-sun", artistId: "artist-solis-runclub", title: "Electric Sun", coverImage: "/covers/electric-sun.jpg", releaseDate: "2025-06-01", genre: "House" },
];

export const songs: Song[] = [
  { id: "song-neon-tide", title: "Neon Tide", artist: "Aurora Neon", artistId: "artist-aurora-neon", album: "City Afterglow", albumId: "album-city-afterglow", genre: "Synthwave", durationMs: 218000, coverImage: "/covers/city-afterglow.jpg", audioUrl: "https://example.com/audio/neon-tide.mp3", mood: "Chill", popularity: 92 },
  { id: "song-glass-sky", title: "Glass Sky", artist: "Aurora Neon", artistId: "artist-aurora-neon", album: "City Afterglow", albumId: "album-city-afterglow", genre: "Synth-pop", durationMs: 205000, coverImage: "/covers/city-afterglow.jpg", audioUrl: "https://example.com/audio/glass-sky.mp3", mood: "Focus", popularity: 88 },
  { id: "song-velvet-rush", title: "Velvet Rush", artist: "Velvet Atlas", artistId: "artist-velvet-atlas", album: "Velvet Drive", albumId: "album-velvet-drive", genre: "Dance-pop", durationMs: 198000, coverImage: "/covers/velvet-drive.jpg", audioUrl: "https://example.com/audio/velvet-rush.mp3", mood: "Party", popularity: 95 },
  { id: "song-after-midnight-signal", title: "After Midnight Signal", artist: "Velvet Atlas", artistId: "artist-velvet-atlas", album: "Velvet Drive", albumId: "album-velvet-drive", genre: "Electro-pop", durationMs: 223000, coverImage: "/covers/velvet-drive.jpg", audioUrl: "https://example.com/audio/after-midnight-signal.mp3", mood: "Romantic", popularity: 84 },
  { id: "song-harbor-lights", title: "Harbor Lights", artist: "Luna Harbor", artistId: "artist-luna-harbor", album: "Moonwater", albumId: "album-moonwater", genre: "Indie-pop", durationMs: 231000, coverImage: "/covers/moonwater.jpg", audioUrl: "https://example.com/audio/harbor-lights.mp3", mood: "Chill", popularity: 82 },
  { id: "song-blue-hour-loop", title: "Blue Hour Loop", artist: "Luna Harbor", artistId: "artist-luna-harbor", album: "Moonwater", albumId: "album-moonwater", genre: "Dream-pop", durationMs: 244000, coverImage: "/covers/moonwater.jpg", audioUrl: "https://example.com/audio/blue-hour-loop.mp3", mood: "Sad", popularity: 79 },
  { id: "song-static-heartbeat", title: "Static Heartbeat", artist: "Static Bloom", artistId: "artist-static-bloom", album: "Slow Signal", albumId: "album-slow-signal", genre: "Alt-electronic", durationMs: 214000, coverImage: "/covers/slow-signal.jpg", audioUrl: "https://example.com/audio/static-heartbeat.mp3", mood: "Workout", popularity: 76 },
  { id: "song-lowlight-code", title: "Lowlight Code", artist: "Static Bloom", artistId: "artist-static-bloom", album: "Slow Signal", albumId: "album-slow-signal", genre: "Lo-fi", durationMs: 196000, coverImage: "/covers/slow-signal.jpg", audioUrl: "https://example.com/audio/lowlight-code.mp3", mood: "Focus", popularity: 90 },
  { id: "song-satellite-kiss", title: "Satellite Kiss", artist: "Midnight Circuit", artistId: "artist-midnight-circuit", album: "Midnight Protocol", albumId: "album-midnight-protocol", genre: "Synthwave", durationMs: 227000, coverImage: "/covers/midnight-protocol.jpg", audioUrl: "https://example.com/audio/satellite-kiss.mp3", mood: "Romantic", popularity: 87 },
  { id: "song-dusk-driver", title: "Dusk Driver", artist: "Midnight Circuit", artistId: "artist-midnight-circuit", album: "Midnight Protocol", albumId: "album-midnight-protocol", genre: "Nu-disco", durationMs: 212000, coverImage: "/covers/midnight-protocol.jpg", audioUrl: "https://example.com/audio/dusk-driver.mp3", mood: "Party", popularity: 91 },
  { id: "song-sunrise-sprinter", title: "Sunrise Sprinter", artist: "Solis Runclub", artistId: "artist-solis-runclub", album: "Electric Sun", albumId: "album-electric-sun", genre: "House", durationMs: 201000, coverImage: "/covers/electric-sun.jpg", audioUrl: "https://example.com/audio/sunrise-sprinter.mp3", mood: "Workout", popularity: 93 },
  { id: "song-golden-mile", title: "Golden Mile", artist: "Solis Runclub", artistId: "artist-solis-runclub", album: "Electric Sun", albumId: "album-electric-sun", genre: "Dance", durationMs: 208000, coverImage: "/covers/electric-sun.jpg", audioUrl: "https://example.com/audio/golden-mile.mp3", mood: "Happy", popularity: 89 }
];

export const playlists: Playlist[] = [
  { id: "playlist-night-drive", name: "Night Drive Protocol", description: "Retro neon, skyline synths, and cinematic motion.", coverImage: "/covers/night-drive.jpg", userId: "user-ava", visibility: "public", songs: ["song-neon-tide", "song-satellite-kiss", "song-dusk-driver", "song-after-midnight-signal"] },
  { id: "playlist-focus-after-dark", name: "Focus After Dark", description: "Low-light electronics and clear-headed nocturnal loops.", coverImage: "/covers/focus-after-dark.jpg", userId: "user-leo", visibility: "private", songs: ["song-lowlight-code", "song-glass-sky", "song-blue-hour-loop", "song-harbor-lights"] },
  { id: "playlist-sunrise-run", name: "Sunrise Run Club", description: "House and kinetic pop for early miles.", coverImage: "/covers/sunrise-run.jpg", userId: "user-maya", visibility: "public", songs: ["song-sunrise-sprinter", "song-golden-mile", "song-static-heartbeat", "song-velvet-rush"] }
];

const songComments = {
  "song-neon-tide": [
    { user: "ava", reaction: "fire", comment: "Perfect city-lights energy." },
    { user: "maya", reaction: "sparkles", comment: "This belongs on every night run playlist." }
  ],
  "song-lowlight-code": [{ user: "leo", reaction: "brain", comment: "My default writing loop." }]
};

const moodTaglines: Record<string, string> = {
  Happy: "Brightness, lift, and forward motion.",
  Sad: "Soft landings, reflective vocals, and room to breathe.",
  Chill: "Low-glow tracks for drifting and decompressing.",
  Workout: "High-BPM fuel with momentum baked in.",
  Party: "Peak-hour energy and glossy hooks.",
  Romantic: "Warm synths and close-focus melodies.",
  Focus: "Clean textures for deep work and flow."
};

const byId = <T extends { id: string }>(items: T[], id: string) => items.find((item) => item.id === id)!;
const topSongs = [...songs].sort((a, b) => b.popularity - a.popularity);

export function buildHomeResponse(): HomeResponse {
  return {
    brand: "Pulsewave",
    recently_played: songs.slice(0, 6),
    made_for_you: [songs[0], songs[7], songs[8], songs[4], songs[10], songs[11]],
    trending_now: topSongs.slice(0, 6),
    top_artists: [...artists].sort((a, b) => b.monthlyListeners - a.monthlyListeners).slice(0, 5),
    new_releases: [...albums].sort((a, b) => b.releaseDate.localeCompare(a.releaseDate)).slice(0, 5),
    based_on_your_mood: [songs[7], songs[1], songs[4], songs[5], songs[8], songs[0]],
    recommended_playlists: playlists,
    liked_song_ids: ["song-neon-tide", "song-glass-sky", "song-harbor-lights", "song-satellite-kiss"]
  };
}

export function searchAll(query: string, category: string = "all"): SearchResponse {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return {
      songs: [],
      artists: [],
      albums: [],
      genres: ["Synthwave", "Focus", "Workout", "Dream-pop", "House"],
      history: ["Neon Tide", "Workout", "Luna Harbor"]
    };
  }

  const songMatches = songs.filter((song) => [song.title, song.artist, song.genre, song.mood].some((value) => value.toLowerCase().includes(normalized))).slice(0, 8);
  const artistMatches = artists.filter((artist) => [artist.name, ...artist.genres].some((value) => value.toLowerCase().includes(normalized))).slice(0, 6);
  const albumMatches = albums.filter((album) => [album.title, album.genre].some((value) => value.toLowerCase().includes(normalized))).slice(0, 6);

  return {
    songs: category === "artists" || category === "albums" ? [] : songMatches,
    artists: category === "songs" || category === "albums" ? [] : artistMatches,
    albums: category === "songs" || category === "artists" ? [] : albumMatches,
    genres: Array.from(new Set(songs.map((song) => song.genre).filter((genre) => genre.toLowerCase().includes(normalized)))).slice(0, 6),
    history: ["Neon Tide", "Luna Harbor", "Workout"]
  };
}

export function getSongDetail(songId: string): SongDetailResponse {
  const song = byId(songs, songId);
  const artist = byId(artists, song.artistId);
  const album = byId(albums, song.albumId);
  const related = songs.filter((candidate) => candidate.id !== song.id && (candidate.genre === song.genre || candidate.mood === song.mood)).slice(0, 6);
  return {
    song,
    artist,
    album,
    lyrics: [
      "Streetlights bloom across the glass tonight",
      "We move in color, coded in neon light",
      "Every skyline pulse becomes a melody"
    ],
    comments: songComments[songId as keyof typeof songComments] ?? [],
    similar_songs: related,
    recommended_songs: topSongs.filter((candidate) => candidate.id !== song.id).slice(0, 6)
  };
}

export function getArtistDetail(artistId: string): ArtistDetailResponse {
  return {
    artist: byId(artists, artistId),
    popular_songs: songs.filter((song) => song.artistId === artistId).sort((a, b) => b.popularity - a.popularity),
    albums: albums.filter((album) => album.artistId === artistId),
    similar_artists: artists.filter((artist) => artist.id !== artistId).slice(0, 4)
  };
}

export function getAlbumDetail(albumId: string): AlbumDetailResponse {
  const album = byId(albums, albumId);
  const albumSongs = songs.filter((song) => song.albumId === albumId);
  return {
    album,
    artist: byId(artists, album.artistId),
    songs: albumSongs,
    total_duration_ms: albumSongs.reduce((sum, song) => sum + song.durationMs, 0)
  };
}

export function getPlaylistDetail(playlistId: string): PlaylistDetailResponse {
  const playlist = byId(playlists, playlistId);
  const playlistSongs = playlist.songs.map((songId) => byId(songs, songId));
  return {
    playlist,
    songs: playlistSongs,
    total_duration_ms: playlistSongs.reduce((sum, song) => sum + song.durationMs, 0)
  };
}

export function getRecommendationPage(): RecommendationPageResponse {
  return {
    because_you_listened_to: {
      song: byId(songs, "song-neon-tide"),
      results: [byId(songs, "song-satellite-kiss"), byId(songs, "song-glass-sky"), byId(songs, "song-dusk-driver"), byId(songs, "song-harbor-lights"), byId(songs, "song-lowlight-code")]
    },
    similar_users_like: [byId(songs, "song-lowlight-code"), byId(songs, "song-sunrise-sprinter"), byId(songs, "song-velvet-rush"), byId(songs, "song-blue-hour-loop"), byId(songs, "song-golden-mile")],
    mood_based: {
      Happy: [byId(songs, "song-golden-mile")],
      Sad: [byId(songs, "song-blue-hour-loop")],
      Chill: [byId(songs, "song-neon-tide"), byId(songs, "song-harbor-lights")],
      Workout: [byId(songs, "song-sunrise-sprinter"), byId(songs, "song-static-heartbeat")],
      Party: [byId(songs, "song-velvet-rush"), byId(songs, "song-dusk-driver")],
      Romantic: [byId(songs, "song-satellite-kiss"), byId(songs, "song-after-midnight-signal")],
      Focus: [byId(songs, "song-lowlight-code"), byId(songs, "song-glass-sky")]
    },
    genre_based: {
      Synthwave: [byId(songs, "song-neon-tide"), byId(songs, "song-satellite-kiss")],
      "Dance-pop": [byId(songs, "song-velvet-rush")],
      "Dream-pop": [byId(songs, "song-blue-hour-loop")],
      "Lo-fi": [byId(songs, "song-lowlight-code")],
      House: [byId(songs, "song-sunrise-sprinter")]
    },
    trending_in_area: [byId(songs, "song-lowlight-code"), byId(songs, "song-neon-tide"), byId(songs, "song-sunrise-sprinter")],
    mood_taglines: moodTaglines
  };
}
