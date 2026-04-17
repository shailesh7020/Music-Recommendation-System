export type Song = {
  id: string;
  title: string;
  artist: string;
  artistId: string;
  album: string;
  albumId: string;
  genre: string;
  durationMs: number;
  coverImage: string;
  audioUrl: string;
  mood: string;
  popularity: number;
  reason?: string;
  score?: number;
};

export type Artist = {
  id: string;
  name: string;
  bio: string;
  image: string;
  genres: string[];
  monthlyListeners: number;
};

export type Album = {
  id: string;
  artistId: string;
  title: string;
  coverImage: string;
  releaseDate: string;
  genre: string;
};

export type Playlist = {
  id: string;
  name: string;
  description: string;
  coverImage: string;
  userId: string;
  visibility: string;
  songs: string[];
};

export type UserProfile = {
  id: string;
  username: string;
  email: string;
  profileImage: string;
  bio: string;
};

export type Comment = {
  user: string;
  reaction: string;
  comment: string;
};

export type HomeResponse = {
  brand: string;
  recently_played: Song[];
  made_for_you: Song[];
  trending_now: Song[];
  top_artists: Artist[];
  new_releases: Album[];
  based_on_your_mood: Song[];
  recommended_playlists: Playlist[];
  liked_song_ids: string[];
};

export type SearchResponse = {
  songs: Song[];
  artists: Artist[];
  albums: Album[];
  genres: string[];
  history: string[];
};

export type SongDetailResponse = {
  song: Song;
  artist: Artist;
  album: Album;
  lyrics: string[];
  comments: Comment[];
  similar_songs: Song[];
  recommended_songs: Song[];
};

export type ArtistDetailResponse = {
  artist: Artist;
  popular_songs: Song[];
  albums: Album[];
  similar_artists: Artist[];
};

export type AlbumDetailResponse = {
  album: Album;
  artist: Artist;
  songs: Song[];
  total_duration_ms: number;
};

export type PlaylistDetailResponse = {
  playlist: Playlist;
  songs: Song[];
  total_duration_ms: number;
};

export type RecommendationPageResponse = {
  because_you_listened_to: {
    song: Song;
    results: Song[];
  };
  similar_users_like: Song[];
  mood_based: Record<string, Song[]>;
  genre_based: Record<string, Song[]>;
  trending_in_area: Song[];
  mood_taglines: Record<string, string>;
};
