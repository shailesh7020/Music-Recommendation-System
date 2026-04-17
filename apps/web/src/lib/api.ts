import {
  buildHomeResponse,
  demoUser,
  getAlbumDetail,
  getArtistDetail,
  getPlaylistDetail,
  getRecommendationPage,
  getSongDetail,
  searchAll,
} from "@/lib/mock-data";
import type {
  AlbumDetailResponse,
  ArtistDetailResponse,
  HomeResponse,
  PlaylistDetailResponse,
  RecommendationPageResponse,
  SearchResponse,
  SongDetailResponse,
  UserProfile,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function fetchJson<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Request failed for ${path}`);
    }
    return (await response.json()) as T;
  } catch {
    return fallback();
  }
}

export function getCurrentUser(): Promise<UserProfile> {
  return Promise.resolve(demoUser);
}

export function getHome(): Promise<HomeResponse> {
  return fetchJson("/catalog/home", buildHomeResponse);
}

export function searchCatalog(query: string, category?: string): Promise<SearchResponse> {
  const searchParams = new URLSearchParams({ q: query });
  if (category && category !== "all") {
    searchParams.set("category", category);
  }
  return fetchJson(`/catalog/search?${searchParams.toString()}`, () => searchAll(query, category));
}

export function getSong(songId: string): Promise<SongDetailResponse> {
  return fetchJson(`/catalog/songs/${songId}`, () => getSongDetail(songId));
}

export function getArtist(artistId: string): Promise<ArtistDetailResponse> {
  return fetchJson(`/catalog/artists/${artistId}`, () => getArtistDetail(artistId));
}

export function getAlbum(albumId: string): Promise<AlbumDetailResponse> {
  return fetchJson(`/catalog/albums/${albumId}`, () => getAlbumDetail(albumId));
}

export function getPlaylist(playlistId: string): Promise<PlaylistDetailResponse> {
  return fetchJson(`/catalog/playlists/${playlistId}`, () => getPlaylistDetail(playlistId));
}

export function getRecommendations(): Promise<RecommendationPageResponse> {
  return fetchJson("/recommendations", getRecommendationPage);
}
