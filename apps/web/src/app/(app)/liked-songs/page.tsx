"use client";

import { useQuery } from "@tanstack/react-query";

import { MusicCard } from "@/components/shared/music-card";
import { getHome } from "@/lib/api";

export default function LikedSongsPage() {
  const { data } = useQuery({ queryKey: ["liked-songs"], queryFn: getHome });

  if (!data) return <div className="h-64 animate-pulse rounded-3xl bg-white/5" />;

  const likedSongs = [...data.recently_played, ...data.trending_now].filter((song) => data.liked_song_ids.includes(song.id));

  return (
    <div className="space-y-6">
      <div>
        <div className="text-xs uppercase tracking-[0.35em] text-accent">Collection</div>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">Liked Songs</h1>
        <p className="mt-2 text-muted">A clean, premium shelf for the tracks you’ve explicitly kept close.</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {likedSongs.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </div>
    </div>
  );
}
