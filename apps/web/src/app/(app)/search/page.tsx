"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { MusicCard } from "@/components/shared/music-card";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { searchCatalog } from "@/lib/api";

const categories = ["all", "songs", "artists", "albums"] as const;

export default function SearchPage() {
  const [query, setQuery] = useState("neon");
  const [category, setCategory] = useState<(typeof categories)[number]>("all");
  const { data, isLoading } = useQuery({
    queryKey: ["search", query, category],
    queryFn: () => searchCatalog(query, category),
  });

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      <Card className="bg-[#181818] p-8">
        <div className="space-y-4">
          <div>
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Search</div>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight">Search songs, artists, albums, genres</h1>
          </div>
          <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try: Neon, Workout, Dream-pop..." className="max-w-2xl" />
          <div className="flex flex-wrap gap-2">
            {categories.map((item) => (
              <Button
                key={item}
                variant={item === category ? "default" : "secondary"}
                size="sm"
                onClick={() => setCategory(item)}
              >
                {item}
              </Button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2 text-sm text-muted">
            {data?.history.map((item) => (
              <button key={item} onClick={() => setQuery(item)} className="rounded-full bg-white/5 px-3 py-1 hover:bg-white/10">
                {item}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {isLoading || !data ? (
        <div className="grid gap-4 md:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-52 animate-pulse rounded-3xl bg-white/5" />)}</div>
      ) : (
        <div className="space-y-8">
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold">Songs</h2>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {data.songs.map((song) => (
                <MusicCard key={song.id} song={song} />
              ))}
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <Card>
              <h3 className="text-xl font-semibold text-white">Artists</h3>
              <div className="mt-4 space-y-3">
                {data.artists.map((artist) => (
                  <div key={artist.id} className="rounded-2xl bg-white/[0.03] px-4 py-4">
                    <div className="font-medium text-white">{artist.name}</div>
                    <div className="text-sm text-muted">{artist.genres.join(" • ")}</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card>
              <h3 className="text-xl font-semibold text-white">Albums & Genres</h3>
              <div className="mt-4 space-y-3">
                {data.albums.map((album) => (
                  <div key={album.id} className="rounded-2xl bg-white/[0.03] px-4 py-4">
                    <div className="font-medium text-white">{album.title}</div>
                    <div className="text-sm text-muted">{album.genre} • {album.releaseDate}</div>
                  </div>
                ))}
                <div className="flex flex-wrap gap-2 pt-2">
                  {data.genres.map((genre) => (
                    <span key={genre} className="rounded-full bg-white/5 px-3 py-1 text-sm text-white">
                      {genre}
                    </span>
                  ))}
                </div>
              </div>
            </Card>
          </section>
        </div>
      )}
    </motion.div>
  );
}
