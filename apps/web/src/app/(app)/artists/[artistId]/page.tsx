"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { MusicCard } from "@/components/shared/music-card";
import { SectionCarousel } from "@/components/shared/section-carousel";
import { TrackTable } from "@/components/shared/track-table";
import { Card } from "@/components/ui/card";
import { getArtist } from "@/lib/api";

export default function ArtistPage() {
  const params = useParams<{ artistId: string }>();
  const artistId = String(params.artistId);
  const { data, isLoading } = useQuery({ queryKey: ["artist", artistId], queryFn: () => getArtist(artistId) });

  if (isLoading || !data) return <div className="h-72 animate-pulse rounded-3xl bg-white/5" />;

  return (
    <div className="space-y-8">
      <Card className="overflow-hidden bg-gradient-to-r from-cyan-500/20 via-transparent to-emerald-500/20 p-8">
        <div className="grid gap-8 lg:grid-cols-[220px,1fr]">
          <div className="aspect-square rounded-full bg-gradient-to-br from-cyan-500 via-sky-500 to-blue-400 shadow-glow" />
          <div className="flex flex-col justify-end gap-3">
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Artist</div>
            <h1 className="text-5xl font-semibold tracking-tight">{data.artist.name}</h1>
            <p className="max-w-3xl text-muted">{data.artist.bio}</p>
            <div className="text-sm text-white/80">{data.artist.monthlyListeners.toLocaleString()} monthly listeners</div>
          </div>
        </div>
      </Card>
      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">Popular Songs</h2>
        <TrackTable songs={data.popular_songs} />
      </section>
      <SectionCarousel title="Albums" description="Explore the artist’s recent releases.">
        {data.popular_songs.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>
      <Card className="space-y-4">
        <h2 className="text-2xl font-semibold">Similar Artists</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {data.similar_artists.map((artist) => (
            <div key={artist.id} className="rounded-3xl bg-white/[0.04] p-5">
              <div className="text-lg font-semibold text-white">{artist.name}</div>
              <div className="mt-2 text-sm text-muted">{artist.genres.join(" • ")}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
