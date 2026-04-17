"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import Link from "next/link";

import { MusicCard } from "@/components/shared/music-card";
import { SectionCarousel } from "@/components/shared/section-carousel";
import { TrackTable } from "@/components/shared/track-table";
import { Card } from "@/components/ui/card";
import { getHome } from "@/lib/api";
import { formatDuration } from "@/lib/utils";

export default function HomePage() {
  const { data, isLoading } = useQuery({ queryKey: ["home"], queryFn: getHome });

  if (isLoading || !data) {
    return <div className="grid gap-4 md:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-64 animate-pulse rounded-3xl bg-white/5" />)}</div>;
  }

  const heroSong = data.made_for_you[0];

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      <Card className="overflow-hidden bg-gradient-to-r from-emerald-500/30 via-emerald-400/10 to-transparent p-8">
        <div className="grid gap-8 lg:grid-cols-[220px,1fr]">
          <div className="aspect-square rounded-[28px] bg-gradient-to-br from-accent via-emerald-300 to-lime-200 shadow-glow" />
          <div className="flex flex-col justify-end gap-3">
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Made for you</div>
            <h1 className="text-4xl font-semibold tracking-tight lg:text-6xl">{heroSong.title}</h1>
            <p className="max-w-2xl text-muted">
              {heroSong.artist} anchors today’s featured mix. Expect {data.made_for_you.length} tracks tuned for mood,
              similarity, and late-night momentum.
            </p>
            <div className="flex flex-wrap gap-3 text-sm text-white/80">
              <span>{heroSong.genre}</span>
              <span>•</span>
              <span>{heroSong.mood}</span>
              <span>•</span>
              <span>{formatDuration(heroSong.durationMs)}</span>
            </div>
          </div>
        </div>
      </Card>

      <SectionCarousel title="Recently Played" description="Pick up exactly where your last session left off.">
        {data.recently_played.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>

      <SectionCarousel title="Made For You" description="Hybrid recommendations tailored from your history and mood.">
        {data.made_for_you.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>

      <SectionCarousel title="Trending Now" description="The highest-energy picks from across the Pulsewave catalog.">
        {data.trending_now.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>

      <section className="grid gap-4 xl:grid-cols-3">
        <Card className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold text-white">Top Artists</h2>
            <p className="text-sm text-muted">The most-listened names in the current demo catalog.</p>
          </div>
          <div className="space-y-3">
            {data.top_artists.map((artist) => (
              <Link key={artist.id} href={`/artists/${artist.id}`} className="block rounded-2xl bg-white/[0.04] px-4 py-4 hover:bg-white/[0.07]">
                <div className="font-medium text-white">{artist.name}</div>
                <div className="text-sm text-muted">{artist.genres.join(" • ")}</div>
              </Link>
            ))}
          </div>
        </Card>

        <Card className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold text-white">New Releases</h2>
            <p className="text-sm text-muted">Album drops styled like a premium release rail.</p>
          </div>
          <div className="space-y-3">
            {data.new_releases.map((album) => (
              <Link key={album.id} href={`/albums/${album.id}`} className="block rounded-2xl bg-white/[0.04] px-4 py-4 hover:bg-white/[0.07]">
                <div className="font-medium text-white">{album.title}</div>
                <div className="text-sm text-muted">{album.genre} • {album.releaseDate}</div>
              </Link>
            ))}
          </div>
        </Card>

        <Card className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold text-white">Recommended Playlists</h2>
            <p className="text-sm text-muted">Curated sets that feel like editorial playlist covers.</p>
          </div>
          <div className="space-y-3">
            {data.recommended_playlists.map((playlist) => (
              <Link key={playlist.id} href={`/playlists/${playlist.id}`} className="block rounded-2xl bg-white/[0.04] px-4 py-4 hover:bg-white/[0.07]">
                <div className="font-medium text-white">{playlist.name}</div>
                <div className="text-sm text-muted">{playlist.description}</div>
              </Link>
            ))}
          </div>
        </Card>
      </section>

      <SectionCarousel title="Based On Your Mood" description="A dedicated shelf shaped around your current lane.">
        {data.based_on_your_mood.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>

      <section className="space-y-4">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-white">Top picks in this session</h2>
          <p className="text-sm text-muted">A Spotify-style track table with instant playback from the persistent player.</p>
        </div>
        <TrackTable songs={data.made_for_you.slice(0, 8)} />
      </section>
    </motion.div>
  );
}
