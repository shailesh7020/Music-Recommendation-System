"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { MusicCard } from "@/components/shared/music-card";
import { SectionCarousel } from "@/components/shared/section-carousel";
import { TrackTable } from "@/components/shared/track-table";
import { Card } from "@/components/ui/card";
import { getSong } from "@/lib/api";
import { formatDuration } from "@/lib/utils";

export default function SongDetailPage() {
  const params = useParams<{ songId: string }>();
  const songId = String(params.songId);
  const { data, isLoading } = useQuery({ queryKey: ["song", songId], queryFn: () => getSong(songId) });

  if (isLoading || !data) return <div className="h-72 animate-pulse rounded-3xl bg-white/5" />;

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      <Card className="bg-gradient-to-r from-fuchsia-500/20 via-transparent to-emerald-500/20 p-8">
        <div className="grid gap-8 lg:grid-cols-[260px,1fr]">
          <div className="aspect-square rounded-[32px] bg-gradient-to-br from-fuchsia-500 via-pink-500 to-rose-300 shadow-glow" />
          <div className="flex flex-col justify-end gap-3">
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Song</div>
            <h1 className="text-5xl font-semibold tracking-tight">{data.song.title}</h1>
            <p className="text-lg text-muted">{data.artist.name} • {data.album.title}</p>
            <div className="flex flex-wrap gap-3 text-sm text-white/80">
              <span>{data.song.genre}</span>
              <span>•</span>
              <span>{data.song.mood}</span>
              <span>•</span>
              <span>{formatDuration(data.song.durationMs)}</span>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
        <Card className="space-y-4">
          <h2 className="text-2xl font-semibold">Lyrics</h2>
          <div className="space-y-2 text-base leading-7 text-white/90">
            {data.lyrics.map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </Card>
        <Card className="space-y-4">
          <h2 className="text-2xl font-semibold">Comments & Reactions</h2>
          {data.comments.map((comment) => (
            <div key={`${comment.user}-${comment.comment}`} className="rounded-2xl bg-white/[0.04] p-4">
              <div className="text-sm font-medium text-white">{comment.user} • {comment.reaction}</div>
              <p className="mt-1 text-sm text-muted">{comment.comment}</p>
            </div>
          ))}
        </Card>
      </div>

      <section className="space-y-4">
        <h2 className="text-2xl font-semibold">Similar Songs</h2>
        <TrackTable songs={data.similar_songs} />
      </section>

      <SectionCarousel title="Recommended Songs" description="A second recommendation rail for extra depth.">
        {data.recommended_songs.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>
    </motion.div>
  );
}
