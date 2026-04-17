"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

import { MusicCard } from "@/components/shared/music-card";
import { SectionCarousel } from "@/components/shared/section-carousel";
import { TrackTable } from "@/components/shared/track-table";
import { Card } from "@/components/ui/card";
import { getRecommendations } from "@/lib/api";

export default function RecommendationsPage() {
  const { data, isLoading } = useQuery({ queryKey: ["recommendations"], queryFn: getRecommendations });

  if (isLoading || !data) {
    return <div className="grid gap-4 md:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-52 animate-pulse rounded-3xl bg-white/5" />)}</div>;
  }

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
      <Card className="bg-[#181818] p-8">
        <div className="space-y-3">
          <div className="text-xs uppercase tracking-[0.35em] text-accent">AI recommendation studio</div>
          <h1 className="text-4xl font-semibold tracking-tight">Because you listened to {data.because_you_listened_to.song.title}</h1>
          <p className="max-w-3xl text-muted">
            This page combines content similarity, collaborative overlap, mood lanes, and local trend signals into one product surface.
          </p>
        </div>
      </Card>

      <section className="space-y-4">
        <div>
          <h2 className="text-2xl font-semibold">Because you listened to…</h2>
          <p className="text-sm text-muted">Seeded from your strongest recent signal.</p>
        </div>
        <TrackTable songs={data.because_you_listened_to.results} />
      </section>

      <SectionCarousel title="Similar Users Like" description="Collaborative filtering based on overlapping listening profiles.">
        {data.similar_users_like.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>

      {Object.entries(data.mood_based).map(([mood, songs]) => (
        <SectionCarousel key={mood} title={`${mood} Lane`} description={data.mood_taglines[mood]}>
          {songs.map((song) => (
            <MusicCard key={song.id} song={song} />
          ))}
        </SectionCarousel>
      ))}
    </motion.div>
  );
}
