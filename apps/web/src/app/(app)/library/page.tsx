"use client";

import { useQuery } from "@tanstack/react-query";

import { MusicCard } from "@/components/shared/music-card";
import { SectionCarousel } from "@/components/shared/section-carousel";
import { getHome } from "@/lib/api";

export default function LibraryPage() {
  const { data } = useQuery({ queryKey: ["library-home"], queryFn: getHome });

  if (!data) return <div className="h-64 animate-pulse rounded-3xl bg-white/5" />;

  return (
    <div className="space-y-8">
      <SectionCarousel title="Recently Played" description="Your latest sessions in one place.">
        {data.recently_played.map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>
      <SectionCarousel title="Saved For Later" description="A clean home for the songs you keep coming back to.">
        {data.made_for_you.slice(0, 6).map((song) => (
          <MusicCard key={song.id} song={song} />
        ))}
      </SectionCarousel>
    </div>
  );
}
