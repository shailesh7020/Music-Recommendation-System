"use client";

import { motion } from "framer-motion";
import { Play, Plus } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CoverArt } from "@/components/shared/cover-art";
import type { Song } from "@/lib/types";
import { usePlayerStore } from "@/store/player-store";

export function MusicCard({ song }: { song: Song }) {
  const { playTrack } = usePlayerStore();

  return (
    <motion.div whileHover={{ y: -6, scale: 1.01 }} transition={{ duration: 0.18 }}>
      <Card className="group min-w-[220px] space-y-4 bg-[#181818] hover:bg-[#202020]">
        <div className="relative">
          <CoverArt seed={song.id} label={song.title} />
          <Button
            size="icon"
            className="absolute bottom-3 right-3 opacity-0 shadow-glow transition-all group-hover:translate-y-0 group-hover:opacity-100"
            onClick={() => playTrack(song)}
          >
            <Play className="h-4 w-4 fill-current" />
          </Button>
        </div>
        <div className="space-y-1">
          <Link href={`/songs/${song.id}`} className="line-clamp-1 font-semibold text-white">
            {song.title}
          </Link>
          <p className="line-clamp-2 text-sm text-muted">
            {song.artist} • {song.genre}
          </p>
        </div>
        <div className="flex items-center justify-between text-xs text-muted">
          <span>{song.mood}</span>
          <button className="inline-flex items-center gap-1 rounded-full bg-white/5 px-2 py-1 hover:bg-white/10">
            <Plus className="h-3 w-3" />
            Save
          </button>
        </div>
      </Card>
    </motion.div>
  );
}
