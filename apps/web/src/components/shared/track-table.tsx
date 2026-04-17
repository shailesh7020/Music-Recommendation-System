"use client";

import { Clock3, Heart, Play } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { CoverArt } from "@/components/shared/cover-art";
import type { Song } from "@/lib/types";
import { usePlayerStore } from "@/store/player-store";

function formatDuration(durationMs: number) {
  const totalSeconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function TrackTable({ songs }: { songs: Song[] }) {
  const { playTrack, setQueue } = usePlayerStore();

  return (
    <div className="overflow-hidden rounded-3xl border border-white/5 bg-[#181818]">
      <div className="grid grid-cols-[40px,2.5fr,1.2fr,0.8fr] gap-4 border-b border-white/5 px-5 py-3 text-xs uppercase tracking-[0.24em] text-muted">
        <span>#</span>
        <span>Title</span>
        <span>Genre</span>
        <span className="inline-flex justify-end">
          <Clock3 className="h-4 w-4" />
        </span>
      </div>
      <div className="divide-y divide-white/5">
        {songs.map((song, index) => (
          <div
            key={song.id}
            className="grid grid-cols-[40px,2.5fr,1.2fr,0.8fr] gap-4 px-5 py-4 transition-colors hover:bg-white/[0.04]"
          >
            <button
              className="flex h-8 w-8 items-center justify-center rounded-full text-muted hover:bg-white/10 hover:text-white"
              onClick={() => {
                setQueue(songs);
                playTrack(song, songs);
              }}
            >
              {index + 1}
            </button>
            <div className="flex items-center gap-3">
              <CoverArt seed={song.id} label={song.title} className="h-12 w-12 rounded-xl" />
              <div className="min-w-0">
                <Link href={`/songs/${song.id}`} className="block truncate font-medium text-white">
                  {song.title}
                </Link>
                <Link href={`/artists/${song.artistId}`} className="truncate text-sm text-muted hover:text-white">
                  {song.artist}
                </Link>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm text-muted">
              <span>{song.genre}</span>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <Heart className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center justify-end gap-3 text-sm text-muted">
              {song.reason ? <span className="hidden text-right xl:inline">{song.reason}</span> : null}
              <button
                className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-accent text-black"
                onClick={() => {
                  setQueue(songs);
                  playTrack(song, songs);
                }}
              >
                <Play className="h-4 w-4 fill-current" />
              </button>
              <span>{formatDuration(song.durationMs)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
