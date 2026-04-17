"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { TrackTable } from "@/components/shared/track-table";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { getPlaylist } from "@/lib/api";
import { formatDuration } from "@/lib/utils";

export default function PlaylistPage() {
  const params = useParams<{ playlistId: string }>();
  const playlistId = String(params.playlistId);
  const { data, isLoading } = useQuery({ queryKey: ["playlist", playlistId], queryFn: () => getPlaylist(playlistId) });

  if (isLoading || !data) return <div className="h-72 animate-pulse rounded-3xl bg-white/5" />;

  return (
    <div className="space-y-8">
      <Card className="bg-gradient-to-r from-orange-500/20 via-transparent to-yellow-500/20 p-8">
        <div className="grid gap-8 lg:grid-cols-[260px,1fr]">
          <div className="aspect-square rounded-[32px] bg-gradient-to-br from-orange-500 via-amber-400 to-yellow-300 shadow-glow" />
          <div className="flex flex-col justify-end gap-3">
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Playlist</div>
            <h1 className="text-5xl font-semibold tracking-tight">{data.playlist.name}</h1>
            <p className="max-w-2xl text-muted">{data.playlist.description}</p>
            <div className="text-sm text-white/80">
              {data.songs.length} songs • {formatDuration(data.total_duration_ms)} • {data.playlist.visibility}
            </div>
            <div className="flex flex-wrap gap-3 pt-2">
              <Button>Play Playlist</Button>
              <Button variant="secondary">Share Playlist</Button>
              <Button variant="secondary">Duplicate</Button>
            </div>
          </div>
        </div>
      </Card>
      <TrackTable songs={data.songs} />
    </div>
  );
}
