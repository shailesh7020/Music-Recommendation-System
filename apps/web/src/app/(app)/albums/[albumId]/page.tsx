"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { TrackTable } from "@/components/shared/track-table";
import { Card } from "@/components/ui/card";
import { getAlbum } from "@/lib/api";
import { formatDuration } from "@/lib/utils";

export default function AlbumPage() {
  const params = useParams<{ albumId: string }>();
  const albumId = String(params.albumId);
  const { data, isLoading } = useQuery({ queryKey: ["album", albumId], queryFn: () => getAlbum(albumId) });

  if (isLoading || !data) return <div className="h-72 animate-pulse rounded-3xl bg-white/5" />;

  return (
    <div className="space-y-8">
      <Card className="bg-gradient-to-r from-violet-500/20 via-transparent to-fuchsia-500/20 p-8">
        <div className="grid gap-8 lg:grid-cols-[260px,1fr]">
          <div className="aspect-square rounded-[32px] bg-gradient-to-br from-violet-500 via-purple-500 to-fuchsia-400 shadow-glow" />
          <div className="flex flex-col justify-end gap-3">
            <div className="text-xs uppercase tracking-[0.35em] text-accent">Album</div>
            <h1 className="text-5xl font-semibold tracking-tight">{data.album.title}</h1>
            <p className="text-lg text-muted">{data.artist.name} • {data.album.genre}</p>
            <div className="text-sm text-white/80">
              {data.songs.length} songs • {formatDuration(data.total_duration_ms)}
            </div>
          </div>
        </div>
      </Card>
      <TrackTable songs={data.songs} />
    </div>
  );
}
