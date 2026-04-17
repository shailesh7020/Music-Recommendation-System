"use client";

import { Heart, Home, Library, ListMusic, PlusCircle, Search } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Playlist } from "@/lib/types";

const primaryNav = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/search", label: "Search", icon: Search },
  { href: "/library", label: "Library", icon: Library },
  { href: "/liked-songs", label: "Liked Songs", icon: Heart },
  { href: "/recommendations", label: "Recommendations", icon: ListMusic },
];

export function Sidebar({ playlists }: { playlists: Playlist[] }) {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col gap-4">
      <Card className="bg-sidebar p-5">
        <div className="mb-6 flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-gradient-to-br from-accent to-emerald-300 shadow-glow" />
          <div>
            <div className="text-lg font-semibold text-white">Pulsewave</div>
            <div className="text-xs text-muted">Spotify-inspired discovery</div>
          </div>
        </div>
        <nav className="space-y-2">
          {primaryNav.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium transition-colors",
                pathname === href ? "bg-white/10 text-white" : "text-muted hover:bg-white/5 hover:text-white",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          ))}
        </nav>
      </Card>

      <Card className="flex-1 bg-sidebar p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-white">Your Playlists</div>
            <div className="text-xs text-muted">Curated for this demo account</div>
          </div>
          <button className="rounded-full bg-white/10 p-2 text-white hover:bg-white/15">
            <PlusCircle className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-2">
          {playlists.map((playlist) => (
            <Link
              key={playlist.id}
              href={`/playlists/${playlist.id}`}
              className="block rounded-2xl px-3 py-3 text-sm text-muted transition-colors hover:bg-white/5 hover:text-white"
            >
              <div className="font-medium text-white">{playlist.name}</div>
              <div className="line-clamp-1 text-xs text-muted">{playlist.description}</div>
            </Link>
          ))}
        </div>
      </Card>
    </div>
  );
}
