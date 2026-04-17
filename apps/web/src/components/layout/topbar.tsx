"use client";

import { Bell, ChevronLeft, ChevronRight, Search } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import type { UserProfile } from "@/lib/types";

export function Topbar({
  user,
  onSearch,
}: {
  user: UserProfile;
  onSearch?: (value: string) => void;
}) {
  return (
    <div className="sticky top-0 z-20 flex items-center justify-between gap-4 rounded-3xl border border-white/5 bg-black/30 px-4 py-3 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <button className="rounded-full bg-black/60 p-2 text-white">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button className="rounded-full bg-black/60 p-2 text-white">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <div className="relative max-w-xl flex-1">
        <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <Input
          placeholder="Search songs, artists, albums, genres"
          className="pl-11"
          onChange={(event) => onSearch?.(event.target.value)}
        />
      </div>
      <div className="flex items-center gap-3">
        <button className="rounded-full bg-white/10 p-2 text-white hover:bg-white/15">
          <Bell className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-3 rounded-full bg-black/60 px-2 py-1">
          <Avatar>
            <AvatarImage src={user.profileImage} alt={user.username} />
            <AvatarFallback>{user.username.slice(0, 2).toUpperCase()}</AvatarFallback>
          </Avatar>
          <div className="pr-2">
            <div className="text-sm font-medium text-white">{user.username}</div>
            <div className="text-xs text-muted">Premium</div>
          </div>
        </div>
      </div>
    </div>
  );
}
