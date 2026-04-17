"use client";

import { Heart, Home, Library, Search, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const items = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/search", label: "Search", icon: Search },
  { href: "/library", label: "Library", icon: Library },
  { href: "/liked-songs", label: "Liked", icon: Heart },
  { href: "/recommendations", label: "For You", icon: Sparkles },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-28 left-4 right-4 z-20 flex items-center justify-between rounded-full border border-white/10 bg-black/70 px-4 py-3 backdrop-blur-xl lg:hidden">
      {items.map(({ href, label, icon: Icon }) => {
        const active = pathname === href;
        return (
          <Link key={href} href={href} className={cn("flex flex-col items-center gap-1 text-[11px]", active ? "text-accent" : "text-muted")}>
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
