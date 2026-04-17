import { MobileNav } from "@/components/layout/mobile-nav";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { PlayerBar } from "@/components/player/player-bar";
import { demoUser, playlists } from "@/lib/mock-data";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen px-4 py-4 lg:px-5">
      <div className="grid min-h-[calc(100vh-2rem)] gap-4 lg:grid-cols-[280px,1fr]">
        <aside className="hidden lg:block">
          <Sidebar playlists={playlists} />
        </aside>
        <section className="flex min-h-full flex-col gap-4">
          <Topbar user={demoUser} />
          <div className="flex-1">{children}</div>
          <MobileNav />
          <PlayerBar />
        </section>
      </div>
    </main>
  );
}
