import Link from "next/link";
import { Music4, Sparkles, Waves } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const features = [
  {
    title: "Recommendations that adapt",
    copy: "Hybrid content, collaborative, and mood-aware recommendations power every shelf.",
    icon: Sparkles,
  },
  {
    title: "A premium listening shell",
    copy: "Persistent playback, immersive dark UI, and motion-rich card layouts inspired by modern streaming apps.",
    icon: Music4,
  },
  {
    title: "Playlists with personality",
    copy: "Create, duplicate, reorder, and share playlists with a branded Pulsewave look and feel.",
    icon: Waves,
  },
];

export default function LandingPage() {
  return (
    <main className="min-h-screen px-6 py-8 lg:px-10">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-7xl flex-col gap-8">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-11 w-11 rounded-full bg-gradient-to-br from-accent to-emerald-300 shadow-glow" />
            <div>
              <div className="text-xl font-semibold">Pulsewave</div>
              <div className="text-sm text-muted">Discover music that understands you.</div>
            </div>
          </div>
          <div className="flex gap-3">
            <Link href="/login">
              <Button variant="secondary">Log in</Button>
            </Link>
            <Link href="/signup">
              <Button>Start Listening</Button>
            </Link>
          </div>
        </header>

        <section className="grid flex-1 gap-8 lg:grid-cols-[1.3fr,0.9fr]">
          <Card className="flex flex-col justify-between overflow-hidden bg-black/35 p-8 lg:p-12">
            <div className="space-y-6">
              <div className="inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-muted">
                Spotify-inspired. Original brand. Recommendation-first.
              </div>
              <div className="max-w-3xl space-y-5">
                <h1 className="text-5xl font-semibold leading-tight tracking-tight lg:text-7xl">
                  Discover music that <span className="text-accent">understands you.</span>
                </h1>
                <p className="max-w-2xl text-lg text-muted">
                  Pulsewave blends modern streaming UI with hybrid recommendations, mood intelligence,
                  playlist creation, and a persistent player designed for real listening sessions.
                </p>
              </div>
              <div className="flex flex-wrap gap-4">
                <Link href="/home">
                  <Button size="lg">Open Demo App</Button>
                </Link>
                <Link href="/signup">
                  <Button variant="secondary" size="lg">
                    Create Account
                  </Button>
                </Link>
              </div>
            </div>

            <div className="mt-12 grid gap-4 md:grid-cols-3">
              {features.map(({ title, copy, icon: Icon }) => (
                <div key={title} className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
                  <div className="mb-4 inline-flex rounded-2xl bg-white/10 p-3">
                    <Icon className="h-5 w-5 text-accent" />
                  </div>
                  <h2 className="text-lg font-semibold">{title}</h2>
                  <p className="mt-2 text-sm text-muted">{copy}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="flex flex-col gap-4 overflow-hidden bg-[#151515] p-6">
            <div className="rounded-[28px] bg-gradient-to-br from-emerald-500/30 via-black to-fuchsia-500/20 p-6">
              <div className="mb-4 text-sm font-medium text-white/70">Pulsewave Home</div>
              <div className="rounded-[24px] border border-white/10 bg-black/40 p-5">
                <div className="mb-4 h-48 rounded-[20px] bg-gradient-to-br from-accent via-emerald-400 to-lime-300 shadow-glow" />
                <div className="space-y-2">
                  <div className="h-4 w-24 rounded-full bg-white/10" />
                  <div className="h-8 w-56 rounded-full bg-white/20" />
                  <div className="h-4 w-full rounded-full bg-white/10" />
                  <div className="h-4 w-3/4 rounded-full bg-white/10" />
                </div>
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
                <div className="text-sm text-muted">Premium UI</div>
                <div className="mt-2 text-2xl font-semibold">Dark glassmorphism</div>
              </div>
              <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
                <div className="text-sm text-muted">Recommendation Engine</div>
                <div className="mt-2 text-2xl font-semibold">Hybrid + Mood-aware</div>
              </div>
            </div>
          </Card>
        </section>
      </div>
    </main>
  );
}
