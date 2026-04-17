"use client";

import { Pause, Play, Repeat, Shuffle, SkipBack, SkipForward, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { Button } from "@/components/ui/button";
import { CoverArt } from "@/components/shared/cover-art";
import { usePlayerStore } from "@/store/player-store";

function formatTime(seconds: number) {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remaining = safeSeconds % 60;
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

export function PlayerBar() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const {
    activeTrack,
    isPlaying,
    progress,
    duration,
    queue,
    volume,
    isMuted,
    isShuffle,
    repeatMode,
    togglePlayPause,
    playNext,
    playPrevious,
    setProgress,
    setDuration,
    setVolume,
    toggleMute,
    toggleShuffle,
    cycleRepeatMode,
  } = usePlayerStore();

  const repeatLabel = useMemo(
    () => (repeatMode === "all" ? "Repeat all" : repeatMode === "one" ? "Repeat one" : "Repeat off"),
    [repeatMode],
  );

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (!activeTrack) {
      audio.pause();
      return;
    }

    audio.src = activeTrack.audioUrl;
    audio.load();
    if (isPlaying) {
      void audio.play().catch(() => undefined);
    }
  }, [activeTrack]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = isMuted ? 0 : volume;
  }, [volume, isMuted]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      void audio.play().catch(() => undefined);
    } else {
      audio.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA"].includes((event.target as HTMLElement | null)?.tagName ?? "")) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlayPause();
      } else if (event.code === "ArrowRight") {
        playNext();
      } else if (event.code === "ArrowLeft") {
        playPrevious();
      } else if (event.code === "ArrowUp") {
        setVolume(Math.min(1, volume + 0.1));
      } else if (event.code === "ArrowDown") {
        setVolume(Math.max(0, volume - 0.1));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [playNext, playPrevious, setVolume, togglePlayPause, volume]);

  return (
    <div className="sticky bottom-0 z-30 mt-8 rounded-[28px] border border-white/5 bg-[#101010]/95 px-4 py-3 shadow-glow backdrop-blur-xl">
      <audio
        ref={audioRef}
        onTimeUpdate={(event) => setProgress(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onEnded={() => playNext()}
      />
      <div className="grid gap-4 lg:grid-cols-[1.2fr,1.6fr,1fr]">
        <div className="flex items-center gap-4">
          {activeTrack ? (
            <>
              <CoverArt seed={activeTrack.id} label={activeTrack.title} className="h-14 w-14 rounded-2xl" />
              <div>
                <div className="font-medium text-white">{activeTrack.title}</div>
                <div className="text-sm text-muted">{activeTrack.artist}</div>
              </div>
            </>
          ) : (
            <div className="text-sm text-muted">Pick a track to start your session.</div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Button variant="ghost" size="icon" onClick={toggleShuffle} className={isShuffle ? "text-accent" : ""}>
              <Shuffle className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={playPrevious}>
              <SkipBack className="h-4 w-4" />
            </Button>
            <Button size="icon" onClick={togglePlayPause}>
              {isPlaying ? <Pause className="h-4 w-4 fill-current" /> : <Play className="h-4 w-4 fill-current" />}
            </Button>
            <Button variant="ghost" size="icon" onClick={playNext}>
              <SkipForward className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={cycleRepeatMode} className={repeatMode !== "off" ? "text-accent" : ""}>
              <Repeat className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted">
            <span>{formatTime(progress)}</span>
            <input
              type="range"
              min={0}
              max={duration || 1}
              step={1}
              value={Math.min(progress, duration || 0)}
              onChange={(event) => {
                const audio = audioRef.current;
                if (!audio) return;
                const nextTime = Number(event.target.value);
                audio.currentTime = nextTime;
                setProgress(nextTime);
              }}
              className="h-1 flex-1 accent-accent"
            />
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3">
          <div className="hidden text-xs text-muted xl:block">{queue.length} tracks in queue • {repeatLabel}</div>
          <Button variant="ghost" size="icon" onClick={toggleMute}>
            {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </Button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={isMuted ? 0 : volume}
            onChange={(event) => setVolume(Number(event.target.value))}
            className="h-1 w-28 accent-accent"
          />
        </div>
      </div>
    </div>
  );
}
