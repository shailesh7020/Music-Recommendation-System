"use client";

import { create } from "zustand";

import type { Song } from "@/lib/types";

type RepeatMode = "off" | "all" | "one";

type PlayerState = {
  activeTrack: Song | null;
  queue: Song[];
  previous: Song[];
  isPlaying: boolean;
  progress: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  isShuffle: boolean;
  repeatMode: RepeatMode;
  setQueue: (songs: Song[]) => void;
  playTrack: (song: Song, queue?: Song[]) => void;
  togglePlayPause: () => void;
  playNext: () => void;
  playPrevious: () => void;
  setProgress: (value: number) => void;
  setDuration: (value: number) => void;
  setVolume: (value: number) => void;
  toggleMute: () => void;
  toggleShuffle: () => void;
  cycleRepeatMode: () => void;
};

export const usePlayerStore = create<PlayerState>((set) => ({
  activeTrack: null,
  queue: [],
  previous: [],
  isPlaying: false,
  progress: 0,
  duration: 0,
  volume: 0.75,
  isMuted: false,
  isShuffle: false,
  repeatMode: "off",
  setQueue: (songs) => set({ queue: songs }),
  playTrack: (song, queue) =>
    set((state) => ({
      activeTrack: song,
      queue: queue ?? state.queue,
      previous: state.activeTrack ? [...state.previous, state.activeTrack] : state.previous,
      isPlaying: true,
      progress: 0,
    })),
  togglePlayPause: () => set((state) => ({ isPlaying: !state.isPlaying })),
  playNext: () =>
    set((state) => {
      if (!state.activeTrack || state.queue.length === 0) {
        return state;
      }

      const availableQueue = state.isShuffle
        ? [...state.queue].sort(() => Math.random() - 0.5)
        : state.queue;
      const currentIndex = availableQueue.findIndex((song) => song.id === state.activeTrack?.id);
      const nextIndex =
        currentIndex >= 0 && currentIndex < availableQueue.length - 1
          ? currentIndex + 1
          : state.repeatMode === "all"
            ? 0
            : currentIndex;

      const nextTrack = availableQueue[nextIndex];
      if (!nextTrack || nextTrack.id === state.activeTrack.id) {
        return { ...state, isPlaying: state.repeatMode === "one" };
      }

      return {
        activeTrack: nextTrack,
        previous: [...state.previous, state.activeTrack],
        isPlaying: true,
        progress: 0,
      };
    }),
  playPrevious: () =>
    set((state) => {
      const previousTrack = state.previous[state.previous.length - 1];
      if (!previousTrack) return state;
      return {
        activeTrack: previousTrack,
        previous: state.previous.slice(0, -1),
        isPlaying: true,
        progress: 0,
      };
    }),
  setProgress: (value) => set({ progress: value }),
  setDuration: (value) => set({ duration: value }),
  setVolume: (value) => set({ volume: value }),
  toggleMute: () => set((state) => ({ isMuted: !state.isMuted })),
  toggleShuffle: () => set((state) => ({ isShuffle: !state.isShuffle })),
  cycleRepeatMode: () =>
    set((state) => ({
      repeatMode:
        state.repeatMode === "off" ? "all" : state.repeatMode === "all" ? "one" : "off",
    })),
}));
