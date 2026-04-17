import { cn } from "@/lib/utils";

function gradientFromSeed(seed: string) {
  const palettes = [
    "from-emerald-500 via-green-400 to-lime-300",
    "from-fuchsia-600 via-pink-500 to-rose-300",
    "from-cyan-500 via-sky-500 to-blue-400",
    "from-amber-500 via-orange-400 to-yellow-300",
    "from-violet-500 via-purple-500 to-indigo-400",
  ];
  const index = seed.split("").reduce((sum, character) => sum + character.charCodeAt(0), 0) % palettes.length;
  return palettes[index];
}

export function CoverArt({
  seed,
  label,
  className,
}: {
  seed: string;
  label: string;
  className?: string;
}) {
  const initials = label
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <div
      className={cn(
        "flex aspect-square items-center justify-center rounded-2xl bg-gradient-to-br text-lg font-bold tracking-[0.2em] text-white shadow-glow",
        gradientFromSeed(seed),
        className,
      )}
    >
      {initials}
    </div>
  );
}
