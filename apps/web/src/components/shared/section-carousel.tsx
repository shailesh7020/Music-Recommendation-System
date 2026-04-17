import type { ReactNode } from "react";

export function SectionCarousel({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-white">{title}</h2>
          <p className="text-sm text-muted">{description}</p>
        </div>
        {action}
      </div>
      <div className="flex gap-4 overflow-x-auto pb-2">{children}</div>
    </section>
  );
}
