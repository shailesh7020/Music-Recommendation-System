import * as React from "react";

import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "flex h-11 w-full rounded-full border border-white/10 bg-white/10 px-4 py-2 text-sm text-white outline-none placeholder:text-muted focus:border-accent",
          className,
        )}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";
