import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-10">
      <Card className="w-full max-w-md space-y-6 bg-black/40 p-8">
        <div className="space-y-2 text-center">
          <div className="text-sm uppercase tracking-[0.35em] text-accent">Welcome back</div>
          <h1 className="text-3xl font-semibold">Log in to Pulsewave</h1>
          <p className="text-sm text-muted">Demo auth can use any of the sample accounts from the API seed.</p>
        </div>
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <Button variant="secondary" className="w-full">
              Continue with Google
            </Button>
            <Button variant="secondary" className="w-full">
              Continue with GitHub
            </Button>
          </div>
          <div className="relative py-1 text-center text-xs uppercase tracking-[0.3em] text-muted">
            <span className="bg-[#181818] px-3">or use email</span>
          </div>
          <Input placeholder="Email" />
          <Input placeholder="Password" type="password" />
          <div className="flex items-center justify-between text-sm text-muted">
            <label className="flex items-center gap-2">
              <input type="checkbox" className="h-4 w-4 rounded border-white/20 bg-white/5 accent-accent" />
              <span>Remember me</span>
            </label>
            <button type="button" className="text-white transition hover:text-accent">
              Forgot password?
            </button>
          </div>
          <Button className="w-full">Log in</Button>
        </div>
        <div className="space-y-3 text-center text-sm text-muted">
          <p>JWT login is ready in the API. OAuth and password recovery can plug into the reserved actions above next.</p>
          <Link href="/signup" className="text-white">
            Need an account? Create one
          </Link>
        </div>
      </Card>
    </main>
  );
}
