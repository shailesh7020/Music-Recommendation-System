import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function SignupPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-10">
      <Card className="w-full max-w-md space-y-6 bg-black/40 p-8">
        <div className="space-y-2 text-center">
          <div className="text-sm uppercase tracking-[0.35em] text-accent">Join Pulsewave</div>
          <h1 className="text-3xl font-semibold">Create your listening identity</h1>
          <p className="text-sm text-muted">Build playlists, save tracks, and unlock personalized recommendation rails.</p>
        </div>
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <Button variant="secondary" className="w-full">
              Sign up with Google
            </Button>
            <Button variant="secondary" className="w-full">
              Sign up with GitHub
            </Button>
          </div>
          <div className="relative py-1 text-center text-xs uppercase tracking-[0.3em] text-muted">
            <span className="bg-[#181818] px-3">or create with email</span>
          </div>
          <Input placeholder="Username" />
          <Input placeholder="Email" />
          <Input placeholder="Password" type="password" />
          <Input placeholder="Confirm password" type="password" />
          <label className="flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" className="h-4 w-4 rounded border-white/20 bg-white/5 accent-accent" />
            <span>Keep me signed in on this device</span>
          </label>
          <Button className="w-full">Create account</Button>
        </div>
        <div className="space-y-3 text-center text-sm text-muted">
          <p>JWT signup is ready in the API. Social auth and recovery flows are the next backend integrations for these controls.</p>
          <Link href="/login" className="text-white">
            Already have an account? Log in
          </Link>
        </div>
      </Card>
    </main>
  );
}
