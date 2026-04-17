import type { StylizeResponse } from "@/lib/types";

export async function runStylize(formData: FormData): Promise<StylizeResponse> {
  const res = await fetch("/api/stylize", {
    method: "POST",
    body: formData,
  });

  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload?.error ?? "Style transfer request failed.");
  }

  return payload as StylizeResponse;
}
