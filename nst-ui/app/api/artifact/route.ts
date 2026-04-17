import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MIME_BY_EXT: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".csv": "text/csv; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

function safeParam(value: string | null): string | null {
  if (!value) return null;
  if (value.includes("..") || value.includes("/") || value.includes("\\")) return null;
  return value;
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const job = safeParam(searchParams.get("job"));
  const file = safeParam(searchParams.get("file"));

  if (!job || !file) {
    return NextResponse.json({ error: "Invalid artifact path." }, { status: 400 });
  }

  const uiRoot = process.cwd();
  const filePath = path.join(uiRoot, ".runtime", "outputs", job, file);

  try {
    await fs.access(filePath);
    const data = await fs.readFile(filePath);
    const ext = path.extname(file).toLowerCase();
    const contentType = MIME_BY_EXT[ext] ?? "application/octet-stream";

    return new NextResponse(data, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
        "Content-Disposition": `inline; filename=\"${file}\"`,
      },
    });
  } catch {
    return NextResponse.json({ error: "Artifact not found." }, { status: 404 });
  }
}
