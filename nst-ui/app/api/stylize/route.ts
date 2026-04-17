import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { MODEL_OPTIONS } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VALID_MODELS = new Set(MODEL_OPTIONS);
const VALID_INIT_MODES = new Set(["content", "noise", "blend"]);
const VALID_DEVICES = new Set(["auto", "cpu", "cuda", "mps"]);

type ParsedForm = {
  content: File;
  style: File;
  models: string[];
  steps: number;
  styleWeight: number;
  contentWeight: number;
  tvWeight: number;
  initMode: string;
  logInterval: number;
  device: string;
  seed: number;
};

function parseNumber(value: FormDataEntryValue | null, fallback: number): number {
  if (typeof value !== "string") return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseModels(value: FormDataEntryValue | null): string[] {
  if (typeof value !== "string") return ["vgg19"];
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return ["vgg19"];
    const filtered = parsed.filter((m) => typeof m === "string" && VALID_MODELS.has(m as (typeof MODEL_OPTIONS)[number]));
    return filtered.length > 0 ? filtered : ["vgg19"];
  } catch {
    return ["vgg19"];
  }
}

async function writeUploadedFile(file: File, targetPath: string): Promise<void> {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);
  await fs.writeFile(targetPath, buffer);
}

async function resolvePythonBin(workspaceRoot: string): Promise<string> {
  if (process.env.PYTHON_BIN && process.env.PYTHON_BIN.trim()) {
    return process.env.PYTHON_BIN.trim();
  }

  const venvPython = process.platform === "win32"
    ? path.join(workspaceRoot, ".venv", "Scripts", "python.exe")
    : path.join(workspaceRoot, ".venv", "bin", "python");

  try {
    await fs.access(venvPython);
    return venvPython;
  } catch {
    return process.platform === "win32" ? "python" : "python3";
  }
}

function parseCsv(csvText: string): Record<string, string>[] {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  const rows: Record<string, string>[] = [];

  for (const line of lines.slice(1)) {
    const cols = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((header, idx) => {
      row[header] = cols[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

function readFormData(formData: FormData): ParsedForm {
  const content = formData.get("content");
  const style = formData.get("style");

  if (!(content instanceof File) || !(style instanceof File)) {
    throw new Error("Both content and style images are required.");
  }

  const initModeRaw = formData.get("initMode");
  const deviceRaw = formData.get("device");

  const initMode = typeof initModeRaw === "string" && VALID_INIT_MODES.has(initModeRaw) ? initModeRaw : "content";
  const device = typeof deviceRaw === "string" && VALID_DEVICES.has(deviceRaw) ? deviceRaw : "auto";

  return {
    content,
    style,
    models: parseModels(formData.get("models")),
    steps: Math.max(1, Math.floor(parseNumber(formData.get("steps"), 200))),
    styleWeight: Math.max(0, parseNumber(formData.get("styleWeight"), 1_000_000)),
    contentWeight: Math.max(0, parseNumber(formData.get("contentWeight"), 1)),
    tvWeight: Math.max(0, parseNumber(formData.get("tvWeight"), 0.0001)),
    initMode,
    logInterval: Math.max(1, Math.floor(parseNumber(formData.get("logInterval"), 25))),
    device,
    seed: Math.floor(parseNumber(formData.get("seed"), 42)),
  };
}

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const config = readFormData(formData);

    const uiRoot = process.cwd();
    const workspaceRoot = path.resolve(uiRoot, "..");
    const nstProjectRoot = path.join(workspaceRoot, "nst_project_code");
    const scriptPath = path.join(nstProjectRoot, "nst_compare.py");

    await fs.access(scriptPath);

    const jobId = randomUUID();
    const runtimeRoot = path.join(uiRoot, ".runtime");
    const uploadDir = path.join(runtimeRoot, "uploads", jobId);
    const outputDir = path.join(runtimeRoot, "outputs", jobId);
    await fs.mkdir(uploadDir, { recursive: true });
    await fs.mkdir(outputDir, { recursive: true });

    const contentExt = path.extname(config.content.name || "") || ".png";
    const styleExt = path.extname(config.style.name || "") || ".png";
    const contentPath = path.join(uploadDir, `content${contentExt}`);
    const stylePath = path.join(uploadDir, `style${styleExt}`);
    await writeUploadedFile(config.content, contentPath);
    await writeUploadedFile(config.style, stylePath);

    const pythonBin = await resolvePythonBin(workspaceRoot);

    const args = [
      scriptPath,
      "--content",
      contentPath,
      "--style",
      stylePath,
      "--output-dir",
      outputDir,
      "--steps",
      String(config.steps),
      "--style-weight",
      String(config.styleWeight),
      "--content-weight",
      String(config.contentWeight),
      "--tv-weight",
      String(config.tvWeight),
      "--init",
      config.initMode,
      "--log-interval",
      String(config.logInterval),
      "--device",
      config.device,
      "--seed",
      String(config.seed),
      "--models",
      ...config.models,
    ];

    const command = [pythonBin, ...args].join(" ");
    const processResult = await new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve) => {
      const child = spawn(pythonBin, args, {
        cwd: nstProjectRoot,
        env: process.env,
      });

      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("close", (code) => {
        resolve({ code, stdout, stderr });
      });
    });

    if (processResult.code !== 0) {
      return NextResponse.json(
        {
          error: "Python style transfer execution failed.",
          command,
          stdout: processResult.stdout,
          stderr: processResult.stderr,
        },
        { status: 500 }
      );
    }

    const summaryPath = path.join(outputDir, "comparison_summary.csv");
    const summaryText = await fs.readFile(summaryPath, "utf-8");
    const summary = parseCsv(summaryText);

    const artifacts = Object.fromEntries(
      config.models.map((model) => [
        model,
        {
          imageUrl: `/api/artifact?job=${encodeURIComponent(jobId)}&file=${encodeURIComponent(`${model}_stylized.png`)}`,
          historyUrl: `/api/artifact?job=${encodeURIComponent(jobId)}&file=${encodeURIComponent(`${model}_history.csv`)}`,
        },
      ])
    );

    const summaryUrl = `/api/artifact?job=${encodeURIComponent(jobId)}&file=${encodeURIComponent("comparison_summary.csv")}`;

    return NextResponse.json({
      jobId,
      command,
      stdout: processResult.stdout,
      stderr: processResult.stderr,
      summary,
      summaryUrl,
      artifacts,
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Unexpected server error.",
      },
      { status: 500 }
    );
  }
}
