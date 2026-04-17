"use client";

import { useEffect, useMemo, useState } from "react";
import ImageUploader from "@/components/ImageUploader";
import ModelSelector from "@/components/ModelSelector";
import { runStylize } from "@/lib/nstClient";
import type { ModelOption, StylizeResponse } from "@/lib/types";

type FileMap = {
  content: File | null;
  style: File | null;
};

type PreviewMap = {
  content: string | null;
  style: string | null;
};

export default function StylizeForm() {
  const [files, setFiles] = useState<FileMap>({ content: null, style: null });
  const [previews, setPreviews] = useState<PreviewMap>({ content: null, style: null });
  const [models, setModels] = useState<ModelOption[]>(["vgg19"]);
  const [steps, setSteps] = useState(200);
  const [styleWeight, setStyleWeight] = useState(1_000_000);
  const [contentWeight, setContentWeight] = useState(1);
  const [tvWeight, setTvWeight] = useState(0.0001);
  const [initMode, setInitMode] = useState("content");
  const [logInterval, setLogInterval] = useState(25);
  const [device, setDevice] = useState("auto");
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StylizeResponse | null>(null);

  useEffect(() => {
    return () => {
      Object.values(previews).forEach((url) => {
        if (url) URL.revokeObjectURL(url);
      });
    };
  }, [previews]);

  const canSubmit = useMemo(() => {
    return Boolean(files.content && files.style && models.length > 0 && !running);
  }, [files.content, files.style, models.length, running]);

  const onFileSelected = (name: "content" | "style", file: File | null) => {
    setFiles((prev) => ({ ...prev, [name]: file }));
    setPreviews((prev) => {
      if (prev[name]) URL.revokeObjectURL(prev[name]);
      return {
        ...prev,
        [name]: file ? URL.createObjectURL(file) : null,
      };
    });
  };

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!files.content || !files.style) {
      setError("Please provide both content and style images.");
      return;
    }
    if (models.length === 0) {
      setError("Please choose at least one model.");
      return;
    }

    const payload = new FormData();
    payload.append("content", files.content);
    payload.append("style", files.style);
    payload.append("models", JSON.stringify(models));
    payload.append("steps", String(steps));
    payload.append("styleWeight", String(styleWeight));
    payload.append("contentWeight", String(contentWeight));
    payload.append("tvWeight", String(tvWeight));
    payload.append("initMode", initMode);
    payload.append("logInterval", String(logInterval));
    payload.append("device", device);
    payload.append("seed", String(seed));

    try {
      setRunning(true);
      const response = await runStylize(payload);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run style transfer.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid" style={{ gap: 16 }}>
      <form className="grid" onSubmit={onSubmit} style={{ gap: 16 }}>
        <div className="grid grid-2">
          <ImageUploader label="Content image" name="content" previewUrl={previews.content} onFileSelected={onFileSelected} />
          <ImageUploader label="Style image" name="style" previewUrl={previews.style} onFileSelected={onFileSelected} />
        </div>

        <ModelSelector selectedModels={models} onChange={setModels} />

        <div className="panel grid grid-2">
          <label>
            <span className="label">Steps</span>
            <input className="input" type="number" min={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
          <label>
            <span className="label">Style weight</span>
            <input className="input" type="number" min={0} value={styleWeight} onChange={(e) => setStyleWeight(Number(e.target.value))} />
          </label>
          <label>
            <span className="label">Content weight</span>
            <input className="input" type="number" min={0} step="0.1" value={contentWeight} onChange={(e) => setContentWeight(Number(e.target.value))} />
          </label>
          <label>
            <span className="label">TV weight</span>
            <input className="input" type="number" min={0} step="0.00001" value={tvWeight} onChange={(e) => setTvWeight(Number(e.target.value))} />
          </label>
          <label>
            <span className="label">Init mode</span>
            <select className="select" value={initMode} onChange={(e) => setInitMode(e.target.value)}>
              <option value="content">content</option>
              <option value="noise">noise</option>
              <option value="blend">blend</option>
            </select>
          </label>
          <label>
            <span className="label">Device</span>
            <select className="select" value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
              <option value="mps">mps</option>
            </select>
          </label>
          <label>
            <span className="label">Log interval</span>
            <input className="input" type="number" min={1} value={logInterval} onChange={(e) => setLogInterval(Number(e.target.value))} />
          </label>
          <label>
            <span className="label">Seed</span>
            <input className="input" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
        </div>

        <button className="button" type="submit" disabled={!canSubmit}>
          {running ? "Running style transfer..." : "Run comparison"}
        </button>
      </form>

      {error ? (
        <section className="panel">
          <h3 style={{ marginTop: 0, color: "#fda4af" }}>Error</h3>
          <p>{error}</p>
        </section>
      ) : null}

      {result ? (
        <section className="panel grid" style={{ gap: 14 }}>
          <h2 style={{ marginTop: 0, marginBottom: 0 }}>Results</h2>
          <p className="muted" style={{ margin: 0 }}>
            Job: {result.jobId} • <a href={result.summaryUrl}>Download comparison summary CSV</a>
          </p>

          <div className="grid grid-2">
            {Object.entries(result.artifacts).map(([model, artifact]) => {
              const row = result.summary.find((r) => r.model === model);
              return (
                <article key={model} className="result-card">
                  <h3 style={{ marginTop: 0 }}>{model}</h3>
                  <img className="preview" src={artifact.imageUrl} alt={`${model} stylized result`} />
                  <p className="muted">
                    Runtime: {row?.runtime_sec ?? "n/a"}s • Total loss: {row?.total_loss ?? "n/a"}
                  </p>
                  <a href={artifact.historyUrl}>Download history CSV</a>
                </article>
              );
            })}
          </div>

          <details>
            <summary>Backend logs</summary>
            <div className="mono">
              <strong>Command</strong>
              {"\n"}
              {result.command}
              {"\n\n"}
              <strong>STDOUT</strong>
              {"\n"}
              {result.stdout || "(empty)"}
              {"\n\n"}
              <strong>STDERR</strong>
              {"\n"}
              {result.stderr || "(empty)"}
            </div>
          </details>
        </section>
      ) : null}
    </div>
  );
}
