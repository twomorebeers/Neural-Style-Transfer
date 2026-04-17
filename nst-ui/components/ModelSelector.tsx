"use client";

import { MODEL_OPTIONS, type ModelOption } from "@/lib/types";

type ModelSelectorProps = {
  selectedModels: ModelOption[];
  onChange: (next: ModelOption[]) => void;
};

export default function ModelSelector({ selectedModels, onChange }: ModelSelectorProps) {
  const toggleModel = (model: ModelOption) => {
    if (selectedModels.includes(model)) {
      onChange(selectedModels.filter((m) => m !== model));
      return;
    }
    onChange([...selectedModels, model]);
  };

  return (
    <div className="panel">
      <p className="label" style={{ marginTop: 0 }}>
        Models
      </p>
      <div className="checkbox-group" style={{ display: "grid", gap: 8 }}>
        {MODEL_OPTIONS.map((model) => (
          <label key={model} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={selectedModels.includes(model)} onChange={() => toggleModel(model)} />
            <span>{model}</span>
          </label>
        ))}
      </div>
      {selectedModels.length === 0 ? <p className="muted">Pick at least one model.</p> : null}
    </div>
  );
}
