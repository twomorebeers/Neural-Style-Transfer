"use client";

type ImageUploaderProps = {
  label: string;
  name: "content" | "style";
  previewUrl: string | null;
  onFileSelected: (name: "content" | "style", file: File | null) => void;
};

export default function ImageUploader({ label, name, previewUrl, onFileSelected }: ImageUploaderProps) {
  return (
    <div className="panel">
      <label className="label" htmlFor={name}>
        {label}
      </label>
      <input
        id={name}
        className="input"
        type="file"
        name={name}
        accept="image/png,image/jpeg,image/webp"
        onChange={(e) => onFileSelected(name, e.target.files?.[0] ?? null)}
        required
      />
      <div style={{ marginTop: 12 }}>
        {previewUrl ? <img className="preview" src={previewUrl} alt={`${label} preview`} /> : <p className="muted">No image selected.</p>}
      </div>
    </div>
  );
}
