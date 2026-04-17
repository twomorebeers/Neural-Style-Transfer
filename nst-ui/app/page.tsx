import StylizeForm from "@/components/StylizeForm";

export default function HomePage() {
  return (
    <main className="container grid" style={{ gap: 20 }}>
      <section className="panel">
        <h1 style={{ marginTop: 0 }}>Neural Style Transfer Comparator</h1>
        <p className="muted">
          Upload content + style images, choose models, run your Python NST pipeline locally,
          and compare outputs from one UI.
        </p>
      </section>
      <StylizeForm />
    </main>
  );
}
