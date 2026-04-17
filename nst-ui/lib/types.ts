export const MODEL_OPTIONS = ["vgg16", "vgg19", "resnet50", "inception_v3", "squeezenet1_1"] as const;

export type ModelOption = (typeof MODEL_OPTIONS)[number];

export type SummaryRow = {
  model: string;
  input_size: string;
  steps: string;
  content_weight: string;
  style_weight: string;
  tv_weight: string;
  runtime_sec: string;
  content_loss: string;
  style_loss: string;
  tv_loss: string;
  total_loss: string;
  parameters: string;
  output_image: string;
  loss_history_csv: string;
  device: string;
};

export type StylizeResponse = {
  jobId: string;
  command: string;
  stdout: string;
  stderr: string;
  summary: SummaryRow[];
  summaryUrl: string;
  artifacts: Record<string, { imageUrl: string; historyUrl: string }>;
};
