# Neural Style Transfer Project - Pretrained Backbone Comparison

This project implements **optimization-based neural style transfer** with multiple pretrained CNN backbones:

- VGG16
- VGG19
- ResNet50
- Inception V3
- SqueezeNet 1.1 (extra lightweight comparison baseline)

## Files

- `nst_compare.py` - main implementation
- `requirements.txt` - minimal Python dependencies

## Recommended environment

- Python 3.10+
- PyTorch + TorchVision
- GPU optional but strongly recommended

## Install

```bash
pip install -r requirements.txt
```

If you need CUDA, install the PyTorch build that matches your system from the official PyTorch installer page.

## Inputs you need

At minimum you need:

1. **One content image**
   - example: a photograph, landscape, portrait, building, street scene
2. **One style image**
   - example: a painting by Nicolae Grigorescu

For a stronger project demo, prepare:

- 3 content images
- 3 to 5 style paintings
- run several combinations

## Example run

Run only the recommended baseline:

```bash
python nst_compare.py --content content.jpg --style grigorescu.jpg --models vgg19 --steps 300 --output-dir outputs
```

Run all main backbones:

```bash
python nst_compare.py --content content.jpg --style grigorescu.jpg --models vgg16 vgg19 resnet50 inception_v3 squeezenet1_1 --steps 300 --output-dir outputs
```

## Output files

For each model, the script saves:

- stylized image: `MODELNAME_stylized.png`
- loss history: `MODELNAME_history.csv`

It also saves:

- `comparison_summary.csv`

## How to compare the models

Because neural style transfer is partly subjective, use **both qualitative and quantitative comparison**.

### Qualitative criteria

Put the output images side by side and discuss:

- content preservation
- style strength
- color transfer
- brushstroke / texture quality
- visual artifacts
- sharpness

### Quantitative criteria from the script

The script records:

- runtime in seconds
- final content loss
- final style loss
- total variation loss
- total loss
- parameter count

### Important note

Loss values are **most meaningful when comparing runs within the same backbone family**. They are still useful as project evidence, but the best cross-model comparison should include visual inspection.

## Suggested project interpretation

Typical findings you can discuss:

- **VGG19** often gives the most visually pleasing classical NST output.
- **VGG16** is similar but sometimes slightly less rich in texture detail.
- **ResNet50** may preserve structure well but style can look less "painterly".
- **Inception V3** can behave differently because of its architecture and 299x299 input requirement.
- **SqueezeNet** is useful as a small, fast baseline, but output quality is usually lower.

## Suggested presentation structure

1. Problem: stylize a content image with Romanian painter style
2. Data: content images + Nicolae Grigorescu paintings
3. Related work: Gatys NST and later feed-forward methods
4. Why pretrained networks
5. Why VGG19 is the baseline
6. Implementation details
7. Results and comparison
8. Limitations
9. Improvements not implemented

## Good default hyperparameters

- steps: `300`
- style weight: `1e6`
- content weight: `1.0`
- tv weight: `1e-4`
- init: `content`

For stronger stylization, try:

- `--style-weight 5e6`

For smoother images, try:

- `--tv-weight 5e-4`

For more experimental results, try:

- `--init blend`
- `--init noise`

## Fair comparison advice

If you compare models, keep these fixed:

- same content image
- same style image
- same random seed
- same number of steps
- same initialization mode
- same loss weights when possible

Then discuss that the architectures still differ in feature hierarchy, so exact equality is not expected.

## Suggested project conclusion

A strong conclusion is:

> Among the tested pretrained networks, VGG19 remained the best backbone for classical optimization-based neural style transfer, providing the best balance between content preservation and painterly texture quality, while alternative architectures such as ResNet50 and Inception V3 were useful comparison baselines but less consistent for artistic stylization.

