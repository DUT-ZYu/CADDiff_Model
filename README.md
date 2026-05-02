# CPD-MMRS
Official PyTorch implementation of "CADDiff: Calibration-Enhanced Attention Distillation for Portrait Stylization"
<div align="center">

<h1> CADDiff: Calibration-Enhanced Attention Distillation for Portrait Stylization </h1>

<br>

<!-- 彩色标签 (Badges) -->
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.1+-red.svg?logo=PyTorch" alt="PyTorch"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python" alt="Python"></a>

</div>

<br>

## 📣 News
- **[Latest]** The official PyTorch source code and pre-trained models will be released immediately upon acceptance. Please stay tuned!

---

## 📖 Abstract
The development of diffusion models has significantly advanced image stylization research. However, due to the lack of targeted training in stylistic information or appropriate supervision, existing stylization methods often exhibit facial pattern offset and distortion. To this end, we propose CADDiff, a novel diffusion model with calibration-enhanced attention distillation for portrait stylization. Specifically, we design a calibration-enhanced attention distillation (CAD) mechanism that uses ground-truth facial masks to direct the model’s focus to critical facial regions during training, and distills this region-aware behavior via a calibration distillation loss. This removes the need for mask inputs at inference time, enabling precise facial style migration while preserving global content integrity. Additionally, we construct a spectral orthogonal fusion (SOF) module by entangling orthogonalized content and style features in the spectral domain to reduce the content-style feature domain mismatch phenomenon, thereby improving the stylized quality of generated images. Extensive experiments verify that our CADDiff achieves superior results and surpasses previous methods.

## 🚀 Getting Started (Code Coming Soon)

### Prerequisites
- Linux or windows
- Python 3.11+
- PyTorch 2.1+
- CUDA 12.0+
