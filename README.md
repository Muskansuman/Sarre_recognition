# Color-Invariant Saree Design Recognition

A metric-learning pipeline for **saree pattern recognition under color changes**.

The goal is simple: given a saree image whose color has changed, retrieve the same or visually corresponding design using its **surface motif and spatial structure**, rather than relying on its original color palette.

The system converts a 224×224 image into a compact **128-dimensional L2-normalized embedding**. Images are compared using cosine similarity, enabling both image retrieval and verification.

> **Important dataset note:** the current Indian Saree Patterns dataset contains four folder-level identities — `Banarasi`, `Bandhani`, `Ikat`, and `Pichwai`. These are weave/style families rather than individual designs. Therefore, the current cross-image experiment measures family-level identification. A finer-grained dataset with one identity per design and multiple real colorways would be needed for true SKU/design-level recognition.

---

## Key Features

- Color-invariant visual embeddings
- MobileNetV3-Small backbone for lightweight inference
- 128-dimensional normalized embeddings
- ArcFace classification loss for identity separation
- NT-Xent contrastive loss for color-invariant representation learning
- Hue, saturation, brightness and grayscale augmentation
- Cosine-similarity image retrieval
- Cross-image identification
- Verification using a validation-only threshold
- ROC-AUC, EER and TAR@1% FAR evaluation
- Inference latency and FLOPs measurement
- Reproducible seeded experiments
- GPU-ready Kaggle notebook

---

## Pipeline

```text
                    Input Saree Image
                           │
                           ▼
                 ┌───────────────────┐
                 │ Data Preprocessing │
                 │ 224 × 224 crop     │
                 │ ImageNet normalize │
                 └─────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ MobileNetV3-Small   │
                │ ImageNet pretrained │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ 128-D Embedding     │
                 │ L2 Normalized       │
                 └─────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Cosine Similarity     Verification
                 │                   │
                 ▼                   ▼
        Gallery Ranking       Same / Different
                 │                   │
                 ▼                   ▼
       Rank-1 / Rank-5 / mAP   ROC-AUC / EER / TAR
```

---

## Method

### 1. Color augmentation

Training deliberately changes the appearance of each image:

- Random resized crop
- Horizontal flip
- Hue jitter
- Saturation jitter
- Brightness jitter
- Random grayscale

Two independent views are generated from each training image. The model is encouraged to produce similar embeddings for these views even when their colors differ substantially.

This makes **spatial structure and motif geometry** more useful than raw RGB appearance.

### 2. Embedding network

The backbone is **MobileNetV3-Small pretrained on ImageNet**.

The original classifier is replaced with:

```text
Global Average Pooling
        ↓
Bias-free Linear Layer
        ↓
Batch Normalization
        ↓
128-D Embedding
        ↓
L2 Normalization
```

The normalized embedding allows cosine similarity to be computed efficiently as a dot product.

### 3. ArcFace

ArcFace is used as an identity-separation objective:

```text
ArcFace scale = 30
ArcFace margin = 0.30
ArcFace loss weight = 0.5
```

The current dataset contains only four folder-level classes, so the ArcFace contribution is deliberately limited. For a dataset where each folder represents a unique design, the ArcFace weight can be increased.

### 4. NT-Xent

NT-Xent is applied to the two augmented views of each image.

```text
Original image
      │
      ├── View A ──┐
      │            ├── Similar embeddings
      └── View B ──┘
```

The objective encourages the network to treat recolored versions of the same image as the same visual instance.

Temperature:

```text
0.1
```

### 5. Combined objective

The training objective combines identity discrimination and color-invariant contrastive learning:

```text
Total Loss = ArcFace Loss × 0.5 + NT-Xent Loss
```

---

## Dataset

The notebook expects the **Indian Saree Patterns** dataset with the following structure:

```text
indian-saree-patterns/
├── train/
│   ├── Banarasi/
│   ├── Bandhani/
│   ├── Ikat/
│   └── Pichwai/
│
├── valid/
│   ├── Banarasi/
│   ├── Bandhani/
│   ├── Ikat/
│   └── Pichwai/
│
└── test/
    ├── Banarasi/
    ├── Bandhani/
    ├── Ikat/
    └── Pichwai/
```

### Dataset used in the recorded run

| Split | Images |
|---|---:|
| Train | 1,293 |
| Validation | 115 |
| Test | 60 |
| Identities | 4 |

Recorded class distribution:

| Identity | Train | Validation | Test |
|---|---:|---:|---:|
| Banarasi | 432 | 43 | 14 |
| Bandhani | 279 | 22 | 15 |
| Ikat | 303 | 26 | 13 |
| Pichwai | 279 | 24 | 18 |

The official train/validation/test folders are kept disjoint.

---

## Evaluation Protocol

The evaluation is designed specifically around color changes.

### Self-retrieval

A recolored version of a test image is used as the query while the original test images form the gallery.

```text
Original image → Recolor → Query
                         ↓
                 Embedding model
                         ↓
                  Cosine ranking
                         ↓
              Original image retrieved?
```

Rank-1 measures whether the original source image is the nearest gallery item.

### Cross-image retrieval

The test set is divided into a seeded 50/50 gallery/query split, stratified by identity.

The query image is not present in the gallery.

A retrieval is considered correct when a gallery image with the same identity is retrieved.

Metrics:

- Rank-1
- Rank-5
- Mean Average Precision (mAP)

### Same-palette control

A second experiment recolors the gallery using the same transformation applied to the query.

This helps determine whether the model is exploiting a residual color cue instead of the underlying pattern.

### Verification

Verification asks:

> Are these two images the same identity?

Positive pairs include:

- Original image ↔ recolored copy
- Different photo ↔ same identity photo

Negative pairs include:

- Different identity, original appearance
- Different identity with the same recoloring transformation

The verification threshold is determined **only on the validation set** using the equal-error criterion and then frozen for test evaluation.

Reported metrics:

- ROC-AUC
- Equal Error Rate (EER)
- TAR at 1% FAR
- Accuracy at the validation-derived threshold

This prevents the test set from being used to tune the decision threshold.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Backbone | MobileNetV3-Small |
| Image size | 224 × 224 |
| Embedding dimension | 128 |
| Epochs | 10 |
| Batch size | 64 |
| ArcFace scale | 30.0 |
| ArcFace margin | 0.30 |
| ArcFace weight | 0.5 |
| NT-Xent temperature | 0.1 |
| Validation image cap | 256 |
| Random seed | 42 |
| DataLoader workers | 2 |
| Normalization | ImageNet |

---

## Training Results

The recorded run trained for 10 epochs and selected the checkpoint using the validation invariance score rather than simply taking the final epoch.

The best checkpoint was:

```text
Epoch: 9
Validation score: 0.9325
```

Selected training trajectory:

| Epoch | Loss | ID Loss | Color Loss | Val Score |
|---:|---:|---:|---:|---:|
| 1 | 4.552 | 6.946 | 1.079 | 0.727 |
| 2 | 2.401 | 3.280 | 0.761 | 0.842 |
| 3 | 1.639 | 2.028 | 0.625 | 0.882 |
| 4 | 1.346 | 1.483 | 0.605 | 0.913 |
| 5 | 1.081 | 1.057 | 0.553 | 0.920 |
| 6 | 0.853 | 0.698 | 0.504 | 0.932 |
| 7 | 0.817 | 0.671 | 0.482 | 0.931 |
| 8 | 0.736 | 0.518 | 0.478 | 0.931 |
| **9** | **0.737** | **0.550** | **0.462** | **0.933** |
| 10 | 0.712 | 0.449 | 0.488 | 0.932 |

The validation self-similarity peaked earlier and remained stable, while the validation score used for checkpoint selection reached its maximum at epoch 9.

> **Note:** The recorded notebook run stopped before the final identification/verification stage because two evaluation cells were executed before their prerequisite cells. The evaluation code is present in the notebook, but final test Rank-1/Rank-5/mAP and verification metrics should be regenerated by restarting the kernel and running the notebook from top to bottom.

---

## Model Size and Efficiency

The recorded run uses approximately **1.0 million inference parameters** in the embedding network.

The ArcFace layer is a training head and is not required during inference.

```text
Inference:
Image → MobileNetV3-Small → 128-D embedding

Training:
Image → Embedding → ArcFace + NT-Xent
```

The notebook also measures:

- Batch-1 latency
- Batch-32 latency
- Per-image batch-32 latency
- FLOPs
- Inference parameters
- Training parameters including ArcFace

These measurements are hardware-dependent and should be regenerated on the target deployment device.

---

## Running on Kaggle

### 1. Create a Kaggle Notebook

Import the notebook or copy the code into a new Kaggle notebook.

### 2. Attach the dataset

Add the **Indian Saree Patterns** dataset as a Kaggle input.

The notebook automatically searches for a directory containing `train` and `valid`/`val` folders.

### 3. Enable GPU

Use:

```text
Notebook → Settings → Accelerator → GPU
```

The recorded run used:

```text
GPU: NVIDIA Tesla T4
PyTorch: 2.10.0+cu128
```

Internet is required once if ImageNet weights are not already cached.

### 4. Run the notebook from the beginning

For reproducibility, use:

```text
Restart Kernel / Runtime
        ↓
Run All
```

Do **not** execute the evaluation cells independently before the training cell has defined `test_records` and `embed_paths()`.

### 5. Generated outputs

The notebook writes the following files to `/kaggle/working`:

```text
identification.png
verification_roc.png
retrieval_examples.png
identification.csv
self_retrieval.csv
verification.csv
efficiency.csv
training_history.csv
```

---

## Project Structure

```text
.
├── sarre-base.ipynb
├── README.md
└── outputs/
    ├── identification.png
    ├── verification_roc.png
    ├── retrieval_examples.png
    ├── identification.csv
    ├── self_retrieval.csv
    ├── verification.csv
    ├── efficiency.csv
    └── training_history.csv
```

---

## Why This Approach?

A conventional image classifier can easily learn color as a shortcut:

```text
Red saree → Class A
Blue saree → Class B
```

That becomes unreliable when the same design is available in another color.

This project instead creates an embedding space where the desired relationship is:

```text
Same motif + different color
            ↓
     nearby embeddings

Different motif / identity
            ↓
     distant embeddings
```

The combination of strong color augmentation and contrastive learning explicitly discourages the network from depending on palette information.

---

## Limitations

### 1. Dataset identity granularity

The current dataset has only four broad identities. `Banarasi`, `Bandhani`, `Ikat`, and `Pichwai` represent families/styles rather than individual designs.

Therefore, the current experiment should **not** be interpreted as fine-grained saree SKU recognition.

### 2. Synthetic recoloring

Color invariance is tested using synthetic hue/saturation/grayscale transformations. Real-world dye variations, lighting changes, camera white balance and fabric aging may behave differently.

### 3. Small evaluation set

The recorded test split contains only 60 images. Larger, identity-balanced test sets are required for stronger statistical conclusions.

### 4. Real deployment validation

Latency and FLOPs are hardware-dependent. Deployment results should be measured again on the intended CPU/GPU/edge device.

---

## Future Work

The most important next step is to move from **family-level classification** to **true design-level retrieval**.

A stronger dataset would look like:

```text
Design_001/
    red_01.jpg
    blue_01.jpg
    green_01.jpg
    showroom_01.jpg

Design_002/
    red_01.jpg
    yellow_01.jpg
    outdoor_01.jpg
    closeup_01.jpg
```

This would allow the model to learn:

```text
Same design
├── different color
├── different lighting
├── different camera
├── different viewpoint
└── different photo
        ↓
   Same embedding cluster
```

Potential extensions:

- Hard-negative mining
- Larger embedding dimensions
- Stronger metric-learning losses
- Triplet loss comparison
- Supervised contrastive learning
- DINO/ViT-based embeddings
- Real multi-color design identities
- Product/SKU-level retrieval
- ANN search using FAISS
- TensorRT/ONNX deployment
- Edge-device benchmarking
- Large-scale gallery indexing

---

## Citation

If you use this project in research or an engineering report, cite the repository and the underlying dataset.

```bibtex
@misc{color_invariant_saree_recognition,
  title  = {Color-Invariant Saree Design Recognition},
  author = {Arashdeep Singh},
  year   = {2026},
  note   = {Metric learning for color-invariant saree pattern retrieval}
}
```

---

## License

Add the license that applies to your code and verify the dataset's license separately before redistributing the dataset or derived assets.

---

## Summary

This project demonstrates a compact metric-learning system that converts saree images into **color-invariant visual embeddings** and uses those embeddings for retrieval and verification.

The core idea is:

```text
                 Saree Image
                      │
          ┌───────────┴───────────┐
          │                       │
       Original                Recolored
          │                       │
          └───────────┬───────────┘
                      ▼
             MobileNetV3-Small
                      │
                      ▼
                  128-D Vector
                      │
                      ▼
             Cosine Similarity
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Identification          Verification
     Rank-1/Rank-5/mAP       AUC/EER/TAR
```

The resulting system is intentionally lightweight while providing a clean foundation for fine-grained, design-level saree retrieval once a suitable multi-color identity dataset is available.
