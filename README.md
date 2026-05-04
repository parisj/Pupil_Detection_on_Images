# Pupil Detection on Images

A classical computer-vision pipeline for robust pupil detection in close-up eye images and videos, built and evaluated as part of a bachelor thesis. The pipeline is designed to remain accurate in the presence of corneal reflections, which is the dominant failure mode for naive thresholding approaches on real-world eye footage.

The proposed algorithm combines **Haar-like feature detection**, **Active Contours Without Edges (ACWE)**, and **RANSAC ellipse fitting**, and is evaluated against the [Labelled Pupils in the Wild (LPW)](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw) dataset. The full thesis, in which every algorithm is derived and discussed in detail, is available as a PDF in the `Latex/` folder.

## Results

On a 2000-frame LPW video the proposed pipeline achieves:

- **Mean error:** 1.4 pixels
- **Success rate:** 98.9%
- **Throughput:** 1.3 fps

Performance degrades gracefully with image noise. The algorithm has been verified to run on both individual images and full video sequences.

## Contents

```
.
├── Code/
│   └── iris detection/
│       └── pupil_detection.py   # Simplified entry point — start here
├── Latex/                        # Thesis sources and compiled PDF
├── LPW.csv                       # LPW image paths and ground-truth annotations
├── requirements.txt
└── README.md
```

The thesis PDF in `Latex/` is the canonical reference for the methodology and contains the complete derivations, parameter studies, and a full comparison of every algorithm in the repository.

## The proposed pipeline

The detector is a four-stage pipeline. Each stage hands a progressively refined estimate to the next.

### 1. Seedpoint detection — Haar-like features

A Haar-like feature detector sweeps the image and responds strongest where the local intensity profile matches a dark, roughly circular blob surrounded by lighter tissue. The pixel with the **strongest response** is taken as the seedpoint and is, by construction, located inside the pupil. This step replaces brittle global thresholding and is robust to varying illumination across the eye.

### 2. Initial contour

The seedpoint is used to plant a small initial contour inside the pupil region. This gives the contour-evolution stage a sensible starting point and makes the subsequent ACWE iterations converge quickly.

### 3. Boundary refinement — ACWE

The pupil boundary itself is found with **Active Contours Without Edges** (Chan–Vese). ACWE evolves the contour by minimising a region-based energy functional rather than chasing image gradients, which is exactly what is needed for the pupil: gradients near corneal reflections and eyelashes are misleading, but the region statistics inside vs. outside the pupil remain well-separated. The output of this stage is a binary mask of the pupil area.

### 4. Ellipse fitting — RANSAC

The contour of the ACWE mask is the input to a **RANSAC** ellipse fit. RANSAC is critical here: corneal reflections cut visible chunks out of the pupil boundary, leaving a contour with strong outliers. A least-squares fit on the raw contour would be pulled toward the reflection. RANSAC samples minimal subsets, scores ellipse hypotheses by inlier count, and converges on an ellipse that ignores the reflection-induced outliers. The resulting ellipse is the final pupil estimate (centre, axes, orientation).

## Comparison algorithms

The repository also contains the alternative pupil-detection algorithms that were implemented and evaluated against the proposed pipeline. These were used to justify the final design and to quantify how each component (feature detector, boundary refinement, robust fitting) contributes to overall accuracy and reflection robustness. The thesis presents the full comparison; the code is included so the experiments can be reproduced.

## Installation

The project targets Python 3.8–3.10. Clone the repository and install the dependencies into a virtual environment:

```bash
git clone https://github.com/parisj/Pupil_Detection_on_Images.git
cd Pupil_Detection_on_Images
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Note that `requirements.txt` contains a few duplicate package entries with different versions, a leftover from merging environments. If `pip` complains, keep the newer version of each package and remove the older line.

## Quick start

The simplest way to run the detector is the standalone script:

```bash
# Single image
python "Code/iris detection/pupil_detection.py" --image path/to/eye.png

# Video
python "Code/iris detection/pupil_detection.py" --video path/to/eye.mp4
```

For batch evaluation against LPW, point the script at `LPW.csv`, which maps every image to its ground-truth annotation. The script reports the detected ellipse parameters and can render an overlay onto each frame for visual inspection.

## Reproducing the thesis results

1. Download the LPW dataset and place it locally.
2. Open `LPW.csv` and confirm the image paths point to your local copy.
3. Run the evaluation script in `Code/iris detection/`. Per-subject metrics and aggregate numbers are written to the working directory and can be compared directly against the figures reported in the thesis.

## License

No license has been declared for this repository. If you would like to reuse the code, please open an issue to discuss licensing with the author.

## Acknowledgements

- [LPW dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw) — Max Planck Institute for Informatics
- Chan & Vese, *Active Contours Without Edges*, IEEE Trans. Image Processing, 2001
- Fischler & Bolles, *Random Sample Consensus*, Comm. ACM, 1981


Pupil Detection on Images GitRepository contains files for Pupil detection and evaluating them on the [LPW data set](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw)
The thesis can be found under Latex as PDF file. There are all the algorithms discussed and explained. 

- Simplified script for easier use is in Code/iris detection/pupil_detection.py
