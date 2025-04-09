# Real-Time Object Tracking with Adaptive Correlation Filters (MOSSE)

This repository contains the code, data, and documentation for our project: **Real-Time Object Tracking with Adaptive Correlation Filters (MOSSE)**. Our project implements the MOSSE tracker, originally introduced by Bolme et al. in their paper [*Visual Object Tracking using Adaptive Correlation Filters* (2010)](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf). We aim to deliver a robust, real-time object tracker that deepens our understanding of correlation filters and provides a working system for tracking objects in challenging video sequences.

---

## Overview

Tracking objects in video streams has many applications in surveillance, robotics, and augmented reality. The MOSSE tracker is known for its speed and robustness, making it ideal for real-time performance. In this project, we:
- Implement and evaluate the MOSSE tracker.
- Develop an end-to-end pipeline for video I/O, preprocessing, filter initialization, and online filter updates.
- Integrate real-time visualization and performance evaluation, ensuring our system operates at ≥30 FPS on standard hardware.
- Validate our approach using benchmarks (e.g., OTB-2013) and custom video recordings.

---

## Motivation

Real-time object tracking is critical for applications that require immediate and reliable response. The MOSSE tracker, with its adaptive correlation filter approach, offers a balance between computational efficiency and tracking accuracy. By implementing and optimizing MOSSE, our goal is to:
- Gain a deeper understanding of correlation filter-based tracking.
- Develop a system that can reliably track a selected object, even under occlusions and scale changes.
- Achieve performance benchmarks such as maintaining an Intersection-over-Union (IoU) ≥ 0.5 for at least 80% of frames.

---

## Evaluation

Our tracker will be evaluated on benchmark datasets (e.g., OTB-2013) and custom recordings. The key metrics include:
- **Intersection-over-Union (IoU):** The tracker should maintain an IoU ≥ 0.5 for at least 80% of frames in each video sequence.
- **Center-location Error:** We will measure the error between the predicted and actual object centers.
- **Real-Time Performance:** The system should achieve a processing speed of ≥30 FPS on standard hardware.

---

## Resources

- **Software:** Python, OpenCV, NumPy, Matplotlib.
- **Data:** OTB-2013 benchmark videos and custom recordings (e.g., moving ball sequences).
- **Hardware:** Google Colab, personal laptops.
- **Reference Paper:** Bolme et al., *Visual Object Tracking using Adaptive Correlation Filters* (2010).

---

## Group Contributions

- **Alex Marzban (Preprocessing & Core Filter Implementation + Report - Motivation/Approach):**
  - Set up the development environment and dependencies.
  - Implement video I/O and frame preprocessing (grayscale conversion, window cropping, cosine window).
  - Code the MOSSE filter initialization (training on the first frame) and online filter update (Equations 10–12).
  - Draft the Motivation, Approach, and Implementation Details sections of the report.

- **Ayush Sharma (Visualization & Optimization + Report - Implementation Details/Performance):**
  - Develop bounding-box overlay and live visualization (drawing tracked box, PSR display).
  - Integrate occlusion detection using PSR thresholds and implement scale/illumination adaptation.
  - Profile and optimize the code for real-time performance (FFT acceleration, vectorization, target ≥30 FPS).
  - Collect performance data and generate plots.
  - Draft the Real-Time Performance and Results subsections of the report.

- **Fahad Khan (Evaluation Suite & Reporting + Report - Results/Challenges):**
  - Design and implement the evaluation suite (loading benchmark videos, running tracker, computing IoU and center-error).
  - Aggregate results, produce confusion plots (PSR over time, success plots), and compile a demonstration video.
  - Integrate all modules into a single runnable script and prepare code documentation and README.
  - Draft the Discussion, Conclusions, and Challenges/Innovation sections of the report.

### Collaborative Tasks

- All team members participate in code reviews, integration testing, and final editing of the report to ensure consistency and completeness.
