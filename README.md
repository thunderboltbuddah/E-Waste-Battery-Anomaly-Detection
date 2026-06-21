# Industrial Battery Anomaly Detection: E-Waste Sorting & EOL Manufacturing Inspection

An end-to-end computer vision system designed to identify, classify, and isolate battery structural anomalies—specifically distinguishing between within-spec flat geometries and hazardous swollen casings. 

Physical imagery of failing or structurally compromised lithium-ion batteries is dangerously rare and time-consuming to collect manually. This project addresses data scarcity by leveraging a **synthetic data generation pipeline via domain randomization** inside Blender to train a high-performance **YOLOv8 object detection architecture**, deployed as an accessible diagnostic toolkit via a unified **Gradio interface**.

---

## 🎯 Industrial Application Use Cases

This vision pipeline serves a dual-purpose role across the lifecycle of lithium-ion systems:

* **E-Waste Sorting & Safety:** Automates the triage of mixed electronics recycling streams on conveyor lines to safely flag, trace, and isolate volatile, swollen cells before they undergo volatile mechanical crushing or compaction.
* **Manufacturing End-of-Line (EOL) Quality Control:** Functions as a continuous automated inspection gate on factory assembly belts to intercept structural defects, cell leaks, or pouch containment expansion before units pass final verification.

---

## ⚙️ 1. Synthetic Data Generation Pipeline (Blender)

To overcome real-world data scarcity, a synthetic data generation factory was implemented using **Blender's Python API**, generating a dataset of **7,000 high-fidelity images** with corresponding YOLO-formatted annotations.

### Key Pipeline Components:
* **Renderer:** Handled via Blender's `Eevee` engine, leveraging real-time rasterization capabilities to achieve rapid throughput rendering without VRAM bottlenecks.
* **Battery Models:** Composed of standard baseline geometries (`Battery_Normal`) and compromised assets (`Battery_Swollen`). The swollen asset implements custom shape-keys to programmatically alter inflation thresholds.
* **Domain Randomization Techniques:**
    * *Camera Parameters:* Location, tilt, pan, and lens focal length ($85\text{mm}$ range) were randomized to capture multi-scale perspectives.
    * *Industrial Lighting Matrix:* Simulates varied factory shifts (harsh white, cool lab daylight, dim environments, or orange sodium bulbs) by randomly altering light type, position, energy, and shadow soft-maps.
    * *HDRI Shuffling:* Randomizes rolling reflection mapping arrays and ambient strengths to mimic metallic glare modifications.
    * *Swelling Spectrum:* The shape-key value controlling the bulge was randomized between $0.3 \to 1.0$, generating a realistic spectrum of deformation states.
    * *Conveyor Surface Shuffling:* Dynamically shifts the conveyor belt material's color, roughness, and specular properties on every frame iteration.
    * *Object Placement Engine:* Maps screen quadrants to prevent asset clipping while validating coordinate transformations to dynamically project 2D camera coordinates into valid 3D workspace locations.

---

## 📁 2. Data Preparation & Preprocessing (Google Colab)

The synthetic assets were preprocessed inside Google Colab to conform directly to the standard Ultralytics object detection formatting rules:

1.  **Drive Synchronization:** Mounted Google Drive to ingest the synthetic dataset repository into a local Pandas DataFrame for tracking.
2.  **Coordinate Rectification:** Corrected an inverted vertical Y-axis drift in the original labels to align coordinates with standard top-down image pixel configurations.
3.  **Train-Test Split Partitioning:** Segmented unique rendering entries into an $80\% / 20\%$ stratified validation split using a fixed random seed.
4.  **Bounding Box Conversion:** Transformed normalized coordinates ($\text{xmin}, \text{ymin}, \text{xmax}, \text{ymax}$) into standard YOLO coordinates: center coordinates ($\text{x}, \text{y}$) and dimensional scales ($\text{width}, \text{height}$).
5.  **YOLO Formatting Output:** Generated a companion text file configuration for each frame entry containing unified bounding box entries alongside a structured `data.yaml` layout definition file pointing directly to class labels.

---

## 🧠 3. Model Training (YOLOv8)

The training architecture utilizes **transfer learning** principles, adapting a lightweight, pre-trained Convolutional Neural Network backbone (`yolov8n.pt`) to our specialized task.

### Hyperparameters:
* **Input Resolution:** $640 \times 640$ pixels
* **Batch Size:** 32
* **Epochs:** 50
* **Optimization Base:** COCO pre-trained weights

### Performance Evaluation Metrics:

| Metric | Overall Value | normal_battery | swollen_battery |
| :--- | :--- | :--- | :--- |
| **mAP50** | `0.995` | — | — |
| **mAP50-95** | `0.992` | — | — |
| **Precision** | — | `1.000` | `0.999` |
| **Recall** | — | `0.993` | `0.995` |

The exceptional precision and recall scores validate that domain randomization successfully trained the network to distinguish minute geometric casing changes across highly complex lighting profiles.

---

## 🚀 4. Deployment & Inference Implementation

### A. Core Local Inference Script
The deployment pipeline saves the optimized model weights (`best.pt`) directly to a persistent storage directory. A local validation script pulls the layers into runtime memory, executes a forward prediction pass on raw input directories, processes the output array overlays, and renders the bounding boxes onto the source coordinates using target confidence cutoff filters.

### B. Interactive Gradio Interface
For active deployment verification, a web-accessible application dashboard was constructed using Gradio. This module serves as a rapid prototype panel where operators can drag and drop raw inspection frames directly into an intuitive visual UI. The underlying logic extracts spatial box configurations, processes bounding overlays, and outputs a clean text markdown interface breaking down total object counts, individual anomaly densities, and classification certainty scores.

---

## ⚠️ Model Limitations & Inference Best Practices

To guarantee production reliability across both assembly lines and e-waste environments, note the following performance conditions:

* **Camera Angle Bias:** The training data focused on top-down planar positions ($0 \to 30^{\circ}$ tilt variations). To optimize edge segmentation accuracy, cameras should be mounted directly **top-down** over sorting lines or inspection points. Extreme side perspectives may result in a higher miss rate.
* **Object Density Constraints:** The generation layout enforced a maximum threshold of **four batteries per image**. While the network can generalize to larger numbers, optimal accuracy occurs within this layout. Crowded scrap bins should be singularized onto a belt before passing the inspection node.
* **Material Wrappers and Branding Jackets:** The baseline dataset prioritized raw metallic pouch casings and uniform industrial cylindrical skins. While it excels on raw metallic finishes, heavily branded, multi-colored commercial wrappers (e.g., consumer AA/AAA cells) have not been extensively mapped. Expanding the pipeline with random texture overlay masks during generation will improve out-of-distribution e-waste generalization.

---
## 🔗 Dataset Access

The synthetic training dataset generated via the Blender pipeline (containing 7,000 images, YOLO annotations, and the master tracking CSV) is publicly accessible for replication and research:

* **Google Drive Dataset Repository:** [Download Dataset Here](https://drive.google.com/drive/folders/1DtR1P1HqygM0VogRZ3UvEdzG_8tOtSF-?usp=drive_link)

---

<img width="1910" height="1031" alt="Screenshot 2026-06-11 025644" src="https://github.com/user-attachments/assets/c69810e3-b31e-4b10-92ff-e4631dbbcc1c" />
<img width="1918" height="996" alt="Screenshot 2026-06-11 025629" src="https://github.com/user-attachments/assets/34c1d8a7-edaf-456e-bbf1-393f865e321c" />
<img width="731" height="788" alt="download (2)" src="https://github.com/user-attachments/assets/79fed59d-917d-4883-b780-f421191e2f76" />
<img width="731" height="788" alt="download (5)" src="https://github.com/user-attachments/assets/cc21c7f4-8078-487d-9481-813beff7e188" />
<img width="731" height="788" alt="download (1)" src="https://github.com/user-attachments/assets/6916fe8c-e3a6-438d-a524-56c5d834a67c" />
<img width="731" height="788" alt="download (4)" src="https://github.com/user-attachments/assets/9e00054e-e4bf-4282-9ca9-1c271e549273" />
<img width="981" height="889" alt="Screenshot 2026-06-11 025731" src="https://github.com/user-attachments/assets/3ced77c0-dcea-48b4-8a6e-71e6ef0aa309" />

## 🔗 Sample Output

<img width="1536" height="1024" alt="download (7)" src="https://github.com/user-attachments/assets/e4909c5b-7f62-457c-9d6b-4df3759d2699" />
<img width="1200" height="896" alt="download (8)" src="https://github.com/user-attachments/assets/0beb4aef-060f-4c56-98d3-ba1b87a781ef" />
<img width="1200" height="896" alt="download (6)" src="https://github.com/user-attachments/assets/7b0ecc65-e630-4dd4-8c6a-0b6537b001aa" />

