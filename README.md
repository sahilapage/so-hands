# Dexterous Manipulation with a 3D-Printed Hand

An end-to-end project for **dexterous manipulation** using a custom **3D-printed robotic hand** mounted on the SO-ARM100 platform, developed with a blend of **Imitation Learning (IL)** and **Reinforcement Learning (RL)**.

The goal is simple and practical: teach the hand-arm system to perform stable pick-and-place behavior, then iteratively improve policy quality from initial behavior to stronger, more reliable grasps.

---

## 📸 Project Media

### 1) Project Photo

![SO-ARM100 with 3D-printed dexterous hand](media/Screenshot%20from%202026-03-16%2001-44-17.png)

### 2) Hand Mechanism (in making)
/home/sahil/Desktop/so100 arm/so-hands/media/hand_mechanism.mp4
<video controls width="100%" preload="metadata">
	<source src="media/hand_mechanism.mp4" type="video/mp4" />
	Your browser does not support the video tag.
</video>

### 3) Real Hand Build Video (IMG_0447)
/home/sahil/Desktop/so100 arm/so-hands/media/IMG_0447 (1).mp4
<video controls width="100%" preload="metadata">
	<source src="media/IMG_0447%20(1).mp4" type="video/mp4" />
	Your browser does not support the video tag.
</video>

### 4) Pick and Place — Initial Policy
/home/sahil/Desktop/so100 arm/so-hands/media/pick_and_place.mp4
<video controls width="100%" preload="metadata">
	<source src="media/pick_and_place.mp4" type="video/mp4" />
	Your browser does not support the video tag.
</video>

### 5) Pick and Place — Trained Policy (Improved)
/home/sahil/Desktop/so100 arm/so-hands/media/pick_and Place_2.mp4
<video controls width="100%" preload="metadata">
	<source src="media/pick_and%20Place_2.mp4" type="video/mp4" />
	Your browser does not support the video tag.
</video>

---

## 🧠 Project Overview

- **Task**: Dexterous pick-and-place with a multi-finger robotic hand.
- **Learning strategy**:
	- **Imitation Learning** for strong initial behavior.
	- **Reinforcement Learning** for policy refinement and robustness.
- **Simulation engine**: MuJoCo.
- **Hardware direction**: Transfer-oriented development for a real 3D-printed hand mechanism.

---

## 🧩 Repository Structure

- `main.py`: Loads and visualizes the MuJoCo scene.
- `pick_block.py`: Pick-and-place control pipeline with staged grasping and lifting behavior.
- `so+hands/`: Hand-arm MuJoCo assets and XML scene configuration.
- `media/`: Project photo and experiment videos.

---

## ▶️ Quick Start

### Requirements

- Python 3.10+
- MuJoCo 3.6+

### Install

```bash
pip install mujoco
```

### Run scene viewer

```bash
python main.py
```

### Run pick-and-place script

```bash
python pick_block.py
```

---

## 🚀 Vision

Build a reliable dexterous manipulation stack that starts from demonstrations (IL), improves through trial-and-error (RL), and moves toward robust real-world 3D-printed hand control.
