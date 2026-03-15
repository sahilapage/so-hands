# Dexterous Manipulation with a 3D-Printed Hand

An end-to-end project for **dexterous manipulation** using a custom **3D-printed robotic hand** mounted on the SO-ARM100 platform, developed with a blend of **Imitation Learning (IL)** and **Reinforcement Learning (RL)**.

The goal is simple and practical: teach the hand-arm system to perform stable pick-and-place behavior, then iteratively improve policy quality from initial behavior to stronger, more reliable grasps.

---

##  Project Media


![SO-ARM100 with 3D-printed dexterous hand](media/Screenshot%20from%202026-03-16%2001-44-17.png)

https://github.com/user-attachments/assets/9f188c8b-07df-4c28-be88-49698854e292

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
