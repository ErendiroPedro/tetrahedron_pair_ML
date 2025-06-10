# Learning Tetrahedron-Tetrahedron Intersection and Volume with Neural Networks

## Table of Contents
1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Installation Guide](#installation-guide)
4. [How To Use](#how-to-use)

# Introduction

> "I think the universe is pure geometry — basically, a beautiful shape twisting around and dancing over space-time."  
> — *Antony Garrett Lisi*

## The Problem

Picture a sword slicing through water in a video game. Or a drone navigating debris. Or a medical simulation tracking tissue deformation.

These moments all rely on a silent operation: **did these two 3D shapes intersect, and by how much?**

To answer that, we often break complex shapes down into **tetrahedrons**. They’re the atomic unit of 3D geometry—simple, rigid, and expressive. Whether you're simulating physics, running FEM analyses, or building collision systems, it often comes down to tetrahedron-tetrahedron interaction.

Two fundamental questions arise:

- **Do the tetrahedrons intersect?**
- **If so, what's the volume of that intersection?**

It sounds trivial, but the geometry isn't forgiving. Tetrahedron-tetrahedron intersection is an edge case minefield—six faces, four points, endless configurations. The computation is clean in theory, but slow and branching in practice.

Now scale that to millions of pairs per frame.

## Enter Machine Learning

Rather than solving the same geometric predicates again and again, what if we **learned the outcome directly from data**?

Instead of step-by-step logic trees, imagine a neural network trained to recognize intersection patterns. One that doesn’t follow rigid instructions but learns the behavior of tetrahedrons through thousands of examples. And instead of one CPU doing one pair at a time, we batch it—**processing thousands of pairs in parallel on a GPU**.

That's the core idea:  
**Replace brittle logic with learned geometric intuition**, and exploit the parallel nature of deep learning for real-time scalability.

This approach shifts the bottleneck. It trades exactness for speed, while still delivering reliable estimates—especially useful in applications that demand high throughput: physics engines, medical scans, robotics planning.

This project is a prototype of that vision. A fusion of **computational geometry and neural networks**, distilled into a single question:

**Can a model learn to reason about shape?**

# Pipeline Overview

To teach a neural network geometry, we need more than data—we need structure. This pipeline is that structure.

It takes raw tetrahedron pair data and turns it into trained models that can answer two fundamental questions: *do they intersect, and if so, by how much?*

Every stage is modular, traceable, and built with geometric learning in mind.

![Training Flow](ch3/assets/training_flow.png)  
*Figure: End-to-end training pipeline. Each stage is designed for experimentation—data representation, model configuration, training, and evaluation.*

## Design Principles

- **Geometric Awareness**  
  Operations are aware of 3D structure. Sampling, augmentation, and validation all preserve spatial relationships.

- **Modularity**  
  Each part of the system—data prep, model building, training—is isolated, configurable, and swappable.

- **Reproducibility**  
  Every run is version-controlled. Seeds, configs, outputs—all logged, so experiments are replicable down to the byte.

- **Evaluation Built-In**  
  Confusion matrices, volume error histograms, training curves—generated automatically. Performance is always visible.

## Components

### `CPipelineOrchestrator`
The controller. It reads a YAML config, sets paths, decides what to run, and executes it. From data preprocessing to training and evaluation, this class runs the show.

### `CDataProcessor`
Loads raw intersection data and prepares it for learning. Stratified sampling, tetrahedron permutations, spatial normalization, and curriculum sorting—all handled here.

### `GeometryUtils`
Handles low-level geometric operations. Includes transformations, Morton-based sorting, permutation logic, and alignment to canonical frames.

### `CArchitectureManager`
Builds the model based on config. Supports architectures like DeepSet, MLP, TPNet, and custom modules. Handles input dim inference, residual blocks, and task heads.

### `CArtifactManager`
Tracks outputs. Models are saved in TorchScript format for deployment. Metrics, plots, configs—all logged in timestamped folders, ready for inspection or export.

### `CModelTrainer`
Manages training. Supports classification, regression, or multi-task. Includes GPU acceleration, dynamic schedulers, geometric-aware losses, and auto-logging.

### `CEvaluator`
Runs detailed evaluation. Reports accuracy, F1, volume MAE, and more—stratified by intersection type and with full invariance checks. Also includes device-level profiling.

---

Every experiment, transformation, and model choice is traceable and interchangeable.

