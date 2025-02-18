# Tetrahedron Pair Intersection and Volume Computation Using Neural Networks

> "I think the universe is pure geometry — basically, a beautiful shape twisting around and dancing over space-time."  
> — *Antony Garrett Lisi*

## The Problem

Imagine two objects colliding in a video game—a sword slicing through water, a car crashing into a barrier, or a robot arm navigating a cluttered factory. Behind these moments lies a silent, invisible calculation: **did these shapes intersect, and if so, by how much?**

At the heart of this problem is the humble **tetrahedron**—the 3D building block of complex shapes. Tetrahedrons are everywhere: in **simulations, medical imaging, robotics, and game engines**. But when two tetrahedrons collide, answering two questions becomes critical:

- **Do they intersect?**
- **What’s the volume of their overlap?**

This sounds simple. **It’s not.**

## Why This Matters

- **Collision detection** in games and physics engines relies on this to simulate realism.
- **Medical simulations** use it to model tissue deformation during surgery.
- **Robotics** needs it for motion planning: a robot arm must not intersect itself or its environment.

But here’s the catch: tetrahedron intersection checks are **computationally expensive**. Traditional algorithms scale poorly, relying on brute-force math or approximations. In high-stakes applications—like **autonomous vehicles** or **surgical simulations**—speed and precision are **non-negotiable**.

## Enter Machine Learning

What if we could **teach a neural network** to predict intersections and volumes directly from data? Instead of solving equations step-by-step, imagine a model that **learns the geometric “rules” of tetrahedrons**, accelerating computations by orders of magnitude.

This isn’t just about speed—it’s about unlocking new possibilities in **real-time simulations and beyond**.

This project is an exploration of that idea. It’s a fusion of **computational geometry and deep learning**, with a dash of curiosity:

**Can machines learn the language of shapes?**

## Experiment Pipeline Architecture Overview

![Pipeline Architecture](resources/architecture.png)  
*Figure 1: Machine learning pipeline architecture. The pipeline orchestrator manages the workflow, handling configuration loading, data processing (sampling, transformations, augmentations), model building, and training. An artifacts manager handles saving and loading of configurations, trained models, and evaluation reports for reproducibility.*

### Key Components

- **Data Processing**:  

  Prepares raw tetrahedron pair data for training and validation with configurable steps:

    Data Loading & Sampling

        Loads raw geometric data (vertex coordinates, intersection volume labels).

        Samples datasets based on intersection type distributions (e.g., disjoint, fully intersecting).

        For polyhedron intersecting pairs, applies volume-aware uniform sampling to ensure balanced representation across volume ranges.

    Augmentations (Optional)

        Order invariance: Sorts tetrahedrons by coordinates or size to reduce input permutation sensitivity.

        Geometric transformations: Placeholder support for rigid/affine transformations, vertex permutations, and tetrahedron swaps (configurable but not yet implemented).

    Structured Saving

        Outputs processed data to standardized train/val folders for reproducibility.

- **Model Building**:  
  
- **Training & Evaluation**:  
  
- **Artifact Manager**

---

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/tetrahedron-pair-ml.git
   cd tetrahedron-pair-ml
   ```

2. **Install Git LFS (required for dataset access)**:
   ```bash
   sudo apt install git-lfs  # For Ubuntu/Debian
   git lfs install
   ```
   > *Note: If you're using a different Linux distribution, install Git LFS using the appropriate package manager.*

3. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```