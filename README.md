# Deep Learning Foundations: A First-Principles Research-Oriented Study

**A Personal Note:**

This repository documents an independent, self-taught project completed entirely during my spare time. As my personal initiation into deep learning, some implementations may inevitably be rough around the edges. However, I have poured my utmost effort into demystifying the granular details of every module. Driven by a profound fascination with the mathematical principles underlying neural networks, I actively sought out advanced theoretical literature to ensure my foundation was built on rigorous mathematics rather than just code APIs. This repository is the raw, honest footprint of that learning journey.

## Core Contributions

### 1. Manual Derivations of Neural Computation
I performed hand-calculated forward and backward passes, parameter count analyses, and gradient-update derivations for foundational and state-of-the-art architectures:

- **Convolutional Networks (eg. AlexNet)**: Manual computation of activation manifold dimensions and parameter volumes (e.g., Conv1: $55 \times 55 \times 96$ neurons, $96 \times 3 \times 11 \times 11$ parameters).
- **Transformer Architecture**: Hand-derived absolute positional encodings and attention-weight gradients to clarify the inductive biases of self-attention.
- **Autograd and Gradient Descent**: Step-by-step derivation of backpropagation and automatic differentiation, including manual updates of fully-connected layer parameters (see `homework/week2 理论作业02万照.pdf`).

### 2. From-Scratch Implementations of Core Mechanisms
To validate theoretical understanding through computation, I implemented key modules using PyTorch tensors rather than relying on pre-built abstractions:

- Attention mechanisms and a simplified Transformer (`code/week9/transformer.py`).
- Custom recurrent and LSTM cells (`code/week8 RNN/CustomRNN.py`).
- Domain-Adversarial Neural Networks (`code/week11&12 GAN/DANN.py`).

### 3. Systematic Architectural Benchmark & Structural Debugging (Week 5)
A controlled comparative study on a 100×100 classification task evaluated six CNN architectures (LeNet, AlexNet, VGG6, NiN, GoogLeNet, ResNet). Rather than treating models as black boxes, this module focused on rigorous evaluation, stochasticity control (dual-run baselines), and deep architectural diagnostics (see `homework/week5作业_万照.pdf`):

- **Architectural Diagnostics & "Model Surgery"**: Successfully elevated underperforming networks (NiN, AlexNet) from ~50% (random guessing) to ~87% accuracy through structural modifications rather than blind hyperparameter tuning:
  - **NiN Diagnosis**: Identified that the ReLU activation in the terminal block caused "information truncation," fundamentally corrupting the CrossEntropyLoss gradient computation. Resolved this by manually rewriting the final network block and injecting Batch Normalization to stabilize the deep architecture.
  - **AlexNet Diagnosis**: Diagnosed a severe parameter explosion (yielding ~1.26 million weights in the flattened FC layer) and late-epoch validation loss rebound. Corrected this by structurally downsizing the FC nodes (4096 → 512) and intensifying the Dropout penalty (0.5 → 0.7) to explicitly force feature forgetting.

- **Feature Representation Insights**: Visualized intermediate feature maps across deep, medium, and shallow layers (e.g., GoogLeNet Conv1 vs. Conv21) to analyze abstraction levels. Demonstrated empirically that deep networks with small receptive fields fundamentally outperform unmodified large-image networks on constrained spatial dimensions.

## Repository Structure

- **`/notebook`**: Scanned handwritten notes and manual derivations.
- **`/math`**: A curated collection of reference textbooks and mathematical papers I independently studied to build my theoretical intuition, including seminal texts like *The Modern Mathematics of Deep Learning*, *Geometric Deep Learning*, and *Mathematics for Machine Learning*.
- **`/code`**: From-scratch and semi-from-scratch implementations organized by week and topic.
- **`/homework`**: Weekly theoretical assignments and analytical reports covering 12 modules.
- **`/paper`**: Curated literature (ResNet, Transformer, DETR, YOLO, etc.) aligned with the weekly syllabus.

## Technologies

Python, PyTorch, NumPy, Matplotlib

## Reproducibility

Most implementations are directly executable after installing the dependencies:

```bash
pip install torch numpy matplotlib
python code/week4/week4_LeNet.py
```

Scripts are organized by week and topic. Some rely on local data files (e.g., `mnist.npy`) or small checkpoints; these are excluded from version control via `.gitignore`.

## Research Objective & Future Outlook

The initial objective of this study was to establish a mathematically rigorous and structurally grounded foundation in deep learning. However, beyond this 12-week exploration, my journey into this field continues today.

I maintain an intense curiosity and an insatiable drive to explore the theoretical limits of AI. While self-study has laid the groundwork, I recognize the boundaries of solitary exploration. My ultimate hope is that, under the rigorous guidance and mentorship of a PhD advisor, I can break through existing theoretical bottlenecks, connect these foundational dots, and apply them precisely to solve complex physical and chemical challenges within the AI4S paradigm.
