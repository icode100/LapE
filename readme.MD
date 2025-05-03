
# LapE: Laplace Embeddings for Knowledge Graph Reasoning

LapE is a knowledge graph embedding model that represents entities and relations as parameters of **Laplace distributions** instead of point vectors. It extends the GammaE framework to allow uncertainty modeling via distributions and supports complex reasoning tasks like multi-hop inference.

> 🚀 Inspired by GammaE, LapE introduces a probabilistic embedding space to capture **uncertainty** and **semantic nuances** in entity-relation interactions.

---

## 🧠 Key Features & Nuances of LapE

- **Probabilistic Embeddings**: Entities are encoded as **Laplace distributions** (mean `μ` and scale `b`) instead of deterministic vectors.
  
- **Distance Metric**: Uses **Laplace-based KL divergence** or related probabilistic distances for scoring triples, allowing soft reasoning.

- **Extension of GammaE**: LapE is structurally inspired by [GammaE](https://github.com/dyang67/GammaE) but introduces critical changes in how embeddings are initialized, regularized, and scored.

- **Learned Uncertainty**: The scale parameter `b` allows the model to capture and propagate uncertainty during reasoning.

- **Compatible with complex reasoning**: Suitable for tasks like:
  - Link prediction
  - Multi-hop question answering
  - Path queries

---
## Improvements
The model performs significantly better than GammaE on FB15k-237 dataset where inverse relations are missing. The results can be observed from the results section of the code base.

---

## Dataset
The datasets used here is standard KGResasoning datasets published by stanford i.e, FB15k-237, NELL and FB15k. To download the dataset [click here](http://snap.stanford.edu/betae/KG_data.zip)
## 📁 Directory Structure
```
LapE/
├── code/
│   ├── __init__.py           # Package initialization
│   ├── dataloader.py         # Data loading utilities
│   ├── LapE.py              # Core LapE implementation
│   ├── main.py              # Main entry point
│   ├── models.py            # Model definitions
│   ├── run.py              # Training runner
│   └── utils.py            # Helper utilities
├── results/
│   ├── kg-reasoning-gammae-fb15k-237.log
│   ├── kg-reasoning-gammae-nell-995.log
│   ├── kg-reasoning-lape-fb15k-237.log
│   ├── kg-reasoning-lape-nell-995.log
│   └── kg-reasoning-lape.ipynb
├── README.md                # Project documentation
└── requirements.txt
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/LapE.git
cd LapE
pip install -r requirements.txt
````

---

## 🚀 Usage

1. using jupyter notebook:
The notebook contains the required parameters and training and evaluation setup 
2. using bash
    ```bash
    cd code
    python -m run.py
    ```
---

## ⚙️ Model Hyperparameters

| Parameter         | Description                          | Example |
| ----------------- | ------------------------------------ | ------- |
| `hidden_dim`      | Dimensionality of entity vectors (μ) | 800     |
| `gamma`           | Margin for scoring function          | 60      |
| `embedding_range` | Controls init scale of embeddings    | auto    |
| `batch_size`      | Mini-batch size during training      | 512     |

Note: Entity embeddings are stored as shape `(n_entities, 2 * hidden_dim)` to encode both mean and scale.

---

## 📌 Acknowledgments

Some components of this repository, especially those related to training pipeline, data handling, and initial scoring functions, are adapted from the excellent [GammaE](https://github.com/dyang67/GammaE) repository.

> Credits to the original authors of GammaE for foundational insights and codebase.

---

