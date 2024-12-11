![Project](https://img.shields.io/badge/Project-hsledge-blue)
![Author](https://img.shields.io/badge/Author-aquinordg-green)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Version](https://img.shields.io/badge/Version-1.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# HammerSLEDge (HSLEDge): Support, Length, Exclusivity and Difference Weigthed for Group Evaluation

HSLEDge is a Python library for evaluating clustering results using a semantic-based approach. Unlike traditional distance-based metrics, this method leverages the semantic relationship between significant frequent patterns identified among cluster items. This internal validation technique is particularly effective for data organized in **categorical form**.

---

## 🔥 Features

- **Semantic Descriptors**: Analyze feature support in clusters.
- **Particularization of Descriptors**: Refine cluster descriptors using customizable thresholds.
- **SLED Indicators**: Evaluate clusters based on Support (S), Length deviation (L), Exclusivity (E), and Descriptor support Difference (D).
- **Customizable Aggregation**: Choose from harmonic, geometric, or median aggregation for SLED indicators.

---

## 🛠 Installation

Install using *git* and *pip install*:

```bash
pip install git+https://github.com/aquinordg/hsledge.git

```

---

## 🚀 Usage

### Importing the Library
```python
import pandas as pd
import numpy as np
from hsledge import hsledge_score, hsledge_score_clusters, semantic_descriptors
```

### Example Workflow
```python
# Generate a random binary dataset
X = np.random.randint(0, 2, (100, 5))

# Specify the number of clusters
num_clusters = 3

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate the HSLEDge score
average_score = hsledge_score(X, labels, aggregation='median')
print(f"Average HSLEDge Score: {average_score}\n")

# Generate semantic descriptors
report = semantic_descriptors(X, labels, particular_threshold=0.5, report_form=True)

# Print cluster descriptors
for i in range(num_clusters):
    print(f"Cluster {i}:\n{report[i]}\n")
```

---

## 📜 Functions Overview

### `hsledge_score`
**Computes the average HSLEDge score for all clusters.**

#### Parameters:
- **`X`**: Binary feature matrix of shape `(n_samples, n_features)`.
- **`labels`**: Cluster labels for each sample.
- **`W`**: Weighting factors for the SLED indicators (default `[0.3, 0.1, 0.5, 0.1]`).
- **`particular_threshold`**: Threshold for descriptor particularization (`None` for no particularization).
- **`aggregation`**: Aggregation method (`'harmonic'`, `'geometric'`, or `'median'`).

#### Returns:
- **`score`**: Average HSLEDge score.

---

### `hsledge_score_clusters`
**Computes the HSLEDge score for individual clusters.**

#### Parameters:
- Same as `hsledge_score`, with the addition of:
  - **`aggregation=None`**: If `None`, returns scores for each SLED indicator separately.

#### Returns:
- **`scores`**: Aggregated HSLEDge scores for each cluster.
- **`score_matrix`**: Individual SLED indicator scores if `aggregation=None`.

---

### `semantic_descriptors`
**Computes semantic descriptors based on feature support in clusters.**

#### Parameters:
- **`X`**: Binary feature matrix of shape `(n_samples, n_features)`.
- **`labels`**: Cluster labels for each sample.
- **`particular_threshold`**: Threshold for descriptor particularization.
- **`report_form`**: If `True`, returns descriptors as a sorted dictionary for each cluster.

#### Returns:
- **`descriptors`**: Matrix with particularized feature support in clusters.
- **`report`**: Sorted dictionary of significant features in each cluster (if `report_form=True`).

---

### `particularize_descriptors`
**Particularizes descriptors based on support thresholds.**

#### Parameters:
- **`descriptors`**: Feature support matrix of shape `(n_clusters, n_features)`.
- **`particular_threshold`**: Threshold for particularization (default `1.0`).

#### Returns:
- **`descriptors`**: Matrix with particularized support values.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

We welcome contributions to HSLEDge! To contribute:
1. Fork this repository.
2. Create a new branch for your feature.
3. Submit a pull request with your changes.

For questions or information, feel free to reach out at: [aquinordga@gmail.com](mailto:aquinordga@gmail.com).

---

## 👨‍💻 Author

Developed by AQUINO, R. D. G. 
[![Lattes](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-plataforma-lattes-32.png)](http://lattes.cnpq.br/2373005809061037)
[![ORCID](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-orcid-32.png)](https://orcid.org/0000-0002-8486-8354)
[![Google Scholar](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-google-scholar-32.png)](https://scholar.google.com/citations?user=r5WsvKgAAAAJ&hl)

---

## 💬 Feedback

Feel free to open an issue or contact me for feedback or feature requests. Your input is highly appreciated!




