# 🌍 Ethiopian Household Geo-Analysis: Accessibility & Agricultural Productivity

A collection of four machine learning and geospatial analysis notebooks built on the **Ethiopian Household Geo-Variables (Wave 5)** dataset. This project supports MSc thesis research on rural infrastructure, agricultural productivity, and household accessibility in Ethiopia.

---

## 📁 Project Structure

```
├── Accessiblity.ipynb                    # Household accessibility classification (ML)
├── agricultural_productivity.ipynb       # Agricultural productivity correlation & VIF analysis
├── productivity_classifcation_analysis.ipynb  # Zone-aware 3-class productivity classifier
└── road_accessiblity_Analysis.ipynb      # Road accessibility classification & analysis
```

---

## 📓 Notebook Descriptions

### 1. `Accessiblity.ipynb` — Household Accessibility Classification
Classifies households into **High / Medium / Low** accessibility categories using geospatial, environmental, and distance-based features.

**Key steps:**
- Data loading from Google Drive (LSMS Ethiopia Wave 5)
- Exploratory data analysis: missing values, outliers (IQR method), and descriptive statistics
- Feature engineering: one-hot encoding of `ssa_aez09` and `landcov`
- **Accessibility Index (AI)** created via PCA on 5 distance variables (`dist_road`, `dist_market`, `dist_popcenter`, `dist_border`, `dist_admhq`)
- **Accessibility Risk Score** combining normalized distance, terrain, urban, and population density features
- SMOTE for class imbalance handling

---

### 2. `agricultural_productivity.ipynb` — Productivity Correlation & VIF Analysis
Performs comprehensive **correlation and multicollinearity analysis** on agricultural productivity variables.

**Key steps:**
- Data preprocessing: drop columns >40% missing, winsorize outliers (1–99th percentile)
- Full correlation heatmap with annotated values
- Target correlation ranking (top features correlated with `evimax_avg`)
- **Variance Inflation Factor (VIF)** analysis with green/yellow/red categorization
- Visualization of problematic features (VIF > 5 flagged as moderate; >10 as critical)

---

### 3. `productivity_classifcation_analysis.ipynb` — Zone-Aware Productivity Classifier
Builds a full **Decision Support System (DSS)** for 3-class agricultural productivity prediction with interactive mapping.

**Key steps:**
- Feature engineering: soil quality aggregation (`sq1/sq2`, `sq5/sq6` averaged), boolean-to-int conversion
- Correlation heatmap for numeric variables
- **KMeans ecological clustering** (7 clusters) on climate & elevation features (`af_bio_1_x`, `af_bio_12_x`, `srtm_1k`) with Silhouette scoring
- Target: `evimax_class` — Low / Medium / High (quantile-based)
- **LightGBM classifier** with `GridSearchCV` (5-fold CV) tuning `n_estimators`, `learning_rate`, `num_leaves`
- Interactive **Folium map** of predicted vs. actual productivity by household location

---

### 4. `road_accessiblity_Analysis.ipynb` — Road Accessibility Analysis
Classifies households by **road access quality** and models the relationship between road distance and geospatial features.

**Key steps:**
- Standard preprocessing pipeline (missing value filtering, winsorizing)
- One-hot encoding of `ssa_aez09` and `landcov`
- **Road access classification** using distance bins:
  - Very Good: < 2 km
  - Good: 2–5 km
  - Moderate: 5–10 km
  - Poor: > 10 km
- Correlation heatmap and VIF analysis with `dist_road` as target
- Top 20 features correlated with road distance visualized

---

## 📊 Dataset

| Field | Details |
|---|---|
| **Source** | Ethiopian Socioeconomic Survey (ESS) / LSMS-ISA — Wave 5 |
| **File** | `eth_householdgeovariables_y5.csv` |
| **Key Features** | Distance variables, climate indices, terrain, land cover, population density, soil quality, vegetation indices (EVI) |

---

## 🛠️ Tech Stack & Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn geopandas folium statsmodels
```

| Library | Usage |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, PCA, KMeans, GridSearchCV, classification |
| `lightgbm` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE oversampling |
| `statsmodels` | VIF calculation |
| `matplotlib`, `seaborn` | Static visualizations |
| `geopandas`, `folium` | Geospatial analysis & interactive maps |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up data
Upload `eth_householdgeovariables_y5.csv` to your Google Drive at:
```
MyDrive/other/data/Data for MSC Thesis/eth_householdgeovariables_y5.csv
```

### 4. Run in Google Colab
Each notebook mounts Google Drive automatically:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Open any notebook in [Google Colab](https://colab.research.google.com/) and run all cells.

---

## 🔬 Methodology Overview

```
Raw Data (eth_householdgeovariables_y5.csv)
        │
        ▼
Preprocessing
  ├── Drop columns > 40% missing
  ├── Winsorize outliers (1st–99th percentile)
  └── One-hot encode categorical features
        │
        ├──► Accessibility Analysis
        │       ├── PCA → Accessibility Index
        │       ├── Weighted Risk Score
        │       └── 3-class Classification + SMOTE
        │
        ├──► Agricultural Productivity
        │       ├── Correlation Heatmap
        │       └── VIF Multicollinearity Analysis
        │
        ├──► Productivity Classification
        │       ├── KMeans Ecological Clustering
        │       ├── LightGBM + GridSearchCV
        │       └── Interactive Folium Map
        │
        └──► Road Accessibility
                ├── Distance-based Binning
                ├── Correlation Analysis
                └── VIF Analysis
```

---

## 📈 Key Results Summary

- **Accessibility Index** built from 5 distance dimensions via PCA captures rural isolation
- **Road access quality** ranges from Very Good (<2 km) to Poor (>10 km) from the nearest road
- **LightGBM** with zone-aware ecological clustering achieves competitive accuracy on 3-class productivity prediction
- **VIF analysis** identifies and flags multicollinear features (threshold: VIF > 10)

---

## 📌 Notes
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
- All notebooks are designed to run on **Google Colab** with Google Drive mounted
- The dataset path is hardcoded — update `file_path` if your Drive structure differs
- SMOTE is applied only to training data to prevent data leakage
- Spatial coordinates (`lat_dd_mod`, `lon_dd_mod`) are excluded from model features to avoid spatial leakage


## 🙏 Acknowledgements

- Data sourced from the [Ethiopian Statistical Service]
- Built and run on [Google Colab](https://colab.research.google.com/)
