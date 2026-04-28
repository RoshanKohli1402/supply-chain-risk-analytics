# 🚚 Supply Chain Risk & Disruption Analytics System

> An end-to-end data analytics platform for predicting delivery delays, scoring supplier risk, and forecasting demand — built on 39K+ real supply chain transactions.

---

## 📌 Problem Statement

Global supply chains suffer from reactive decision-making. By the time a disruption is visible, financial damage is already done. This system shifts that — using machine learning to **predict late deliveries before they happen**, score suppliers by risk tier, and forecast demand patterns to enable proactive procurement decisions.

**Key findings from the data:**
- 57.7% of all orders were delivered late
- Second Class shipping averages 2 days late — worst across all modes
- Pet Shop, Book Shop & Technology departments are highest risk
- 18.6% of orders (7,383) generated negative profit

---

## 🏗️ System Architecture

```
Raw Data (DataCo 39K records)
        ↓
Data Cleaning & Feature Engineering (Python / Pandas)
        ↓
    ┌─────────────────────────────────┐
    │         ML Pipeline             │
    │  • XGBoost Late Delivery Model  │
    │  • Supplier Risk Scoring        │
    │  • Prophet Demand Forecasting   │
    └─────────────────────────────────┘
        ↓
REST API Layer (FastAPI) ← [Partner: Backend]
        ↓
Power BI Dashboard + React Frontend
```

---

## 📊 ML Models & Results

### 1. Late Delivery Prediction (XGBoost)
Predicts whether an order will be delivered late before it ships.

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.77** |
| Accuracy | 71% |
| Precision (Late) | 80% |
| Recall (Late) | 66% |

**Top predictive features (SHAP):**
- `shipping_risk_score` — shipping mode is the dominant signal
- `days_shipping_scheduled` — longer planned windows = higher risk
- `customer_segment` — segment behavior matters more than discounts

### 2. Supplier Risk Scorecard
Composite risk score (0–100) per department using weighted metrics:
- Late delivery rate (40%)
- Average delay days (30%)
- Loss order rate (20%)
- Discount pressure (10%)

| Department | Risk Score | Tier |
|------------|-----------|------|
| Pet Shop | 82.7 | 🔴 High Risk |
| Book Shop | 75.6 | 🔴 High Risk |
| Technology | 70.1 | 🔴 High Risk |
| Fan Shop | 59.7 | 🟡 Medium Risk |
| Golf | 18.0 | 🟢 Low Risk |

### 3. Demand Forecasting (Prophet)
90-day forward forecast on daily order volume with seasonality decomposition.

**Key patterns:**
- Thursday/Friday = peak order days
- January–February seasonal dip every year
- Slight upward demand trend into 2018

---

## 🗂️ Repository Structure

```
supply-chain-risk-analytics/
│
├── data/
│   ├── supply_chain_cleaned.csv       # Cleaned dataset (39,682 rows)
│   └── supplier_risk_scorecard.csv    # Risk scores by department
│
├── models/
│   ├── xgb_risk_model.pkl             # Trained XGBoost model
│   └── minmax_scaler.pkl              # Feature scaler
│
├── notebooks/
│   └── supply_chain_analysis.ipynb    # Full EDA + ML pipeline
│
├── outputs/
│   ├── eda_overview.png               # 6-panel EDA dashboard
│   ├── shap_importance.png            # Feature importance (SHAP)
│   ├── shap_beeswarm.png              # SHAP value distribution
│   ├── supplier_risk_scorecard.png    # Risk scorecard visualization
│   ├── demand_forecast.png            # Prophet forecast chart
│   └── forecast_components.png       # Trend + seasonality breakdown
│
├── forecasts/
│   └── demand_forecast.csv            # 90-day forecast output
│
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Forecasting | Prophet (Meta) |
| Visualization | Matplotlib, Seaborn |
| Backend API | FastAPI, Spring Boot |
| Database | PostgreSQL |
| BI Dashboard | Power BI |
| Deployment | Docker |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/RoshanKohli1402/supply-chain-risk-analytics.git
cd supply-chain-risk-analytics

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap prophet matplotlib seaborn joblib

# Run the notebook
jupyter notebook notebooks/supply_chain_analysis.ipynb
```

---

## 📈 Dataset

**DataCo Smart Supply Chain Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- Records: 39,683 transactions
- Period: January 2015 – January 2018
- Features: 53 original columns → 38 after cleaning

---

## 👥 Team

| Name | Role | Scope |
|------|------|-------|
| Roshan Kohli | Data & Analytics | EDA, ML Pipeline, Risk Scoring, Demand Forecasting, Power BI |
| Vanshika Pandey | Backend Engineering | FastAPI/Spring Boot, PostgreSQL Schema, REST APIs, Docker |

---

## 🔗 Related Projects

- [Credit Risk Scoring System](https://github.com/RoshanKohli1402/credit-risk-scoring-system) — Loan default prediction with IFRS 9-aligned reporting
- [Regulatory Compliance Platform](https://github.com/RoshanKohli1402/regulatory-compliance-platform) — Real-time regulatory intelligence dashboard
