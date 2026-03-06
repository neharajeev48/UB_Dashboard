# 🏦 Universal Bank — Personal Loan Intelligence Suite

A comprehensive, multi-tab Streamlit dashboard for analysing personal loan acceptance patterns among Universal Bank's 5,000 customers.

## 🎯 Objective

> **Which customers are likely to accept a personal loan offer?**

The dashboard addresses this through four analytical lenses and prescribes personalised offers to high-propensity customers.

---

## 📊 Dashboard Structure

| Tab | Type | Description |
|-----|------|-------------|
| 📊 Descriptive Analysis | *What happened?* | Distribution of demographics, financial behaviour, product holdings, and loan acceptance across all segments |
| 🔍 Diagnostic Analysis | *Why did it happen?* | Correlations, Chi-Square tests, drill-down sunburst charts, and segment deep-dives |
| 🤖 Predictive Analysis | *What will happen?* | 3 ML models (Logistic Regression, Random Forest, Gradient Boosting), ROC curves, feature importance, live propensity simulator |
| 💊 Prescriptive & Offers | *What should we do?* | Personalised offer assignment, hot prospects table, strategic recommendations, ROI matrix |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ub-loan-dashboard.git
cd ub-loan-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Make sure `UniversalBank.csv` is in the same directory as `app.py`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (ensure `UniversalBank.csv` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

---

## 📁 File Structure

```
ub-loan-dashboard/
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Dataset (5,000 customers × 14 features)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

---

## 🗂️ Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| ID | Integer | Customer identifier |
| Age | Integer | Customer age in years |
| Experience | Integer | Years of professional experience |
| Income | Integer | Annual income in $K |
| ZIP Code | Integer | Home ZIP code |
| Family | Integer | Family size (1–4) |
| CCAvg | Float | Average monthly credit card spend ($K) |
| Education | Integer | 1=Undergrad, 2=Graduate, 3=Advanced/Prof |
| Mortgage | Integer | Mortgage value ($K) |
| **Personal Loan** | **Binary** | **Target: 1=Accepted, 0=Not Accepted** |
| Securities Account | Binary | 1=Has securities account |
| CD Account | Binary | 1=Has certificate of deposit |
| Online | Binary | 1=Uses online banking |
| CreditCard | Binary | 1=Has bank credit card |

---

## 🔬 Key Findings

- **9.6% base acceptance rate** — highly selective campaign opportunity
- **Income** is the #1 predictor (r≈0.50) — customers >$100K accept at ~30%
- **CD Account holders** accept at ~29% — 3× the base rate; use CD maturity as trigger
- **CC Spend >$3K/month** strongly co-occurs with loan acceptance
- **Random Forest AUC ≈ 0.98** — near-perfect predictability from available features
- **~500 high-propensity customers** (>60% probability, no existing loan) represent the core campaign list

---

## 🤖 Models Used

- **Logistic Regression** — baseline interpretable model
- **Random Forest** (200 trees, class-weight balanced) — primary production model
- **Gradient Boosting** (200 estimators) — ensemble cross-check

All models evaluated via **5-fold cross-validated AUC**.

---

## 💊 Offer Tiers

| Tier | Probability | Offer Strategy |
|------|-------------|---------------|
| 🔥 Very Hot | 80%+ | Pre-approved express loan, 0 processing fee, 1-click accept |
| 🟡 Hot | 60–80% | Incentive bundle (cashback / CC upgrade), targeted messaging |
| 🟣 Interested | 40–60% | Soft pre-qualification, EMI calculator, financial health check |
| 🔵 Warm | 20–40% | Education campaign, loan awareness content |
| ❄️ Cold | <20% | Relationship deepening, re-assess in 6 months |
| ✅ Existing | Has loan | Top-up / refinancing at preferential rates |

---

## 🛠 Tech Stack

- **Streamlit** — dashboard framework
- **Plotly** — interactive charts (sunburst drill-down, funnel, scatter, heatmap, gauge)
- **scikit-learn** — ML models
- **SciPy** — statistical tests (Chi-square, t-test)
- **Pandas / NumPy** — data wrangling

---

*Built as a comprehensive analytics showcase for Universal Bank's personal loan campaign optimisation.*
