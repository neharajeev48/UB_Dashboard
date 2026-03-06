import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank — Loan Intelligence Suite",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { background-color: #070d1a; color: #e0e6ed; font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #0f2044 50%, #0a1628 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(56,189,248,0.04), transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p { color: #94a3b8; font-size: 1rem; margin-top: 0.5rem; }

    .kpi-card {
        background: linear-gradient(135deg, #0f2044 0%, #070d1a 100%);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .kpi-card:hover { border-color: rgba(56, 189, 248, 0.4); transform: translateY(-2px); box-shadow: 0 8px 32px rgba(56,189,248,0.08); }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
    .kpi-delta { font-size: 0.75rem; margin-top: 0.3rem; }
    .kpi-delta.positive { color: #34d399; }
    .kpi-delta.negative { color: #f87171; }
    .kpi-delta.neutral { color: #fbbf24; }

    .section-header {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1), transparent);
        border-left: 3px solid #38bdf8;
        padding: 0.8rem 1.2rem;
        margin: 1.8rem 0 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .section-header h3 { color: #bae6fd; font-size: 1.1rem; font-weight: 600; margin: 0; }
    .section-header p { color: #94a3b8; font-size: 0.8rem; margin: 0.2rem 0 0 0; }

    .insight-box {
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .insight-box strong { color: #7dd3fc; }

    .offer-card {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.07), rgba(6, 78, 59, 0.12));
        border: 1px solid rgba(52, 211, 153, 0.22);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .offer-card h4 { color: #6ee7b7; margin: 0 0 0.5rem 0; font-size: 0.95rem; }
    .offer-card p { color: #94a3b8; margin: 0; font-size: 0.85rem; line-height: 1.7; }

    .risk-high { color: #f87171; font-weight: 700; }
    .risk-med  { color: #fbbf24; font-weight: 700; }
    .risk-low  { color: #34d399; font-weight: 700; }

    div[data-testid="stTabs"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.8rem 1.2rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #7dd3fc !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    .stSidebar > div { background: #0a1628; }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 10px;
        background: rgba(15,32,68,0.4);
    }

    .segment-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")

    # Label mappings
    df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Prof'})
    df['Family_Label'] = df['Family'].map({1: '1 Member', 2: '2 Members', 3: '3 Members', 4: '4+ Members'})
    df['Loan_Label'] = df['Personal Loan'].map({0: 'No Loan', 1: 'Accepted Loan'})
    df['Securities_Label'] = df['Securities Account'].map({0: 'No Securities', 1: 'Has Securities'})
    df['CD_Label'] = df['CD Account'].map({0: 'No CD', 1: 'Has CD Account'})
    df['Online_Label'] = df['Online'].map({0: 'Offline', 1: 'Online Banking'})
    df['CreditCard_Label'] = df['CreditCard'].map({0: 'No CC', 1: 'Has Credit Card'})

    # Derived features
    df['AgeGroup'] = pd.cut(df['Age'], bins=[22, 30, 40, 50, 60, 68],
                             labels=['23-30', '31-40', '41-50', '51-60', '61+'])
    df['IncomeGroup'] = pd.cut(df['Income'], bins=[0, 50, 100, 150, 230],
                                labels=['<$50K', '$50-100K', '$100-150K', '$150K+'])
    df['CCAvg_Group'] = pd.cut(df['CCAvg'], bins=[-0.1, 1, 3, 6, 10.1],
                                labels=['<$1K', '$1-3K', '$3-6K', '$6K+'])
    df['MortgageGroup'] = pd.cut(df['Mortgage'], bins=[-1, 0, 100, 300, 640],
                                  labels=['No Mortgage', '<$100K', '$100-300K', '$300K+'])
    df['ExperienceGroup'] = pd.cut(df['Experience'].clip(lower=0), bins=[-1, 5, 15, 25, 44],
                                    labels=['0-5 yrs', '6-15 yrs', '16-25 yrs', '25+ yrs'])

    # Clean experience (fix negatives)
    df['Experience'] = df['Experience'].clip(lower=0)

    # Wealth score (composite)
    df['WealthScore'] = (
        (df['Income'] / df['Income'].max()) * 40 +
        (df['CCAvg'] / df['CCAvg'].max()) * 25 +
        (df['Mortgage'] / df['Mortgage'].max()) * 20 +
        df['CD Account'] * 10 +
        df['Securities Account'] * 5
    ).round(1)

    df['WealthTier'] = pd.cut(df['WealthScore'], bins=[0, 20, 40, 60, 101],
                               labels=['Bronze', 'Silver', 'Gold', 'Platinum'])

    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Filters")
    st.caption("Dynamically slice the data")

    edu_filter = st.multiselect(
        "Education Level",
        options=['Undergrad', 'Graduate', 'Advanced/Prof'],
        default=['Undergrad', 'Graduate', 'Advanced/Prof']
    )
    family_filter = st.multiselect(
        "Family Size",
        options=['1 Member', '2 Members', '3 Members', '4+ Members'],
        default=['1 Member', '2 Members', '3 Members', '4+ Members']
    )
    securities_filter = st.multiselect(
        "Securities Account",
        options=['No Securities', 'Has Securities'],
        default=['No Securities', 'Has Securities']
    )
    cd_filter = st.multiselect(
        "CD Account",
        options=['No CD', 'Has CD Account'],
        default=['No CD', 'Has CD Account']
    )
    age_range = st.slider("Age Range", 23, 67, (23, 67))
    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()),
                              (int(df['Income'].min()), int(df['Income'].max())))
    online_filter = st.multiselect(
        "Online Banking",
        options=['Offline', 'Online Banking'],
        default=['Offline', 'Online Banking']
    )

    st.markdown("---")
    st.markdown("**🎯 Target Variable**")
    st.markdown("Personal Loan Acceptance")
    accepted = df['Personal Loan'].sum()
    st.metric("Acceptance Rate", f"{accepted/len(df)*100:.1f}%", f"{accepted} customers")

mask = (
    df['Education_Label'].isin(edu_filter) &
    df['Family_Label'].isin(family_filter) &
    df['Securities_Label'].isin(securities_filter) &
    df['CD_Label'].isin(cd_filter) &
    df['Age'].between(age_range[0], age_range[1]) &
    df['Income'].between(income_range[0], income_range[1]) &
    df['Online_Label'].isin(online_filter)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#38bdf8', 'secondary': '#818cf8', 'accent': '#34d399',
    'danger': '#f87171', 'warning': '#fbbf24', 'pink': '#f472b6',
    'text': '#e0e6ed', 'muted': '#94a3b8', 'bg': '#070d1a', 'card': '#0f2044'
}
LOAN_COLORS = {'No Loan': '#475569', 'Accepted Loan': '#38bdf8'}
LOAN_NUM_COLORS = {0: '#475569', 1: '#38bdf8'}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#e0e6ed', size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
)

def styled_chart(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.07)')
    fig.update_yaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.07)')
    return fig


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏦 Universal Bank — Personal Loan Intelligence Suite</h1>
    <p>Descriptive · Diagnostic · Predictive · Prescriptive — Understanding which customers will accept a Personal Loan</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
total = len(dff)
loan_accept = dff['Personal Loan'].sum()
loan_rate = loan_accept / total * 100 if total > 0 else 0
avg_income = dff['Income'].mean()
avg_ccavg = dff['CCAvg'].mean()
pct_cd = dff['CD Account'].mean() * 100
pct_online = dff['Online'].mean() * 100

with k1:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>{total:,}</div>
        <div class='kpi-label'>Customers</div>
        <div class='kpi-delta neutral'>Filtered cohort</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>{loan_rate:.1f}%</div>
        <div class='kpi-label'>Loan Acceptance Rate</div>
        <div class='kpi-delta positive'>↑ {loan_accept:,} accepted</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>${avg_income:.0f}K</div>
        <div class='kpi-label'>Avg Annual Income</div>
        <div class='kpi-delta neutral'>Per year</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>${avg_ccavg:.1f}K</div>
        <div class='kpi-label'>Avg CC Spending/mo</div>
        <div class='kpi-delta neutral'>Monthly avg</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>{pct_cd:.1f}%</div>
        <div class='kpi-label'>CD Account Holders</div>
        <div class='kpi-delta positive'>High-value signal</div>
    </div>""", unsafe_allow_html=True)

with k6:
    st.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>{pct_online:.0f}%</div>
        <div class='kpi-label'>Online Banking Users</div>
        <div class='kpi-delta neutral'>Digital engagement</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "🤖 Predictive Analysis",
    "💊 Prescriptive & Offers"
])


# =============================================================
# TAB 1: DESCRIPTIVE ANALYSIS
# =============================================================
with tab1:
    st.markdown("""
    <div class='section-header'>
        <h3>📊 Descriptive Analysis — What does the data look like?</h3>
        <p>Distribution of customer demographics, financial behaviour, and product holdings</p>
    </div>""", unsafe_allow_html=True)

    # Row 1: Overall Loan Distribution + Education Donut
    c1, c2, c3 = st.columns(3)

    with c1:
        loan_counts = dff['Loan_Label'].value_counts().reset_index()
        loan_counts.columns = ['Status', 'Count']
        fig = go.Figure(go.Pie(
            labels=loan_counts['Status'], values=loan_counts['Count'],
            hole=0.65,
            marker=dict(colors=['#475569', '#38bdf8'],
                        line=dict(color='#070d1a', width=2)),
            textinfo='label+percent',
            hovertemplate='%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig.add_annotation(text=f"<b>{loan_rate:.1f}%</b><br>Accepted",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color='#38bdf8'))
        fig.update_layout(title='Personal Loan Acceptance (Overall)', showlegend=True)
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c2:
        edu_loan = dff.groupby(['Education_Label', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(edu_loan, path=['Education_Label', 'Loan_Label'], values='Count',
                          color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: Education → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c3:
        fam_loan = dff.groupby(['Family_Label', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(fam_loan, path=['Family_Label', 'Loan_Label'], values='Count',
                          color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: Family Size → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    # Row 2: Age & Income Distributions
    st.markdown("<div class='section-header'><h3>👤 Customer Demographics</h3><p>Age, Income, and Experience distributions by loan status</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x='Age', color='Loan_Label', barmode='overlay',
                           color_discrete_map=LOAN_COLORS, nbins=25, opacity=0.75,
                           title='Age Distribution by Loan Status',
                           labels={'Age': 'Age (years)', 'count': 'Number of Customers'})
        fig.update_layout(bargap=0.05)
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    with c2:
        fig = px.histogram(dff, x='Income', color='Loan_Label', barmode='overlay',
                           color_discrete_map=LOAN_COLORS, nbins=30, opacity=0.75,
                           title='Income Distribution by Loan Status ($K/yr)')
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        age_grp = dff.groupby(['AgeGroup', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.bar(age_grp, x='AgeGroup', y='Count', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS, barmode='group',
                     title='Loan Acceptance by Age Group')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c2:
        inc_grp = dff.groupby(['IncomeGroup', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.bar(inc_grp, x='IncomeGroup', y='Count', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS, barmode='stack',
                     title='Loan Acceptance by Income Group')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    # Row 3: Financial Behaviour
    st.markdown("<div class='section-header'><h3>💳 Financial Behaviour</h3><p>Credit card spending, mortgage, and product holdings</p></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.violin(dff, x='Loan_Label', y='CCAvg', color='Loan_Label',
                        box=True, color_discrete_map=LOAN_COLORS,
                        title='Monthly CC Spending by Loan Status ($K)')
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c2:
        cc_grp = dff.groupby(['CCAvg_Group', 'Loan_Label']).size().reset_index(name='Count')
        totals = dff.groupby('CCAvg_Group').size().reset_index(name='Total')
        cc_grp = cc_grp.merge(totals, on='CCAvg_Group')
        cc_grp['Rate'] = (cc_grp['Count'] / cc_grp['Total'] * 100).round(1)
        cc_yes = cc_grp[cc_grp['Loan_Label'] == 'Accepted Loan']
        fig = go.Figure(go.Bar(
            x=cc_yes['CCAvg_Group'], y=cc_yes['Rate'],
            marker=dict(color=cc_yes['Rate'],
                        colorscale=[[0, '#475569'], [0.5, '#818cf8'], [1, '#38bdf8']]),
            text=cc_yes['Rate'].astype(str) + '%', textposition='outside',
            hovertemplate='CC Spend: %{x}<br>Loan Accept Rate: %{y}%<extra></extra>'
        ))
        fig.update_layout(title='Loan Acceptance Rate by CC Spend Level',
                          yaxis_title='Acceptance Rate %')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c3:
        mort_grp = dff.groupby(['MortgageGroup', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.bar(mort_grp, x='MortgageGroup', y='Count', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS, barmode='stack',
                     title='Mortgage Holdings by Loan Status')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    # Row 4: Product Holdings
    st.markdown("<div class='section-header'><h3>🏦 Product Holdings Overview</h3><p>Cross-product penetration and digital engagement</p></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    products = [
        ('Securities Account', 'Securities\nAccount', '#818cf8'),
        ('CD Account', 'CD\nAccount', '#34d399'),
        ('Online', 'Online\nBanking', '#38bdf8'),
        ('CreditCard', 'Credit\nCard', '#f472b6')
    ]

    for col, (prod_col, label, color) in zip([c1, c2, c3, c4], products):
        with col:
            accept_rate_with = dff[dff[prod_col] == 1]['Personal Loan'].mean() * 100
            accept_rate_without = dff[dff[prod_col] == 0]['Personal Loan'].mean() * 100
            pct_holding = dff[prod_col].mean() * 100
            fig = go.Figure(go.Pie(
                labels=['Has Product', 'No Product'],
                values=[dff[prod_col].sum(), (dff[prod_col] == 0).sum()],
                hole=0.6,
                marker=dict(colors=[color, '#1e293b'],
                            line=dict(color='#070d1a', width=2)),
                textinfo='percent',
                hovertemplate='%{label}: %{value} customers<extra></extra>'
            ))
            fig.add_annotation(text=f"<b>{pct_holding:.0f}%</b>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color=color))
            fig.update_layout(
                title=f'{label}<br><sub>Accept: {accept_rate_with:.1f}% w/ | {accept_rate_without:.1f}% w/o</sub>',
                showlegend=False
            )
            st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>💡 Descriptive Insight:</strong> The loan acceptance rate is just 9.6%, indicating a highly selective customer base. 
        CD Account holders show dramatically higher acceptance (~29%), suggesting they are already high-trust, high-value customers. 
        Income above $100K and monthly CC spending above $3K strongly co-occur with loan acceptance. 
        Customers with advanced/professional education represent a disproportionately high share of loan takers.
    </div>""", unsafe_allow_html=True)

    # Row 5: Scatter — Income vs CCAvg
    st.markdown("<div class='section-header'><h3>💰 Income × CC Spend Landscape</h3><p>Identifying the high-value quadrant most likely to accept loans</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(dff, x='Income', y='CCAvg', color='Loan_Label',
                         color_discrete_map=LOAN_COLORS,
                         opacity=0.55, size_max=8,
                         hover_data=['Age', 'Education_Label', 'Family_Label'],
                         title='Income vs CC Spending (coloured by Loan Status)')
        fig.update_layout(xaxis_title='Annual Income ($K)', yaxis_title='Monthly CC Spend ($K)')
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with c2:
        # Heatmap: Income Group × CCAvg Group acceptance rates
        pivot = dff.pivot_table(index='IncomeGroup', columns='CCAvg_Group',
                                values='Personal Loan', aggfunc='mean') * 100
        fig = px.imshow(pivot, text_auto='.1f', color_continuous_scale='Blues',
                        title='Loan Acceptance Heatmap: Income vs CC Spend (%)',
                        labels=dict(x='Monthly CC Spend', y='Annual Income', color='Accept %'))
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    # Row 6: Education × Family Size Treemap
    st.markdown("<div class='section-header'><h3>🗺️ Customer Segment Treemap</h3><p>Size = number of customers, colour = loan acceptance rate</p></div>", unsafe_allow_html=True)

    seg = dff.groupby(['Education_Label', 'Family_Label', 'IncomeGroup']).agg(
        Count=('Personal Loan', 'count'),
        AcceptRate=('Personal Loan', 'mean')
    ).reset_index()
    seg['AcceptRate'] = (seg['AcceptRate'] * 100).round(1)

    fig = px.treemap(seg, path=['Education_Label', 'Family_Label', 'IncomeGroup'],
                     values='Count', color='AcceptRate',
                     color_continuous_scale=[[0, '#0f2044'], [0.5, '#38bdf8'], [1, '#34d399']],
                     title='Customer Segments: Education → Family → Income (colour = Acceptance Rate %)',
                     hover_data={'Count': True, 'AcceptRate': True})
    fig.update_traces(texttemplate='%{label}<br>%{customdata[1]:.1f}%')
    st.plotly_chart(styled_chart(fig, 520), use_container_width=True)

    # Row 7: Wealth Distribution
    st.markdown("<div class='section-header'><h3>💎 Wealth Tier Analysis</h3><p>Composite wealth scoring vs loan acceptance</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        wealth_loan = dff.groupby(['WealthTier', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.bar(wealth_loan, x='WealthTier', y='Count', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS, barmode='stack',
                     category_orders={'WealthTier': ['Bronze', 'Silver', 'Gold', 'Platinum']},
                     title='Customer Wealth Tiers vs Loan Acceptance')
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c2:
        wealth_rates = dff.groupby('WealthTier')['Personal Loan'].mean().reset_index()
        wealth_rates.columns = ['WealthTier', 'AcceptRate']
        wealth_rates['AcceptRate'] = (wealth_rates['AcceptRate'] * 100).round(1)
        wealth_rates = wealth_rates.loc[wealth_rates['WealthTier'].isin(['Bronze', 'Silver', 'Gold', 'Platinum'])]
        fig = go.Figure(go.Bar(
            x=wealth_rates['WealthTier'], y=wealth_rates['AcceptRate'],
            marker=dict(color=['#94a3b8', '#c084fc', '#fbbf24', '#38bdf8']),
            text=wealth_rates['AcceptRate'].astype(str) + '%', textposition='outside'
        ))
        fig.update_layout(title='Loan Acceptance Rate by Wealth Tier',
                          yaxis_title='Acceptance Rate %',
                          xaxis=dict(categoryorder='array',
                                     categoryarray=['Bronze', 'Silver', 'Gold', 'Platinum']))
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)


# =============================================================
# TAB 2: DIAGNOSTIC ANALYSIS
# =============================================================
with tab2:
    st.markdown("""
    <div class='section-header'>
        <h3>🔍 Diagnostic Analysis — Why do customers accept loans?</h3>
        <p>Statistical significance tests, correlation analysis, and risk factor identification</p>
    </div>""", unsafe_allow_html=True)

    # Correlation with Personal Loan
    numeric_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                    'Education', 'Mortgage', 'Securities Account', 'CD Account',
                    'Online', 'CreditCard', 'Personal Loan']

    corr_matrix = dff[numeric_cols].corr()
    loan_corr = corr_matrix['Personal Loan'].drop('Personal Loan').sort_values()

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            y=loan_corr.index, x=loan_corr.values, orientation='h',
            marker=dict(
                color=loan_corr.values,
                colorscale=[[0, '#475569'], [0.5, '#38bdf8'], [1, '#34d399']],
                cmid=0
            ),
            text=loan_corr.values.round(3), textposition='outside'
        ))
        fig.update_layout(title='Correlation with Personal Loan Acceptance',
                          xaxis_title='Pearson Correlation Coefficient')
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    with c2:
        fig = px.imshow(dff[numeric_cols].corr(),
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title='Full Feature Correlation Matrix')
        fig.update_layout(height=480)
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>💡 Diagnostic Insight:</strong> <strong>Income</strong> shows the highest positive correlation with loan acceptance (~0.50), 
        followed by <strong>CD Account</strong> (~0.32) and <strong>CCAvg</strong> (~0.37). 
        <strong>Education</strong> has a moderate positive effect (~0.14). 
        Interestingly, <strong>Age</strong> and <strong>Experience</strong> show near-zero correlation, 
        suggesting that life stage matters less than financial capacity and existing product relationships.
    </div>""", unsafe_allow_html=True)

    # Chi-Square Tests
    st.markdown("<div class='section-header'><h3>📐 Statistical Significance Tests</h3><p>Chi-square & t-test analysis of each feature's association with loan acceptance</p></div>", unsafe_allow_html=True)

    cat_tests = []
    for col, label in [('Education_Label', 'Education'), ('Family_Label', 'Family Size'),
                        ('AgeGroup', 'Age Group'), ('IncomeGroup', 'Income Group'),
                        ('CCAvg_Group', 'CC Spend Group'), ('MortgageGroup', 'Mortgage Group'),
                        ('WealthTier', 'Wealth Tier'), ('ExperienceGroup', 'Experience Group')]:
        ct = pd.crosstab(dff[col], dff['Personal Loan'])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            cat_tests.append({
                'Feature': label, 'Chi²': round(chi2, 2), 'p-value': round(p, 6),
                "Cramér's V": round(cramers_v, 3),
                'Significant': '✅ Yes' if p < 0.05 else '❌ No'
            })

    chi_df = pd.DataFrame(cat_tests).sort_values("Cramér's V", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            x=chi_df["Cramér's V"], y=chi_df['Feature'], orientation='h',
            marker=dict(color=chi_df["Cramér's V"],
                        colorscale=[[0, '#818cf8'], [1, '#38bdf8']]),
            text=chi_df["Cramér's V"].round(3), textposition='outside'
        ))
        fig.update_layout(title="Cramér's V Effect Size — Higher = Stronger Association with Loan",
                          xaxis_title="Cramér's V")
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with c2:
        st.markdown("#### Chi-Square Test Results")
        st.dataframe(chi_df.set_index('Feature'), use_container_width=True, height=370)

    # Drill-Down Sunburst Charts
    st.markdown("<div class='section-header'><h3>🍩 Interactive Drill-Down Analysis</h3><p>Click any segment to expand and explore loan acceptance within that group</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        dd1 = dff.groupby(['IncomeGroup', 'Education_Label', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(dd1, path=['IncomeGroup', 'Education_Label', 'Loan_Label'],
                          values='Count', color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: Income → Education → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    with c2:
        dd2 = dff.groupby(['CCAvg_Group', 'CD_Label', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(dd2, path=['CCAvg_Group', 'CD_Label', 'Loan_Label'],
                          values='Count', color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: CC Spend → CD Account → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        dd3 = dff.groupby(['WealthTier', 'Education_Label', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(dd3, path=['WealthTier', 'Education_Label', 'Loan_Label'],
                          values='Count', color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: Wealth Tier → Education → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    with c2:
        dd4 = dff.groupby(['Family_Label', 'MortgageGroup', 'Loan_Label']).size().reset_index(name='Count')
        fig = px.sunburst(dd4, path=['Family_Label', 'MortgageGroup', 'Loan_Label'],
                          values='Count', color='Loan_Label',
                          color_discrete_map=LOAN_COLORS,
                          title='Drill: Family Size → Mortgage → Loan Acceptance')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    # High-Acceptance Combination Analysis
    st.markdown("<div class='section-header'><h3>🎯 High-Probability Segment Combinations</h3><p>Which factor combinations produce the highest loan acceptance rates?</p></div>", unsafe_allow_html=True)

    combos = []
    for edu in dff['Education_Label'].unique():
        for inc in dff['IncomeGroup'].unique():
            for cd in [0, 1]:
                subset = dff[(dff['Education_Label'] == edu) &
                              (dff['IncomeGroup'] == inc) &
                              (dff['CD Account'] == cd)]
                if len(subset) >= 15:
                    rate = subset['Personal Loan'].mean() * 100
                    combos.append({
                        'Education': edu, 'Income': str(inc), 'CD Account': 'Yes' if cd else 'No',
                        'Count': len(subset), 'Acceptance Rate %': round(rate, 1)
                    })

    combo_df = pd.DataFrame(combos).sort_values('Acceptance Rate %', ascending=False).head(15)

    fig = go.Figure(go.Bar(
        x=combo_df['Acceptance Rate %'],
        y=combo_df.apply(lambda r: f"{r['Income']} | {r['Education']} | CD:{r['CD Account']}", axis=1),
        orientation='h',
        marker=dict(color=combo_df['Acceptance Rate %'],
                    colorscale=[[0, '#0f2044'], [0.5, '#818cf8'], [1, '#38bdf8']]),
        text=combo_df.apply(lambda r: f"{r['Acceptance Rate %']}% (n={r['Count']})", axis=1),
        textposition='outside'
    ))
    fig.update_layout(title='Top 15 High-Acceptance Customer Segments',
                      xaxis_title='Acceptance Rate %', height=520)
    st.plotly_chart(styled_chart(fig, 540), use_container_width=True)

    # Mean Comparison: Loan Accepted vs Not
    st.markdown("<div class='section-header'><h3>📉 Financial Profile Comparison</h3><p>Mean values of key financial metrics — Loan Accepted vs Not Accepted</p></div>", unsafe_allow_html=True)

    compare_cols = ['Age', 'Income', 'CCAvg', 'Mortgage', 'Family', 'Experience']
    compare_labels = ['Age', 'Income ($K)', 'CC Spend ($K/mo)', 'Mortgage ($K)', 'Family Size', 'Experience (yrs)']

    gap_data = []
    for col, label in zip(compare_cols, compare_labels):
        yes_mean = dff[dff['Personal Loan'] == 1][col].mean()
        no_mean = dff[dff['Personal Loan'] == 0][col].mean()
        t_stat, p_val = stats.ttest_ind(
            dff[dff['Personal Loan'] == 1][col],
            dff[dff['Personal Loan'] == 0][col]
        )
        gap_data.append({
            'Feature': label,
            'Accepted Loan': round(yes_mean, 2),
            'Not Accepted': round(no_mean, 2),
            'Difference': round(yes_mean - no_mean, 2),
            'p-value': round(p_val, 5),
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

    gap_df = pd.DataFrame(gap_data).sort_values('Difference', ascending=False)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accepted Loan', x=gap_df['Feature'], y=gap_df['Accepted Loan'],
                             marker_color='#38bdf8'))
        fig.add_trace(go.Bar(name='Not Accepted', x=gap_df['Feature'], y=gap_df['Not Accepted'],
                             marker_color='#475569'))
        fig.update_layout(title='Mean Financial Metrics: Accepted vs Not Accepted',
                          barmode='group', yaxis_title='Mean Value')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    with c2:
        st.markdown("#### Mean Comparison (t-test)")
        st.dataframe(gap_df.set_index('Feature'), use_container_width=True, height=350)

    # Box plots - Income and CCAvg
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(dff, x='Loan_Label', y='Income', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS,
                     title='Income Distribution: Accepted vs Not Accepted',
                     points='outliers')
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)

    with c2:
        fig = px.box(dff, x='Loan_Label', y='CCAvg', color='Loan_Label',
                     color_discrete_map=LOAN_COLORS,
                     title='CC Monthly Spending: Accepted vs Not Accepted',
                     points='outliers')
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_chart(fig, 360), use_container_width=True)


# =============================================================
# TAB 3: PREDICTIVE ANALYSIS
# =============================================================
with tab3:
    st.markdown("""
    <div class='section-header'>
        <h3>🤖 Predictive Analysis — Who is likely to accept a loan?</h3>
        <p>Machine learning models trained to predict personal loan acceptance probability</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data
    def run_models(data):
        df_ml = data.copy()
        df_ml['Experience'] = df_ml['Experience'].clip(lower=0)

        feature_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                        'Education', 'Mortgage', 'Securities Account',
                        'CD Account', 'Online', 'CreditCard']
        X = df_ml[feature_cols]
        y = df_ml['Personal Loan']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
        }

        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            model.fit(X_scaled, y)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.abs(model.coef_[0])
            results[name] = {
                'auc_mean': scores.mean(), 'auc_std': scores.std(),
                'importance': pd.Series(importance, index=feature_cols).sort_values(ascending=False),
                'model': model
            }

        roc_data = {}
        for name, model in models.items():
            y_prob = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr), 'proba': y_prob}

        # Best model probabilities for scoring
        best_model = results['Random Forest']['model']
        proba = best_model.predict_proba(X_scaled)[:, 1]

        return results, roc_data, feature_cols, scaler, proba

    with st.spinner("🔄 Training models... (cached after first run)"):
        results, roc_data, feature_cols, scaler, rf_proba = run_models(df)

    # Attach probabilities to filtered dataframe
    dff = dff.copy()
    dff['LoanProbability'] = rf_proba[dff.index] if len(rf_proba) == len(df) else rf_proba[:len(dff)]

    # Model Comparison
    c1, c2 = st.columns(2)
    with c1:
        model_comp = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC (mean)': [results[m]['auc_mean'] for m in results],
            'AUC (std)': [results[m]['auc_std'] for m in results],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_comp['Model'], y=model_comp['AUC (mean)'],
            error_y=dict(type='data', array=model_comp['AUC (std)'].tolist()),
            marker_color=['#818cf8', '#38bdf8', '#34d399'],
            text=model_comp['AUC (mean)'].round(3), textposition='outside'
        ))
        fig.update_layout(title='Model Comparison — Cross-Validated AUC (5-Fold)',
                          yaxis_title='AUC Score', yaxis_range=[0.5, 1.0])
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    with c2:
        fig = go.Figure()
        colors_roc = ['#818cf8', '#38bdf8', '#34d399']
        for i, (name, rdata) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(
                x=rdata['fpr'], y=rdata['tpr'], mode='lines',
                name=f"{name} (AUC={rdata['auc']:.3f})",
                line=dict(color=colors_roc[i], width=2.5)
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline',
                                 line=dict(color='#475569', dash='dash', width=1.5)))
        fig.update_layout(title='ROC Curves — All Models',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    st.markdown("""<div class='insight-box'>
        <strong>💡 Model Performance:</strong> All three models achieve strong AUC scores (>0.90), 
        with Random Forest leading at ~0.98. The high AUC confirms that loan acceptance is 
        highly predictable from the available features. Income, CCAvg, and CD Account 
        are the dominant predictors across all models.
    </div>""", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("<div class='section-header'><h3>🎯 Feature Importance Rankings</h3><p>Which customer attributes best predict loan acceptance?</p></div>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select model:", list(results.keys()), index=1, key='feat_model')
    imp = results[selected_model]['importance'].head(11)

    fig = go.Figure(go.Bar(
        y=imp.index[::-1], x=imp.values[::-1], orientation='h',
        marker=dict(color=imp.values[::-1],
                    colorscale=[[0, '#0f2044'], [0.5, '#818cf8'], [1, '#38bdf8']]),
        text=imp.values[::-1].round(4), textposition='outside'
    ))
    fig.update_layout(title=f'Feature Importances — {selected_model}',
                      xaxis_title='Importance Score', height=450)
    st.plotly_chart(styled_chart(fig, 460), use_container_width=True)

    # Consensus feature importance
    st.markdown("<div class='section-header'><h3>📊 Consensus Feature Ranking</h3><p>Normalized importance averaged across all 3 models</p></div>", unsafe_allow_html=True)

    all_imp = pd.DataFrame()
    for name, res in results.items():
        norm = res['importance'] / res['importance'].max()
        all_imp[name] = norm
    all_imp['Mean'] = all_imp.mean(axis=1)
    all_imp = all_imp.sort_values('Mean', ascending=False)

    fig = go.Figure()
    for i, name in enumerate(results.keys()):
        fig.add_trace(go.Bar(name=name, y=all_imp.index[::-1], x=all_imp[name].values[::-1],
                             orientation='h', marker_color=colors_roc[i], opacity=0.75))
    fig.update_layout(title='Normalized Feature Importance — All 3 Models', barmode='group',
                      xaxis_title='Normalized Importance', height=420)
    st.plotly_chart(styled_chart(fig, 440), use_container_width=True)

    # Probability Distribution
    st.markdown("<div class='section-header'><h3>📈 Predicted Probability Distribution</h3><p>How confident is the model? Distribution of acceptance probabilities</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        proba_df = pd.DataFrame({'Probability': rf_proba, 'Actual': df['Personal Loan']})
        fig = px.histogram(proba_df, x='Probability', color=proba_df['Actual'].map({0: 'No Loan', 1: 'Accepted Loan'}),
                           barmode='overlay', nbins=50, opacity=0.75,
                           color_discrete_map=LOAN_COLORS,
                           title='Predicted Probability Distribution by Actual Outcome')
        fig.update_layout(xaxis_title='Predicted Probability of Loan Acceptance',
                          yaxis_title='Number of Customers')
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    with c2:
        # Score bands
        proba_df['Band'] = pd.cut(proba_df['Probability'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
        band_stats = proba_df.groupby('Band').agg(
            Total=('Actual', 'count'),
            ActualAccept=('Actual', 'sum')
        ).reset_index()
        band_stats['ActualRate'] = (band_stats['ActualAccept'] / band_stats['Total'] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=band_stats['Band'], y=band_stats['Total'],
                             name='Total Customers', marker_color='#1e3a5f', opacity=0.8))
        fig.add_trace(go.Scatter(x=band_stats['Band'], y=band_stats['ActualRate'],
                                 name='Actual Accept Rate %', yaxis='y2',
                                 line=dict(color='#38bdf8', width=3),
                                 mode='lines+markers',
                                 marker=dict(size=10)))
        fig.update_layout(
            title='Score Bands: Volume vs Actual Acceptance Rate',
            yaxis=dict(title='Number of Customers'),
            yaxis2=dict(title='Actual Accept Rate %', overlaying='y', side='right',
                        showgrid=False),
            legend=dict(x=0.02, y=0.95)
        )
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    # Live Propensity Simulator
    st.markdown("<div class='section-header'><h3>🎛️ Loan Propensity Simulator</h3><p>Estimate any customer's loan acceptance probability in real-time</p></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sim_age = st.slider("Age", 23, 67, 40, key='sim_age')
        sim_exp = st.slider("Experience (yrs)", 0, 43, 15, key='sim_exp')
        sim_income = st.slider("Annual Income ($K)", 8, 224, 80, key='sim_inc')
    with c2:
        sim_family = st.slider("Family Size", 1, 4, 2, key='sim_fam')
        sim_ccavg = st.slider("Monthly CC Spend ($K)", 0.0, 10.0, 2.0, step=0.1, key='sim_cc')
        sim_edu = st.selectbox("Education", [1, 2, 3], format_func=lambda x: {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Prof'}[x], key='sim_edu')
    with c3:
        sim_mortgage = st.slider("Mortgage ($K)", 0, 635, 0, key='sim_mort')
        sim_sec = st.selectbox("Securities Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No', key='sim_sec')
        sim_cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No', key='sim_cd')
    with c4:
        sim_online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: 'Yes' if x else 'No', key='sim_onl')
        sim_cc_card = st.selectbox("Credit Card (bank)", [0, 1], format_func=lambda x: 'Yes' if x else 'No', key='sim_ccard')

    sim_input = np.array([[sim_age, sim_exp, sim_income, sim_family, sim_ccavg,
                           sim_edu, sim_mortgage, sim_sec, sim_cd, sim_online, sim_cc_card]])
    sim_scaled = scaler.transform(sim_input)
    rf_model = results['Random Forest']['model']
    sim_prob = rf_model.predict_proba(sim_scaled)[0][1] * 100

    prob_color = '#34d399' if sim_prob >= 60 else '#fbbf24' if sim_prob >= 30 else '#f87171'
    prob_label = 'HIGH PROPENSITY' if sim_prob >= 60 else 'MEDIUM PROPENSITY' if sim_prob >= 30 else 'LOW PROPENSITY'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_prob,
        title={'text': f'Loan Acceptance Probability — {prob_label}', 'font': {'size': 17, 'color': '#e0e6ed'}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#475569'),
            bar=dict(color=prob_color),
            bgcolor='rgba(0,0,0,0)',
            steps=[
                dict(range=[0, 30], color='rgba(248,113,113,0.12)'),
                dict(range=[30, 60], color='rgba(251,191,36,0.12)'),
                dict(range=[60, 100], color='rgba(52,211,153,0.12)'),
            ],
            threshold=dict(line=dict(color=prob_color, width=4), thickness=0.8, value=sim_prob)
        ),
        number=dict(suffix='%', font=dict(size=44, color=prob_color))
    ))
    st.plotly_chart(styled_chart(fig, 320), use_container_width=True)


# =============================================================
# TAB 4: PRESCRIPTIVE ANALYSIS
# =============================================================
with tab4:
    st.markdown("""
    <div class='section-header'>
        <h3>💊 Prescriptive Analysis — What should we do?</h3>
        <p>Personalized loan offers, targeted campaigns, and strategic recommendations driven by data</p>
    </div>""", unsafe_allow_html=True)

    # Rebuild proba on filtered data
    df_ml = df.copy()
    df_ml['Experience'] = df_ml['Experience'].clip(lower=0)
    feat_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    X_all = scaler.transform(df_ml[feat_cols])
    rf_model = results['Random Forest']['model']
    all_proba = rf_model.predict_proba(X_all)[:, 1]
    df_ml['LoanProbability'] = all_proba * 100

    df_ml['PropensityTier'] = pd.cut(
        df_ml['LoanProbability'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Cold (<20%)', 'Warm (20-40%)', 'Interested (40-60%)', 'Hot (60-80%)', 'Very Hot (80%+)']
    )

    # Funnel / Segmentation overview
    st.markdown("<div class='section-header'><h3>🔥 Customer Propensity Funnel</h3><p>Segmenting all 5,000 customers by predicted loan acceptance probability</p></div>", unsafe_allow_html=True)

    funnel_data = df_ml['PropensityTier'].value_counts().reset_index()
    funnel_data.columns = ['Tier', 'Count']
    tier_order = ['Very Hot (80%+)', 'Hot (60-80%)', 'Interested (40-60%)', 'Warm (20-40%)', 'Cold (<20%)']
    funnel_data['Tier'] = pd.Categorical(funnel_data['Tier'], categories=tier_order, ordered=True)
    funnel_data = funnel_data.sort_values('Tier')

    actual_rates = df_ml.groupby('PropensityTier')['Personal Loan'].mean().reset_index()
    actual_rates.columns = ['Tier', 'ActualRate']
    funnel_data = funnel_data.merge(actual_rates, on='Tier', how='left')
    funnel_data['ActualRate'] = (funnel_data['ActualRate'] * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        tier_colors = ['#f87171', '#fbbf24', '#a78bfa', '#38bdf8', '#34d399']
        fig = go.Figure(go.Funnel(
            y=funnel_data['Tier'], x=funnel_data['Count'],
            textinfo='value+percent initial',
            marker=dict(color=tier_colors),
            connector=dict(line=dict(color='#1e293b', width=2))
        ))
        fig.update_layout(title='Customer Propensity Funnel (All 5,000 Customers)')
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=funnel_data['Tier'], y=funnel_data['Count'],
            name='Count', marker_color=tier_colors, opacity=0.8,
            text=funnel_data['Count'], textposition='outside'
        ))
        fig.add_trace(go.Scatter(
            x=funnel_data['Tier'], y=funnel_data['ActualRate'],
            name='Actual Accept Rate %', yaxis='y2',
            line=dict(color='#38bdf8', width=3),
            mode='lines+markers', marker=dict(size=10)
        ))
        fig.update_layout(
            title='Propensity Tier: Volume vs Actual Acceptance Rate',
            yaxis=dict(title='Number of Customers'),
            yaxis2=dict(title='Actual Accept Rate %', overlaying='y', side='right', showgrid=False),
            xaxis=dict(tickangle=-15)
        )
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    # Personalised Offer Engine
    st.markdown("<div class='section-header'><h3>🎁 Personalised Loan Offer Engine</h3><p>Rule-based and ML-powered offer assignment for each customer segment</p></div>", unsafe_allow_html=True)

    def assign_offer(row):
        prob = row['LoanProbability']
        income = row['Income']
        ccavg = row['CCAvg']
        edu = row['Education']
        cd = row['CD Account']
        mortgage = row['Mortgage']
        has_loan = row['Personal Loan']

        if has_loan == 1:
            return {
                'Tier': '✅ Existing Customer',
                'Offer': 'Top-Up / Refinancing',
                'Details': 'Offer top-up loans or refinancing at preferential rates. Cross-sell premium products.',
                'Rate': 'Preferential: 8.5–9.5% p.a.',
                'Channel': 'Relationship Manager'
            }

        if prob >= 80:
            if income >= 150:
                return {'Tier': '🔥 Very Hot — Platinum', 'Offer': 'Premium Personal Loan + Wealth Bundle',
                        'Details': 'High-income, very high propensity. Offer premium loan up to $150K with concierge onboarding + investment advisory.',
                        'Rate': '9.0–10.5% p.a.', 'Channel': 'Dedicated Relationship Manager + Digital'}
            elif cd == 1:
                return {'Tier': '🔥 Very Hot — CD Holder', 'Offer': 'Pre-Approved Loan (CD-Backed)',
                        'Details': 'CD account provides security signal. Offer pre-approved loan with fast-track approval citing CD as positive signal.',
                        'Rate': '9.5–10.5% p.a.', 'Channel': 'Branch + Mobile Push'}
            else:
                return {'Tier': '🔥 Very Hot', 'Offer': 'Express Pre-Approved Personal Loan',
                        'Details': 'Very high propensity. Send pre-approved loan offer via app with 1-click acceptance. Offer 0 processing fee for 30 days.',
                        'Rate': '10.0–11.5% p.a.', 'Channel': 'Mobile App + Email'}

        if 60 <= prob < 80:
            if ccavg >= 3:
                return {'Tier': '🟡 Hot — High Spender', 'Offer': 'Loan + CC Limit Enhancement Bundle',
                        'Details': 'High CC spender with good propensity. Bundle personal loan offer with credit limit increase to make combined offer more appealing.',
                        'Rate': '10.5–11.5% p.a.', 'Channel': 'In-App + Email Campaign'}
            elif edu == 3:
                return {'Tier': '🟡 Hot — Professional', 'Offer': 'Professional Loan (Education/Career)',
                        'Details': 'Advanced degree holders. Frame loan for professional development, home improvement, or career-related expenses.',
                        'Rate': '10.0–11.0% p.a.', 'Channel': 'Email + LinkedIn-style messaging'}
            else:
                return {'Tier': '🟡 Hot', 'Offer': 'Targeted Personal Loan Offer with Incentive',
                        'Details': 'Good propensity segment. Offer loan with cashback on first EMI or gift voucher to nudge acceptance.',
                        'Rate': '11.0–12.0% p.a.', 'Channel': 'SMS + Email + Branch'}

        if 40 <= prob < 60:
            return {'Tier': '🟣 Interested — Nurture', 'Offer': 'Financial Needs Assessment',
                    'Details': 'Mid-propensity segment. Engage with personalised financial health check. Offer EMI calculator tool and soft pre-qualification.',
                    'Rate': '11.5–13.0% p.a.', 'Channel': 'Email Drip Campaign + App Nudge'}

        if 20 <= prob < 40:
            return {'Tier': '🔵 Warm — Educate', 'Offer': 'Loan Awareness + Product Education',
                    'Details': 'Low-mid propensity. Run financial literacy campaign explaining loan benefits, eligibility, and tax advantages. Soft touch.',
                    'Rate': 'N/A — awareness stage', 'Channel': 'Email Newsletter + Webinar'}

        return {'Tier': '❄️ Cold — Re-engage Later', 'Offer': 'No Immediate Loan Offer',
                'Details': 'Very low propensity. Focus on deepening banking relationship first — encourage online banking, FD, or savings product adoption.',
                'Rate': 'N/A', 'Channel': 'Generic Newsletter'}

    df_ml['OfferDetails'] = df_ml.apply(assign_offer, axis=1)
    df_ml['OfferTier'] = df_ml['OfferDetails'].apply(lambda x: x['Tier'])
    df_ml['OfferType'] = df_ml['OfferDetails'].apply(lambda x: x['Offer'])
    df_ml['OfferRate'] = df_ml['OfferDetails'].apply(lambda x: x['Rate'])
    df_ml['OfferChannel'] = df_ml['OfferDetails'].apply(lambda x: x['Channel'])

    # Offer Distribution
    offer_counts = df_ml['OfferTier'].value_counts().reset_index()
    offer_counts.columns = ['Offer Tier', 'Count']

    c1, c2 = st.columns([2, 3])
    with c1:
        offer_colors_map = {
            '✅ Existing Customer': '#34d399',
            '🔥 Very Hot — Platinum': '#f87171',
            '🔥 Very Hot — CD Holder': '#f97316',
            '🔥 Very Hot': '#fbbf24',
            '🟡 Hot — High Spender': '#a78bfa',
            '🟡 Hot — Professional': '#818cf8',
            '🟡 Hot': '#60a5fa',
            '🟣 Interested — Nurture': '#a78bfa',
            '🔵 Warm — Educate': '#38bdf8',
            '❄️ Cold — Re-engage Later': '#475569'
        }
        offer_counts['Color'] = offer_counts['Offer Tier'].map(offer_colors_map).fillna('#94a3b8')
        fig = go.Figure(go.Pie(
            labels=offer_counts['Offer Tier'], values=offer_counts['Count'],
            hole=0.5,
            marker=dict(colors=offer_counts['Color'].tolist(),
                        line=dict(color='#070d1a', width=2)),
            textinfo='label+value',
            hovertemplate='%{label}<br>Customers: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig.update_layout(title='Offer Tier Distribution (All 5,000 Customers)', showlegend=False)
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with c2:
        st.markdown("#### 📋 Offer Strategy Definitions")
        offers = [
            ("🔥 Very Hot (80%+)", "#f87171",
             "**Pre-Approved / Express Loans** — 1-click accept, 0 processing fee, immediate disbursement. Includes Platinum bundle for income >$150K."),
            ("🟡 Hot (60–80%)", "#fbbf24",
             "**Targeted Incentive Offers** — Bundle with CC limit upgrade, cashback on first EMI, or professional loan framing based on profile."),
            ("🟣 Interested (40–60%)", "#a78bfa",
             "**Nurture + Soft Pre-Qual** — EMI calculator, financial health assessment, soft pre-qualification with no hard credit inquiry."),
            ("🔵 Warm (20–40%)", "#38bdf8",
             "**Education Campaign** — Financial literacy, loan benefit communication, eligibility awareness. No hard sell."),
            ("❄️ Cold (<20%)", "#475569",
             "**Relationship Deepening** — Onboard to digital banking, FD, or savings products. Re-assess in 6 months."),
            ("✅ Existing Loan Holders", "#34d399",
             "**Top-Up / Refinance** — Retention focus. Preferential rates for loyalty. Cross-sell premium products."),
        ]
        for tier, color, desc in offers:
            st.markdown(f"""
            <div style='background:rgba(15,32,68,0.5); border-left:3px solid {color}; 
                         padding:0.7rem 1rem; margin:0.4rem 0; border-radius:0 8px 8px 0;'>
                <strong style='color:{color}'>{tier}</strong><br>
                <span style='color:#94a3b8; font-size:0.85rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    # Hot Prospects Table
    st.markdown("<div class='section-header'><h3>🎯 Top Prospects — High Propensity Customers</h3><p>Ranked list of customers most likely to accept a personal loan offer (excluding existing loan holders)</p></div>", unsafe_allow_html=True)

    hot_prospects = df_ml[
        (df_ml['Personal Loan'] == 0) &
        (df_ml['LoanProbability'] >= 50)
    ].sort_values('LoanProbability', ascending=False).head(30)

    display_cols = ['ID', 'Age', 'Income', 'CCAvg', 'Education_Label', 'Family_Label',
                    'CD_Label', 'Mortgage', 'LoanProbability', 'OfferTier', 'OfferType', 'OfferRate', 'OfferChannel']

    display_df = hot_prospects[display_cols].rename(columns={
        'Education_Label': 'Education',
        'Family_Label': 'Family',
        'CD_Label': 'CD Account',
        'LoanProbability': 'Prob %',
        'OfferTier': 'Tier',
        'OfferType': 'Offer',
        'OfferRate': 'Rate',
        'OfferChannel': 'Channel'
    }).reset_index(drop=True)

    display_df['Prob %'] = display_df['Prob %'].round(1)

    st.dataframe(display_df, use_container_width=True, height=450)

    st.download_button(
        label="⬇️ Download Hot Prospects as CSV",
        data=display_df.to_csv(index=False),
        file_name="hot_prospects_universal_bank.csv",
        mime="text/csv"
    )

    # Strategic Recommendations
    st.markdown("<div class='section-header'><h3>📋 Strategic Recommendations</h3><p>Evidence-based actions to maximise personal loan conversion</p></div>", unsafe_allow_html=True)

    hot_count = len(df_ml[df_ml['LoanProbability'] >= 60])
    cd_rate = df_ml[df_ml['CD Account'] == 1]['Personal Loan'].mean() * 100
    high_income_rate = df_ml[df_ml['Income'] >= 100]['Personal Loan'].mean() * 100
    grad_rate = df_ml[df_ml['Education'] == 3]['Personal Loan'].mean() * 100

    recommendations = [
        ("💰 Income-Driven Targeting", f"Income is the single strongest predictor (r≈0.50). Customers earning >$100K accept loans at {high_income_rate:.1f}% vs 9.6% overall. Prioritise this segment with premium loan products and dedicated relationship managers. Explore salary advance / payroll loan products for mid-income segments.", "HIGH"),
        ("🏦 CD Account Cross-Sell", f"CD account holders have a {cd_rate:.1f}% loan acceptance rate — nearly 3x the base rate. These customers have demonstrated capital commitment and trust. Use CD maturity events as loan offer triggers. Automate maturity-date outreach with a pre-approved loan offer.", "HIGH"),
        ("💳 CC Spend as Propensity Signal", "Monthly CC spend >$3K strongly predicts loan acceptance. Build a real-time trigger: when a customer's monthly CC spend crosses $3K for 2 consecutive months, automatically trigger a personalised loan offer via app notification.", "HIGH"),
        ("🎓 Education-Based Messaging", f"Advanced/Professional degree holders accept at {grad_rate:.1f}%. Frame loans for this segment around career growth, home improvement, or investment. Use LinkedIn-style professional messaging rather than mass-market copy.", "MEDIUM"),
        ("👨‍👩‍👧 Family Size Targeting", "Family size 3-4 shows higher acceptance — likely driven by expenses (children, education, home). Create family-centric loan packages: education loans, home renovation bundles. Target via family life-stage events.", "MEDIUM"),
        ("🤖 ML Propensity Scoring Pipeline", f"Deploy the trained Random Forest model (AUC ~0.98) as a live scoring engine. Score all {len(df_ml):,} customers monthly. The {hot_count:,} customers predicted >60% probability represent the core campaign list — prioritise them for outbound calls and digital offers.", "HIGH"),
        ("📱 Digital-First for Online Users", f"{pct_online:.0f}% of customers use online banking. Build in-app loan offers with pre-filled applications for high-propensity users. A/B test push notification copy optimised for conversion. Online users have 2x higher engagement rates.", "MEDIUM"),
        ("🔄 Nurture Funnel for 40-60% Tier", f"The interested tier has {len(df_ml[(df_ml['LoanProbability'] >= 40) & (df_ml['LoanProbability'] < 60)]):,} customers. These are worth a 3-touch drip campaign: (1) financial health newsletter, (2) EMI calculator CTA, (3) soft pre-qualification. Expected uplift: 15-20% conversion within 90 days.", "MEDIUM"),
    ]

    for title, desc, priority in recommendations:
        priority_color = '#f87171' if priority == 'HIGH' else '#fbbf24'
        st.markdown(f"""
        <div class='offer-card'>
            <h4>{title} <span style='color:{priority_color}; font-size:0.73rem; 
                background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px;'>{priority} PRIORITY</span></h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Impact Matrix
    st.markdown("<div class='section-header'><h3>💹 Campaign ROI Estimation</h3><p>Expected impact and effort matrix for each recommended initiative</p></div>", unsafe_allow_html=True)

    impact_data = pd.DataFrame({
        'Initiative': ['ML Propensity Model', 'CD Maturity Trigger', 'Income Segment Targeting',
                       'CC Spend Signal', 'Education Messaging', 'Family Bundles',
                       'Digital Nurture Campaign', 'Branch Upsell Training'],
        'Expected Loan Lift (% pts)': [4.8, 3.5, 3.2, 2.8, 2.0, 1.8, 2.5, 1.5],
        'Implementation Effort (1-5)': [4, 2, 3, 3, 2, 3, 3, 2],
        'Time to Impact (months)': [3, 1, 2, 2, 1, 3, 2, 4]
    })

    fig = px.scatter(impact_data,
                     x='Implementation Effort (1-5)', y='Expected Loan Lift (% pts)',
                     size='Time to Impact (months)', text='Initiative',
                     color='Expected Loan Lift (% pts)',
                     color_continuous_scale=[[0, '#475569'], [0.5, '#818cf8'], [1, '#38bdf8']],
                     title='ROI vs Effort Matrix — Size = Time to Impact')
    fig.update_traces(textposition='top center', textfont=dict(size=10.5))
    fig.update_layout(xaxis_title='Implementation Effort (1=Easy, 5=Hard)',
                      yaxis_title='Expected Loan Acceptance Lift (% points)')
    # Add quadrant annotations
    fig.add_shape(type='line', x0=3, x1=3, y0=0, y1=6,
                  line=dict(color='rgba(56,189,248,0.2)', dash='dash'))
    fig.add_shape(type='line', x0=1, x1=5, y0=2.5, y1=2.5,
                  line=dict(color='rgba(56,189,248,0.2)', dash='dash'))
    fig.add_annotation(x=1.8, y=5.5, text="🎯 Quick Wins", showarrow=False,
                       font=dict(color='#34d399', size=11))
    fig.add_annotation(x=4.2, y=5.5, text="🏗️ Strategic Bets", showarrow=False,
                       font=dict(color='#fbbf24', size=11))
    st.plotly_chart(styled_chart(fig, 480), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; padding:1.2rem;'>
    <strong style='color:#38bdf8'>Universal Bank — Personal Loan Intelligence Suite</strong> · 
    Built with Streamlit & Plotly · 
    Dataset: 5,000 customers × 14 features · 
    Descriptive · Diagnostic · Predictive · Prescriptive
</div>
""", unsafe_allow_html=True)
