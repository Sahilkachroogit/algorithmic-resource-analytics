"""
app.py — Algorithmic Resource & Telemetry Analytics
AICTE | IBM SkillsBuild Data Analytics with AI Internship
Student: Sahil Rashid Kachroo
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Cloud Compute Telemetry Analytics',
    layout='wide',
    page_icon='🖥️',
)

# ── CSS override for dark card styling ───────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: #12122A;
        border: 1px solid #2E2E4A;
        border-radius: 10px;
        padding: 16px;
    }
    .risk-box {
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'telemetry_raw.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={'algorithm':'category',
                                           'input_size':'int64',
                                           'execution_time_us':'float64',
                                           'memory_bytes':'float64'})
    else:
        # Inline synthetic fallback
        np.random.seed(42)
        N = np.unique(np.logspace(1, np.log10(5_000_000), 50).astype(int))
        rows = []
        for n in N:
            rows.append({'algorithm':'binary_exponentiation','input_size':int(n),
                         'execution_time_us': 0.0003*n*np.log2(n+1),
                         'memory_bytes': 64.0})
            rows.append({'algorithm':'vector_reallocation','input_size':int(n),
                         'execution_time_us': 0.002*n,
                         'memory_bytes': float(sum(2**k*4 for k in range(int(np.log2(n+1))+1)))})
            rows.append({'algorithm':'linked_list_traversal','input_size':int(n),
                         'execution_time_us': 0.018*n,
                         'memory_bytes': n*16.0})
        df = pd.DataFrame(rows)
        df['algorithm'] = df['algorithm'].astype('category')
    df['log_input_size'] = np.log1p(df['input_size'])
    return df

@st.cache_resource
def load_models(df):
    base = os.path.dirname(os.path.abspath(__file__))
    t_path = os.path.join(base, 'model_time.pkl')
    m_path = os.path.join(base, 'model_mem.pkl')
    le = LabelEncoder()
    le.fit(df['algorithm'].cat.categories)
    if os.path.exists(t_path) and os.path.exists(m_path):
        return joblib.load(t_path), joblib.load(m_path), le
    # Train inline
    X = np.column_stack([df['log_input_size'].values,
                         le.transform(df['algorithm'])])
    rf_t = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf_m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf_t.fit(X, df['execution_time_us'].values)
    rf_m.fit(X, df['memory_bytes'].values)
    return rf_t, rf_m, le

df = load_data()
model_time, model_mem, le = load_models(df)

ALGOS   = list(df['algorithm'].cat.categories)
PALETTE = {'binary_exponentiation':'#00C8FF',
            'vector_reallocation':  '#FF6B6B',
            'linked_list_traversal':'#A8FF78'}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg', width=80)
    st.title('⚙️ Control Panel')
    st.markdown('---')

    selected_algo = st.selectbox('Algorithm Type', ALGOS,
                                  index=ALGOS.index('linked_list_traversal'))
    input_n = st.number_input('Input Size N', min_value=10, max_value=5_000_000,
                               value=100_000, step=10_000)
    core_count   = st.slider('VM Core Count', 1, 128, 8)
    hourly_cost  = st.number_input('Hourly Server Cost ($)', 0.10, 50.0, 2.50, step=0.10)
    st.markdown('---')
    st.caption('AICTE | IBM SkillsBuild\nData Analytics with AI Internship')
    st.caption('Student: Sahil Rashid Kachroo')

# ── Predictions ───────────────────────────────────────────────────────────────
algo_enc  = le.transform([selected_algo])[0]
X_pred    = np.array([[np.log1p(input_n), algo_enc]])

pred_time_us  = float(model_time.predict(X_pred)[0])
pred_mem_b    = float(model_mem.predict(X_pred)[0])
pred_time_ms  = pred_time_us / 1000.0
pred_mem_mb   = pred_mem_b   / (1024**2)

# Cloud cost: time_s * cores * (hourly/3600)
pred_time_s   = pred_time_us / 1e6
cloud_cost    = pred_time_s * core_count * (hourly_cost / 3600)

# ── Header ────────────────────────────────────────────────────────────────────
st.title('🖥️ Cloud Compute Telemetry Analytics')
st.markdown('**AICTE | IBM SkillsBuild Data Analytics with AI Internship** — Sahil Rashid Kachroo')
st.markdown('---')

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric('⏱️ Predicted Runtime', f'{pred_time_ms:,.3f} ms',
           help='Random Forest prediction for selected N and algorithm')
c2.metric('💾 Peak Memory', f'{pred_mem_mb:,.4f} MB',
           help='Predicted peak memory allocation')
c3.metric('💰 Cloud Infra Cost', f'${cloud_cost:.6f}',
           help='Estimated cloud cost for one execution')
c4.metric('🖧 VM Cores', f'{core_count}',
           help='Selected VM core count')

# ── Anomaly / Risk Indicator ──────────────────────────────────────────────────
if pred_mem_mb < 10:
    risk_color, risk_label = '#28a745', '✅ NORMAL — Memory usage within safe bounds'
elif pred_mem_mb < 100:
    risk_color, risk_label = '#fd7e14', '⚠️ ELEVATED — Memory pressure increasing; consider scaling'
else:
    risk_color, risk_label = '#dc3545', '🚨 CRITICAL — Memory fault risk high; optimise algorithm or increase VM'

st.markdown(
    f'<div class="risk-box" style="background:{risk_color}22;border:2px solid {risk_color};color:{risk_color}">'
    f'{risk_label}</div>',
    unsafe_allow_html=True
)

st.markdown('---')

# ── Charts ────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

# Chart 1: Actual vs Predicted Execution Time
with col_l:
    st.subheader('① Actual vs. Predicted Execution Time')
    sub = df[df['algorithm'] == selected_algo].sort_values('input_size')
    X_sub = np.column_stack([sub['log_input_size'].values,
                              np.full(len(sub), algo_enc)])
    pred_sub = model_time.predict(X_sub)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sub['input_size'], y=sub['execution_time_us'],
                               name='Actual', mode='lines',
                               line=dict(color=PALETTE[selected_algo], width=2)))
    fig1.add_trace(go.Scatter(x=sub['input_size'], y=pred_sub,
                               name='Predicted', mode='lines',
                               line=dict(color='white', width=1.5, dash='dash')))
    fig1.add_vline(x=input_n, line_dash='dot', line_color='yellow',
                   annotation_text=f'N={input_n:,}', annotation_position='top right')
    fig1.update_layout(xaxis_type='log', yaxis_type='log',
                        xaxis_title='Input Size N', yaxis_title='Execution Time (µs)',
                        template='plotly_dark', height=350, margin=dict(t=30))
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Memory Allocation Growth Curves
with col_r:
    st.subheader('② Memory Allocation Growth Curves')
    fig2 = go.Figure()
    for algo in ALGOS:
        grp = df[df['algorithm'] == algo].sort_values('input_size')
        fig2.add_trace(go.Scatter(
            x=grp['input_size'], y=grp['memory_bytes']/1024,
            name=algo, mode='lines', fill='tozeroy',
            line=dict(color=PALETTE[algo], width=2),
            fillcolor=PALETTE[algo].replace('#', 'rgba(').replace('FF', 'FF,').rstrip(',') + ',0.15)'
            if False else PALETTE[algo]
        ))
    fig2.update_layout(xaxis_type='log', xaxis_title='Input Size N',
                        yaxis_title='Memory (KB)', template='plotly_dark',
                        height=350, margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

col_l2, col_r2 = st.columns(2)

# Chart 3: Residual Error Distribution
with col_l2:
    st.subheader('③ Residual Error Distribution')
    X_all  = np.column_stack([df['log_input_size'].values,
                               le.transform(df['algorithm'])])
    pred_all_t = model_time.predict(X_all)
    residuals  = df['execution_time_us'].values - pred_all_t
    fig3 = px.histogram(x=residuals, nbins=40, template='plotly_dark',
                         color_discrete_sequence=['#00C8FF'],
                         labels={'x': 'Residual (µs)', 'y': 'Count'})
    fig3.add_vline(x=0, line_dash='dash', line_color='red', line_width=1.5)
    fig3.update_layout(height=350, margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4: Cloud Cost Scaling Threshold
with col_r2:
    st.subheader('④ Cloud Cost Risk Threshold')
    COST_PER_US = 1e-9
    RISK_MULT   = 10
    fig4 = go.Figure()
    for algo in ALGOS:
        grp  = df[df['algorithm'] == algo].sort_values('input_size')
        cost = grp['execution_time_us'] * COST_PER_US * core_count * (hourly_cost / 3600)
        baseline  = cost.iloc[0] if cost.iloc[0] > 0 else cost[cost > 0].iloc[0]
        risk_mask = cost >= baseline * RISK_MULT
        fig4.add_trace(go.Scatter(x=grp['input_size'], y=cost,
                                   name=algo, mode='lines',
                                   line=dict(color=PALETTE[algo], width=2)))
        if risk_mask.any():
            risk_n = grp['input_size'][risk_mask].min()
            fig4.add_vline(x=risk_n, line_dash='dot', line_color=PALETTE[algo])
    fig4.update_layout(xaxis_type='log', xaxis_title='Input Size N',
                        yaxis_title='Cost ($)', template='plotly_dark',
                        height=350, margin=dict(t=30))
    st.plotly_chart(fig4, use_container_width=True)
