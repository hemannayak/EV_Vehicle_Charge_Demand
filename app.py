import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(page_title="🔋 EV Demand Forecast App", layout="wide")

# === Load Model & Data ===
@st.cache_data
def load_model():
    return joblib.load("forecasting_ev_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

model = load_model()
df = load_data()

# === Sidebar Navigation ===
st.sidebar.title("🔧 Settings")
selected_view = st.sidebar.radio("Choose view:", ["📈 Single County Forecast", "📊 Compare Counties", "📥 Download Data"])
forecast_years = st.sidebar.slider("Forecast Duration (Years)", 1, 5, 3)
forecast_horizon = forecast_years * 12

# === App Title ===
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🔮 EV Adoption Forecasting Tool</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AICTE Internship – Cycle 2 by S4F</h4>", unsafe_allow_html=True)

# === Shared Preprocessing Logic ===
def generate_forecast(county, horizon):
    county_df = df[df['County'] == county].sort_values("Date")
    code = county_df['county_encoded'].iloc[0]
    latest_date = county_df['Date'].max()
    months_since = county_df['months_since_start'].max()
    hist = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cum = list(np.cumsum(hist))

    forecast = []
    for i in range(1, horizon + 1):
        months_since += 1
        date = latest_date + pd.DateOffset(months=i)
        lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_1 = (lag1 - lag2) / lag2 if lag2 else 0
        pct_3 = (lag1 - lag3) / lag3 if lag3 else 0
        slope = np.polyfit(range(len(cum)), cum, 1)[0] if len(cum) == 6 else 0

        row = pd.DataFrame([{
            'months_since_start': months_since,
            'county_encoded': code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_1,
            'ev_total_pct_change_3': pct_3,
            'ev_growth_slope': slope
        }])

        pred = model.predict(row)[0]
        forecast.append({"Date": date, "Predicted EV Total": round(pred)})

        hist.append(pred)
        hist = hist[-6:]
        cum.append(cum[-1] + pred)
        cum = cum[-6:]

    return pd.DataFrame(forecast)

# === View: Single County Forecast ===
if selected_view == "📈 Single County Forecast":
    counties = sorted(df['County'].dropna().unique().tolist())
    selected_county = st.selectbox("Choose County", counties)

    forecast_df = generate_forecast(selected_county, forecast_horizon)
    historical = df[df['County'] == selected_county][['Date', 'Electric Vehicle (EV) Total']].copy()
    historical['Cumulative EV'] = historical['Electric Vehicle (EV) Total'].cumsum()
    forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical['Cumulative EV'].iloc[-1]

    combined = pd.concat([
        historical[['Date', 'Cumulative EV']].assign(Source="Historical"),
        forecast_df[['Date', 'Cumulative EV']].assign(Source="Forecast")
    ])

    # Plot
    st.subheader(f"📊 Cumulative EV Forecast – {selected_county} ({forecast_years} years)")
    chart = alt.Chart(combined).mark_line(point=True).encode(
        x='Date:T',
        y='Cumulative EV:Q',
        color='Source:N',
        tooltip=['Date:T', 'Cumulative EV:Q', 'Source:N']
    ).properties(width=900, height=400)
    st.altair_chart(chart, use_container_width=True)

    # Metrics
    with st.container():
        col1, col2 = st.columns(2)
        start = historical['Cumulative EV'].iloc[-1]
        end = forecast_df['Cumulative EV'].iloc[-1]
        pct_change = ((end - start) / start) * 100 if start > 0 else 0
        col1.metric("📦 Current EV Count", f"{int(start)}")
        col2.metric("🚀 Forecasted Growth", f"{pct_change:.2f}%")

# === View: Compare Counties ===
elif selected_view == "📊 Compare Counties":
    counties = st.multiselect("Select up to 3 counties", sorted(df['County'].unique()), max_selections=3)

    if counties:
        all_dfs = []
        for cty in counties:
            fc = generate_forecast(cty, forecast_horizon)
            hist = df[df['County'] == cty][['Date', 'Electric Vehicle (EV) Total']].copy()
            hist['Cumulative EV'] = hist['Electric Vehicle (EV) Total'].cumsum()
            fc['Cumulative EV'] = fc['Predicted EV Total'].cumsum() + hist['Cumulative EV'].iloc[-1]
            combined = pd.concat([
                hist[['Date', 'Cumulative EV']],
                fc[['Date', 'Cumulative EV']]
            ])
            combined['County'] = cty
            all_dfs.append(combined)

        final_df = pd.concat(all_dfs)

        st.subheader(f"📊 EV Trend Comparison – {forecast_years} Year Forecast")
        chart = alt.Chart(final_df).mark_line(point=True).encode(
            x='Date:T',
            y='Cumulative EV:Q',
            color='County:N',
            tooltip=['Date:T', 'Cumulative EV:Q', 'County:N']
        ).properties(width=900, height=450)
        st.altair_chart(chart, use_container_width=True)

# === View: Data Download ===
elif selected_view == "📥 Download Data":
    st.subheader("📤 Download Processed Data")

    st.write("You can download:")
    col1, col2 = st.columns(2)

    # Original
    csv = df.to_csv(index=False).encode('utf-8')
    col1.download_button("⬇️ Download Raw Dataset", data=csv, file_name="preprocessed_ev_data.csv", mime="text/csv")

    # Sample forecast
    sample_county = df['County'].unique()[0]
    sample_fc = generate_forecast(sample_county, forecast_horizon)
    forecast_csv = sample_fc.to_csv(index=False).encode('utf-8')
    col2.download_button("⬇️ Download Sample Forecast", data=forecast_csv, file_name="sample_forecast.csv", mime="text/csv")

# === Footer ===
st.markdown("---")
st.markdown("⚡ *Project developed as part of AICTE Internship Cycle 2 by Team S4F*")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: #fcf7f7;
            color: #000000;
        }
        .stApp {
            background: linear-gradient(to right, #c2d3f2, #7f848a);
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("🔧 Settings")
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 3)
forecast_horizon = forecast_years * 12

st.sidebar.info("Select forecast duration and counties to view future EV adoption trends.")

# === Page Header ===
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

st.image("ev-car-factory.jpg", use_container_width=True)

st.markdown("""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: #FFFFFF;'>
        Select a county and see the forecasted EV adoption trend for the next few years.
    </div>
""", unsafe_allow_html=True)

# === Load and cache data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === Forecast generator ===
def forecast_ev(county, horizon):
    county_df = df[df['County'] == county].sort_values("Date")
    code = county_df['county_encoded'].iloc[0]
    months = county_df['months_since_start'].max()
    last_date = county_df['Date'].max()

    hist = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cum_hist = list(np.cumsum(hist))

    future = []
    for i in range(1, horizon + 1):
        months += 1
        f_date = last_date + pd.DateOffset(months=i)
        lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]
        roll = np.mean([lag1, lag2, lag3])
        pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        slope = np.polyfit(range(len(cum_hist)), cum_hist, 1)[0] if len(cum_hist) == 6 else 0

        row = {
            'months_since_start': months,
            'county_encoded': code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll,
            'ev_total_pct_change_1': pct1,
            'ev_total_pct_change_3': pct3,
            'ev_growth_slope': slope
        }

        pred = model.predict(pd.DataFrame([row]))[0]
        future.append({"Date": f_date, "Predicted EV Total": round(pred)})
        hist.append(pred)
        hist = hist[-6:]
        cum_hist.append(cum_hist[-1] + pred)
        cum_hist = cum_hist[-6:]

    return pd.DataFrame(future)

# === County selection ===
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
forecast_df = forecast_ev(county, forecast_horizon)

# === Cumulative EV Calculation ===
hist_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
hist_cum['Source'] = 'Historical'
hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    hist_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot Graph ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")

fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} ({forecast_years} Years Forecast)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# === Growth Summary ===
hist_total = hist_cum['Cumulative EV'].iloc[-1]
forecast_total = forecast_df['Cumulative EV'].iloc[-1]

if hist_total > 0:
    growth = ((forecast_total - hist_total) / hist_total) * 100
    trend = "increase 📈" if growth > 0 else "decrease 📉"
    st.success(f"Based on the forecast, EV adoption in **{county}** is expected to **{trend} by {growth:.2f}%** over {forecast_years} years.")
else:
    st.warning("Historical EV total is zero; forecast comparison is not possible.")

# === Comparison Section ===
st.markdown("---")
st.header("🔁 Compare EV Adoption in Multiple Counties")

multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)

if multi_counties:
    comp_data = []

    for cty in multi_counties:
        cty_hist = df[df['County'] == cty].sort_values("Date")
        cty_forecast = forecast_ev(cty, forecast_horizon)

        hist = cty_hist[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist['Cumulative EV'] = hist['Electric Vehicle (EV) Total'].cumsum()

        cty_forecast['Cumulative EV'] = cty_forecast['Predicted EV Total'].cumsum() + hist['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist[['Date', 'Cumulative EV']],
            cty_forecast[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comp_data.append(combined_cty)

    comp_df = pd.concat(comp_data)

    st.subheader("📈 Multi-County EV Forecast Comparison")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title(f"EV Adoption Trends – {forecast_years}-Year Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County")
    st.pyplot(fig)

    # Growth Comparison
    st.success("📊 Forecasted Growth Percentages:")
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        base = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        final = cty_df['Cumulative EV'].iloc[-1]
        if base > 0:
            pct = ((final - base) / base) * 100
            st.markdown(f"- **{cty}**: {pct:.2f}% growth")

# === Download CSV ===
st.markdown("---")
st.download_button("⬇️ Download Forecast Data", data=forecast_df.to_csv(index=False).encode('utf-8'),
                   file_name=f"{county}_ev_forecast_{forecast_years}years.csv", mime='text/csv')

st.markdown("Prepared for the **AICTE Internship Cycle 2 by Hemanth Nayak**")
