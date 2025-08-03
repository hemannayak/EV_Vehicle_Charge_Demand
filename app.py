import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64

# --- MODERN GRADIENT + SIDEBAR + BANNER ---
st.set_page_config(page_title="EV Forecast", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(120deg, #E0EAFC 0%, #CFDEF3 100%, #c2d3f2 60%, #7f848a 100%);
        }
        .reportview-container .markdown-text-container {
            font-family: 'Segoe UI', sans-serif;
        }
        .metric-box {
            background: #f5fafd;
            border-radius: 10px;
            border: 1px solid #d1e3fc;
            padding: 12px 20px;
            margin-bottom: 15px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        }
        .metric-label {
            font-size: 20px;
            font-weight: 600;
            color: #48639e;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #19325b;
        }
        .help-box {
            font-size: 16px;
            background: #e6f3ff;
            border-radius: 12px;
            border-left: 5px solid #448aff;
            padding: 16px;
            margin-bottom: 24px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("ev-car-factory.jpg", use_container_width=True)
    st.markdown("<div class='help-box'><b>How to use:</b><br/>"
        "• <b>Select</b> one or more counties (or the whole state) and set your forecast years. <br/>"
        "• View <b>EV adoption trends, growth %</b>, and compare counties.<br/>"
        "• <b>Download</b> results as CSV.<br/>"
        "• See dynamic metrics instantly updated.<br/>"
        "• <b>Banner</b>: EV car factory for visual context.<br/>"
        "<br/><b>Data:</b> Results shown are automated forecasts. Interpret cautiously for decision making.<br/>"
        "<br/><b>Need Help?</b> See feature explanations below.<br/>"
        "</div>",
        unsafe_allow_html=True)

    st.markdown("<b>Feature Explanations:</b>", unsafe_allow_html=True)
    st.markdown("""
    • <b>County/State selection</b>: Model automatically generates forecasts for your chosen area(s).<br/>
    • <b>Forecast duration</b>: Choose 1–5 years.<br/>
    • <b>Growth metrics</b>: Instantly see historic and projected EV totals, plus % growth.<br/>
    • <b>Dynamic Visualization</b>: Plots update on every selection change.<br/>
    • <b>Export</b>: Get your forecast as a downloadable CSV.<br/>
    """, unsafe_allow_html=True)

st.markdown("")  # Spacer for layout

# Prominent banner at very top
st.markdown("""
    <div style="width:100%; border-radius: 14px; margin-bottom: 18px; overflow: hidden;">
        <img src="https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80"
            style="display:block; width:100%; height:200px; object-fit:cover; border-radius:14px;">
    </div>
""", unsafe_allow_html=True)

# === STYLISH TITLE & SUBTITLE ===
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

# Local image in layout
st.image("ev-car-factory.jpg", use_container_width=True)

st.markdown("""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: #FFFFFF;'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
""", unsafe_allow_html=True)

# === LOAD MODEL ===
model = joblib.load('forecasting_ev_model.pkl')

# === LOAD DATA FUNCTION ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ==== New: Multi-county selection & custom horizon ====
st.markdown("## 🔍 Forecast Settings", unsafe_allow_html=True)
st.write("Choose one or more counties and forecast duration (years).")

counties = sorted(df['County'].dropna().unique().tolist())
selected_counties = st.multiselect("📍 Choose county/counties", counties, max_selections=5, default=[counties[0]])
years = st.slider("For how many years to forecast ahead?", min_value=1, max_value=5, value=3)
forecast_horizon = years * 12  # Months

if not selected_counties:
    st.warning("Please select at least one county to continue.")
    st.stop()

metrics_placeholder = st.empty()
chart_placeholder = st.empty()
download_placeholder = st.empty()

all_forecast_data = []

# === FORECAST FOR EACH COUNTY SELECTED ===
for county in selected_counties:
    cty_df = df[df['County'] == county].sort_values("Date")
    if len(cty_df) < 6:
        st.warning(f"Not enough data for {county}. Skipping.")
        continue
    cty_code = cty_df['county_encoded'].iloc[0]
    hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
    cum_ev = list(np.cumsum(hist_ev))
    months_since = cty_df['months_since_start'].max()
    last_date = cty_df['Date'].max()

    future_rows_cty = []
    for i in range(1, forecast_horizon + 1):
        forecast_date = last_date + pd.DateOffset(months=i)
        months_since += 1
        lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cum = cum_ev[-6:]
        ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

        new_row = {
            'months_since_start': months_since,
            'county_encoded': cty_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_slope
        }
        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

        hist_ev.append(pred)
        if len(hist_ev) > 6:
            hist_ev.pop(0)
        cum_ev.append(cum_ev[-1] + pred)
        if len(cum_ev) > 6:
            cum_ev.pop(0)

    hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
    hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

    fc_df = pd.DataFrame(future_rows_cty)
    fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
    combined_cty = pd.concat([hist_cum[['Date', 'Cumulative EV']], fc_df[['Date', 'Cumulative EV']]], ignore_index=True)
    combined_cty['County'] = county
    combined_cty['Type'] = ["Historical"] * hist_cum.shape[0] + ["Forecast"] * fc_df.shape[0]
    all_forecast_data.append(combined_cty)

# --- METRICS: Show for each selected county ---
if all_forecast_data:
    metrics_cols = st.columns(len(all_forecast_data))
    for idx, (cty, col) in enumerate(zip(selected_counties, metrics_cols)):
        cty_forecast = all_forecast_data[idx]
        last_hist_idx = cty_forecast['Type'].eq("Historical").sum() - 1
        curr_total = int(cty_forecast['Cumulative EV'].iloc[last_hist_idx])
        forecast_total = int(cty_forecast['Cumulative EV'].iloc[-1])
        absolute_growth = forecast_total - curr_total
        growth_pct = ((forecast_total - curr_total) / curr_total * 100) if curr_total > 0 else float("nan")
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-label'>{cty} County</div>"
                        f"<div class='metric-value'>Current Total: {curr_total:,}</div>"
                        f"<div style='font-size:20px; color:#248f36;'>Forecast: {forecast_total:,}</div>"
                        f"<div style='font-size:18px; color:#3087ca;'>Growth: {growth_pct:.2f}%</div></div>", unsafe_allow_html=True)

# --- DYNAMIC FORECAST GRAPH ---
if all_forecast_data:
    fig, ax = plt.subplots(figsize=(13, 5))
    for cty_predict in all_forecast_data:
        ax.plot(cty_predict['Date'], cty_predict['Cumulative EV'], marker='o', label=cty_predict['County'].iloc[0])
    ax.set_title(f"Cumulative EV Adoption Forecast ({years} years)", fontsize=16, color='#19325b')
    ax.set_xlabel("Date", color='#48639e')
    ax.set_ylabel("Cumulative EV Count", color='#19325b')
    ax.grid(True, alpha=0.25)
    fig.patch.set_facecolor('#f5fafd')
    ax.legend()
    chart_placeholder.pyplot(fig)

# --- DOWNLOAD BUTTON FOR FORECAST CSV ---
if all_forecast_data:
    full_csv = pd.concat(all_forecast_data, ignore_index=True)
    csv_bytes = full_csv.to_csv(index=False).encode()
    download_placeholder.download_button(
        label="Download Forecast CSV",
        data=csv_bytes,
        file_name="ev_forecast.csv",
        mime="text/csv",
        help="Download the combined forecast data for all selected counties."
    )

# --- Original SINGLE FORECAST section (optional: keep for reference) ---
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties (legacy method)")

multi_counties = st.multiselect("Select up to 3 counties to compare", counties, max_selections=3)

if multi_counties:
    comparison_data = []
    forecast_horizon_legacy = 36  # Still 3 years for reference unless tied to slider

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        if len(cty_df) < 6:
            st.warning(f"Not enough data for {cty}. Skipping.")
            continue
        cty_code = cty_df['county_encoded'].iloc[0]
        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon_legacy + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)
            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    # Combine all data for plotting
    if comparison_data:
        comp_df = pd.concat(comparison_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in comp_df.groupby('County'):
            ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cumulative EV Count", color='white')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor('#1c1c1c')
        ax.tick_params(colors='white')
        ax.legend(title="County")
        st.pyplot(fig)
        
        # Display % growth
        growth_summaries = []
        for cty in multi_counties:
            cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
            historical_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon_legacy - 1]
            forecasted_total = cty_df['Cumulative EV'].iloc[-1]
            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
            else:
                growth_summaries.append(f"{cty}: N/A (no historical data)")
        growth_sentence = " | ".join(growth_summaries)
        st.success(f"Forecasted EV adoption growth over next 3 years — {growth_sentence}")

st.success("Forecast complete")
st.markdown("Prepared for the **AICTE Internship Cycle 2 by Hemanth Nayak**")
