import asyncio
import logging
import os
import sys
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the project root to the python path so imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.config import load_config
from src.app.context import RunContext

# We must import these inside an event loop or wrapper functions otherwise imports could crash Streamlit 
# if they contain asyncio.run inside module level
from src.app.main import run_backtest, run_live_cycle

# ==========================================
# UI Setup
# ==========================================
st.set_page_config(
    page_title="ACO Portfolio Optimizer",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐜 Ant Colony Portfolio Optimizer")
st.markdown(
    "Use **Swarm Intelligence** to construct robust stock portfolios based on technical, fundamental, and momentum factors."
)

# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.header("⚙️ Configuration")

run_mode = st.sidebar.radio(
    "Run Mode",
    options=["Live Optimization", "Cached Optimization", "Walk-Forward Backtest"],
    help="Live Optimization fetches fresh data. Cached Optimization uses previously downloaded files. Backtest simulates past performance.",
)

st.sidebar.subheader("Universe")
universe_mode = st.sidebar.selectbox(
    "Asset Search Scope",
    ["Custom Ticker List", "S&P 500 (Top 50)", "S&P 500 (Top 100)"]
)

if universe_mode == "Custom Ticker List":
    tickers_input = st.sidebar.text_area(
        "Tickers (comma separated)",
        value="AAPL,MSFT,GOOGL,AMZN,META,TSLA,NFLX,NVDA,JPM,V,UNH,JNJ,PG,HD,BAC",
        help="Leave blank to use the S&P 500 top 50 fallback.",
    )
else:
    st.sidebar.info(f"The system will automatically load the {universe_mode} dataset.")
    tickers_input = ""

st.sidebar.subheader("ACO Parameters")
num_ants = st.sidebar.slider("Number of Ants", min_value=10, max_value=200, value=30, step=10)
num_iterations = st.sidebar.slider("Iterations", min_value=10, max_value=500, value=100, step=10)
max_weight = st.sidebar.slider("Max Weight per Asset (%)", min_value=5, max_value=100, value=20, step=1)

if run_mode == "Walk-Forward Backtest":
    st.sidebar.subheader("Backtest Settings")
    start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=date(2023, 12, 31))

run_button = st.sidebar.button("🚀 Run System", use_container_width=True, type="primary")


# ==========================================
# Helper Functions
# ==========================================
def setup_context(universe_mode: str) -> RunContext:
    """Load config and set up runtime context based on UI inputs."""
    config_path = os.path.join(project_root, "config", "settings.example.yaml")
    config = load_config(config_path)
    
    # Apply UI Overrides
    config.aco.num_ants = num_ants
    config.aco.num_iterations = num_iterations
    config.constraints.max_weight_per_ticker = max_weight / 100.0
    
    if universe_mode == "S&P 500 (Top 50)":
        config.universe_size = 50
    elif universe_mode == "S&P 500 (Top 100)":
        config.universe_size = 100
        
    if run_mode == "Walk-Forward Backtest":
        mode_str = "backtest"
    elif run_mode == "Cached Optimization":
        mode_str = "cached"
    else:
        mode_str = "live"
        
    # Make a dummy logger
    logger = logging.getLogger("streamlit_logger")
    logger.setLevel(logging.INFO)
    
    return RunContext(config, logger, mode=mode_str, session=None, rate_limiter=None, cache_manager=None, start_date=None, end_date=None, random_seed=None, run_id=None)

async def _run_optimization_async(ctx: RunContext, universe: list[str]):
    return await run_live_cycle(ctx, universe)

def run_optimization_ui(universe: list[str] | None, universe_mode: str):
    """Execute live optimization and render results."""
    st.info("Initializing Operator and Agents...")
    ctx = setup_context(universe_mode)
    
    with st.spinner("Fetching data and optimizing swarm (this takes 5-15s)..."):
        operator_result, portfolio = asyncio.run(_run_optimization_async(ctx, universe))
        
    if portfolio is None or portfolio.num_holdings == 0:
        st.error("Optimization failed to find a valid portfolio.")
        return
        
    st.success(f"Optimization Complete! Found optimal allocation across {portfolio.num_holdings} assets.")
    
    # --- Row 1: Key Metrics ---
    cols = st.columns(4)
    cols[0].metric("Fitness Score", f"{portfolio.fitness:.4f}")
    cols[1].metric("Total Holdings", f"{portfolio.num_holdings}")
    
    if 'annualized_return' in portfolio.diagnostics:
        cols[2].metric("Expected Annual Return", f"{portfolio.diagnostics['annualized_return']:.2%}")
    if 'annualized_volatility' in portfolio.diagnostics:
        cols[3].metric("Expected Volatility", f"{portfolio.diagnostics['annualized_volatility']:.2%}")
        
    # --- Row 2: Charts and Tables ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Allocations")
        weights_df = pd.DataFrame(
            list(portfolio.weights.items()), columns=["Ticker", "Weight"]
        ).sort_values("Weight", ascending=False)
        
        # Plotly Donut Chart
        fig = px.pie(
            weights_df, 
            values='Weight', 
            names='Ticker', 
            hole=0.4,
            title="Portfolio Weights"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Weight Matrix")
        st.dataframe(
            weights_df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True
        )
        
        if 'top_pheromone' in portfolio.diagnostics:
            st.subheader("Highest Pheromone Trails")
            ph_df = pd.DataFrame(portfolio.diagnostics['top_pheromone'], columns=["Ticker", "Pheromone Level"])
            st.dataframe(ph_df, use_container_width=True, hide_index=True)
            
    # --- Row 3: Advanced Charts ---
    st.divider()
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Sector Exposure")
        if 'fund_sector' in operator_result.features.columns:
            sectors_df = operator_result.features[['fund_sector']].reset_index()
            sectors_df.columns = ["Ticker", "Sector"]
            sunburst_df = pd.merge(weights_df, sectors_df, on="Ticker", how="left")
            sunburst_df["Sector"] = sunburst_df["Sector"].fillna("Unknown")
            
            fig_sun = px.sunburst(
                sunburst_df,
                path=['Sector', 'Ticker'],
                values='Weight',
                title="Portfolio Weight by Sector"
            )
            fig_sun.update_traces(textinfo="label+percent entry")
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("Sector data not available.")

    with col4:
        st.subheader("Factor Correlation Heatmap")
        st.markdown("Displays how different features correlate across the universe.")
        
        if not operator_result.features.empty:
            # Get numeric columns only for correlation
            numeric_features = operator_result.features.select_dtypes(include=['float64', 'int64'])
            if not numeric_features.empty:
                corr_matrix = numeric_features.corr()
                fig_hm = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Cross-Factor Correlation Matrix"
                )
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Not enough feature data available to plot a correlation heatmap.")

async def _run_backtest_async(ctx: RunContext, universe: list[str], start_d: date, end_d: date):
    return await run_backtest(ctx, universe, start_d, end_d)

def run_backtest_ui(universe: list[str] | None, start_d: date, end_d: date, universe_mode: str):
    """Execute backtest and render rich results."""
    st.info("Initializing Backtester and Agents...")
    ctx = setup_context(universe_mode)
    
    with st.spinner("Running Walk-Forward Backtest... This will take a while."):
        result = asyncio.run(_run_backtest_async(ctx, universe, start_d, end_d))
        
    if not result:
        st.error("Backtest failed.")
        return
        
    st.success("Backtest Complete!")
    
    # --- Key Metrics ---
    metrics = result.metrics
    st.subheader("Performance Summary")
    
    cols = st.columns(5)
    cols[0].metric("Cumulative Return", f"{metrics.get('cumulative_return', 0):.2%}")
    cols[1].metric("Annual Return", f"{metrics.get('annualized_return', 0):.2%}")
    cols[2].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    cols[3].metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    cols[4].metric("Information Ratio", f"{metrics.get('information_ratio', 0):.2f}")
    
    # --- Equity Curve ---
    st.subheader("Equity Curve")
    chart_data = pd.DataFrame({
        "ACO Portfolio": result.equity_curve.values,
    }, index=result.equity_curve.index)
    
    if "equal_weight" in result.baseline_curves:
        chart_data["Equal Weight (Benchmark)"] = result.baseline_curves["equal_weight"].values
        
    st.line_chart(chart_data)
    
    # --- Drawdown Curve ---
    st.subheader("Drawdown Profile")
    drawdowns = result.equity_curve / result.equity_curve.cummax() - 1.0
    
    st.area_chart(drawdowns)

    # --- Monthly Returns Heatmap ---
    st.subheader("Monthly Returns Heatmap")
    
    try:
        monthly_series = result.equity_curve.resample("ME").last()
    except ValueError:
        monthly_series = result.equity_curve.resample("M").last()
        
    monthly_returns = monthly_series.pct_change().dropna()
    
    if len(monthly_returns) > 0:
        heatmap_df = pd.DataFrame({
            "Year": monthly_returns.index.year,
            "MonthStr": monthly_returns.index.strftime('%b'),
            "Return": monthly_returns.values
        })
        
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_df['MonthStr'] = pd.Categorical(heatmap_df['MonthStr'], categories=months_order, ordered=True)
        
        heatmap_pivot = heatmap_df.pivot_table(index="Year", columns="MonthStr", values="Return", aggfunc="mean").round(4)
        
        fig_month_hm = px.imshow(
            heatmap_pivot,
            text_auto=".2%",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Monthly Returns (%)"
        )
        st.plotly_chart(fig_month_hm, use_container_width=True)


# ==========================================
# Main Execution Trigger
# ==========================================
if run_button:
    # Build Universe list
    universe_list = None
    if universe_mode == "Custom Ticker List":
        universe_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not universe_list:
            universe_list = ["AAPL","MSFT"] # fallback so it doesn't crash if custom is chosen but empty
        
    if run_mode in ["Live Optimization", "Cached Optimization"]:
        run_optimization_ui(universe_list, universe_mode)
    else:
        if end_date <= start_date:
            st.error("End date must be strictly after start date.")
        else:
            run_backtest_ui(universe_list, start_date, end_date, universe_mode)
