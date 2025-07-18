import streamlit as st
import pandas as pd
import re
import base64
import os
from datetime import datetime, timedelta
from data_processor import (
    load_data,
    get_dispatcher_billing,
    get_cancellation_stats,
    get_trailer_type_billing,
    get_top_drivers,
    get_carrier_miles,
    get_broker_rate_cfc_stats,
    get_bd_margin_stats,
)
from enhanced_visualizer import (
    plot_total_billing_per_carrier,
    plot_total_billing_per_dispatcher,
    plot_bd_margin_performance,
    plot_loads_per_dispatcher_by_status,
    plot_top_drivers_earnings,
    plot_billing_per_trailer_type,
    plot_average_driver_earnings_weekly,
    plot_total_miles_per_carrier,
    plot_bd_margin_distribution,
    plot_carrier_performance_analysis,
    plot_driver_income_analysis,
    generate_chart_analysis,
    plot_weekly_driver_earnings_vs_target_faceted,
    plot_top_drivers_by_weekly_earnings_improved,
    plot_target_achievement_by_trailer_type_improved,
    plot_weekly_driver_miles_vs_target_faceted,
    plot_revenue_per_mile_per_dispatcher,
    plot_driver_revenue_per_mile_per_dispatcher,
    plot_weekly_driver_revenue_per_mile_vs_target_faceted,
)

def parse_number(value):
    """Convert number format to float, handling US format (1,150.00)."""
    if pd.isna(value) or value == '':
        return 0.0
    
    # Convert to string if it's not already
    value_str = str(value).strip()
    
    # Remove any currency symbols or extra spaces
    value_str = re.sub(r'[^\d.,]', '', value_str)
    
    # Handle US format (1,150.00 -> 1150.00)
    # If comma exists and is followed by exactly 3 digits, it's a thousands separator
    if ',' in value_str:
        # Remove commas (they are thousands separators in US format)
        value_str = value_str.replace(',', '')
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0

def get_weekly_target(trailer_type):
    """Get weekly target earnings based on trailer type."""
    if trailer_type in ['Flatbed', 'Stepdeck']:
        return 6000
    else:
        return 5500

def calculate_weekly_earnings(df, miles_col=None):
    """Calculate driver earnings by week (Tuesday to Monday) based on delivery dates."""
    driver_rate_col = find_driver_rate_column(df)
    if 'DELIVERY DATE' not in df.columns or 'DRIVER NAME' not in df.columns or not driver_rate_col:
        return pd.DataFrame()
    
    # Convert DELIVERY DATE to datetime with flexible parsing
    df_copy = df.copy()
    df_copy['DELIVERY DATE'] = pd.to_datetime(df_copy['DELIVERY DATE'], errors='coerce')
    
    # Debug: Check how many valid dates we have
    valid_dates = df_copy['DELIVERY DATE'].notna().sum()
    total_rows = len(df_copy)
    print(f"Debug: {valid_dates}/{total_rows} rows have valid delivery dates")
    
    # Filter out rows with invalid dates
    df_copy = df_copy[df_copy['DELIVERY DATE'].notna()].copy()
    
    if len(df_copy) == 0:
        print("Debug: No valid delivery dates found, returning empty DataFrame")
        return pd.DataFrame()
    
    # Calculate week start (Tuesday) for each delivery date
    def get_week_start(date):
        if pd.isna(date):
            return pd.NaT
        # Find the most recent Tuesday (weekday 1)
        days_since_tuesday = (date.weekday() - 1) % 7
        return date - timedelta(days=days_since_tuesday)
    
    df_copy['WEEK_START'] = df_copy['DELIVERY DATE'].apply(get_week_start)
    
    # Group by driver and week - include miles if available
    driver_rate_col = find_driver_rate_column(df_copy)
    trailer_col = find_trailer_column(df_copy)
    
    if not trailer_col:
        st.error("❌ Trailer column not found! Expected one of: 'TRAILER TYPE', 'TRAILER', 'trailer_type', 'trailer'")
        return pd.DataFrame()
    
    if miles_col and miles_col in df_copy.columns:
        # Include miles in the grouping
        weekly_earnings = df_copy.groupby(['DRIVER NAME', 'WEEK_START', trailer_col]).agg({
            driver_rate_col: 'sum',
            miles_col: 'sum'
        }).reset_index()
    else:
        # Original grouping without miles
        weekly_earnings = df_copy.groupby(['DRIVER NAME', 'WEEK_START', trailer_col])[driver_rate_col].sum().reset_index()
    
    # Calculate target and percentage
    weekly_earnings['TARGET'] = weekly_earnings[trailer_col].apply(get_weekly_target)
    weekly_earnings['PERCENTAGE_TO_TARGET'] = (weekly_earnings[driver_rate_col] / weekly_earnings['TARGET'] * 100).round(1)
    
    print(f"Debug: Generated {len(weekly_earnings)} weekly earnings records")
    print(f"Debug: Week range: {weekly_earnings['WEEK_START'].min()} to {weekly_earnings['WEEK_START'].max()}")

    # Add trailer group column
    def trailer_group(trailer):
        if trailer in ['Flatbed', 'Stepdeck']:
            return 'Flatbed/Stepdeck'
        elif trailer in ['DryVan', 'Reefer', 'Power Only']:
            return 'Dry Van/Reefer/Power Only'
        else:
            return trailer
    weekly_earnings['TRAILER GROUP'] = weekly_earnings[trailer_col].apply(trailer_group)

    return weekly_earnings

def find_column(df, target):
    target_clean = target.replace('_', '').replace(' ', '').upper()
    for col in df.columns:
        if col.replace('_', '').replace(' ', '').upper() == target_clean:
            return col
    return None

def find_broker_rate_column(df):
    """Find the broker rate column, handling both old and new naming conventions."""
    # Try new format first
    if 'BROKER RATE (FC) [$' in df.columns:
        return 'BROKER RATE (FC) [$'
    # Try old format
    elif 'BROKER RATE (CFC)' in df.columns:
        return 'BROKER RATE (CFC)'
    # Try other variations
    elif 'BROKER RATE' in df.columns:
        return 'BROKER RATE'
    # Try lowercase variations
    elif 'broker_rate' in df.columns:
        return 'broker_rate'
    # Try case-insensitive search
    else:
        for col in df.columns:
            if 'broker' in col.lower() and 'rate' in col.lower():
                return col
    return None

def find_driver_rate_column(df):
    """Find the driver rate column, handling both old and new naming conventions."""
    # Try new format first
    if 'DRIVER RATE [$]' in df.columns:
        return 'DRIVER RATE [$]'
    # Try old format
    elif 'DRIVER RATE' in df.columns:
        return 'DRIVER RATE'
    # Try lowercase variations
    elif 'driver_rate' in df.columns:
        return 'driver_rate'
    # Try case-insensitive search
    else:
        for col in df.columns:
            if 'driver' in col.lower() and 'rate' in col.lower():
                return col
    return None

def find_trailer_column(df):
    """Find the trailer column, handling different naming conventions."""
    # Try different variations
    if 'TRAILER TYPE' in df.columns:
        return 'TRAILER TYPE'
    elif 'TRAILER' in df.columns:
        return 'TRAILER'
    elif 'trailer_type' in df.columns:
        return 'trailer_type'
    elif 'trailer' in df.columns:
        return 'trailer'
    # Try case-insensitive search
    else:
        for col in df.columns:
            if 'trailer' in col.lower():
                return col
    return None

def find_dispatcher_column(df):
    """Find the dispatcher column, handling different naming conventions."""
    # Try different variations
    if 'DISPATCH NAME' in df.columns:
        return 'DISPATCH NAME'
    elif 'FC NAME' in df.columns:
        return 'FC NAME'
    elif 'DISPATCH' in df.columns:
        return 'DISPATCH'
    elif 'dispatcher' in df.columns:
        return 'dispatcher'
    # Try case-insensitive search
    else:
        for col in df.columns:
            col_lower = col.lower()
            if 'dispatch' in col_lower or ('fc' in col_lower and 'name' in col_lower):
                return col
    return None

def find_bd_margin_column(df):
    """Find the BD margin column, handling different naming conventions."""
    candidates = ['BD MARGIN [$]', 'BD MARGIN', 'bd_margin', 'bd margin']
    for c in candidates:
        if c in df.columns:
            return c
    # Try case-insensitive search
    for col in df.columns:
        if 'bd' in col.lower() and 'margin' in col.lower():
            return col
    return None

def get_logo_base64():
    """Get base64 encoded logo for display."""
    logo_path = "jc_logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

st.set_page_config(
    page_title="Freight Operations Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better spacing
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stPlotlyChart {
        margin-bottom: 3rem !important;
    }
    .stSubheader {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    .stMetric {
        margin-bottom: 1rem !important;
    }
    .stExpander {
        margin-top: 1rem !important;
        margin-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Display header with logo
logo_base64 = get_logo_base64()
if logo_base64:
    st.markdown(
        f'''
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <a href="http://localhost:8504/" target="_self" style="font-size: 1.1em; text-decoration: none; color: #007bff; font-weight: bold;">🔄 Switch to Monthly Dashboard</a>
            <img src="data:image/png;base64,{logo_base64}" alt="JC Global Logo" style="height: 50px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'''
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <a href="http://localhost:8504/" target="_self" style="font-size: 1.1em; text-decoration: none; color: #007bff; font-weight: bold;">🔄 Switch to Monthly Dashboard</a>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 20px; letter-spacing: 2px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">JC</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
st.title("📊 Weekly Operations Review Dashboard")

# Sidebar filters
st.sidebar.header("📊 Filters")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Debug: Show available columns
    with st.expander("🔍 Debug: Available Columns", expanded=False):
        st.write("**All columns in your CSV file:**")
        st.write(list(df.columns))
        
        broker_rate_col = find_broker_rate_column(df)
        driver_rate_col = find_driver_rate_column(df)
        trailer_col = find_trailer_column(df)
        
        st.write(f"**Detected broker rate column:** {broker_rate_col}")
        st.write(f"**Detected driver rate column:** {driver_rate_col}")
        st.write(f"**Detected trailer column:** {trailer_col}")
        
        if not broker_rate_col:
            st.error("❌ No broker rate column found! Expected one of: 'BROKER RATE (FC) [$', 'BROKER RATE (CFC)', 'BROKER RATE'")
        if not driver_rate_col:
            st.error("❌ No driver rate column found! Expected one of: 'DRIVER RATE [$]', 'DRIVER RATE'")
        if not trailer_col:
            st.error("❌ No trailer column found! Expected one of: 'TRAILER TYPE', 'TRAILER', 'trailer_type', 'trailer'")
    
    # Convert numeric columns from US format to float
    numeric_columns = ['BROKER RATE', 'FULL MILES TOTAL']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)
    
    # Handle driver rate column separately
    driver_rate_col = find_driver_rate_column(df)
    if driver_rate_col:
        df[driver_rate_col] = df[driver_rate_col].apply(parse_number)
    
    # Convert additional billing columns if they exist
    broker_rate_col = find_broker_rate_column(df)
    if broker_rate_col:
        df[broker_rate_col] = df[broker_rate_col].apply(parse_number)
    
    bd_margin_col = find_bd_margin_column(df)
    if bd_margin_col:
        df[bd_margin_col] = df[bd_margin_col].apply(parse_number)
    
    # Define valid statuses for both billing and driver pay
    valid_statuses = ['Booked', 'Delivered', 'On dispute', 'TONU Received']
    
    # Filter data for valid loads only
    valid_loads = df[df['LOAD STATUS'].isin(valid_statuses)].copy()
    
    # Calculate week start dates for all data to enable week filtering
    if 'DELIVERY DATE' in valid_loads.columns:
        valid_loads['DELIVERY DATE'] = pd.to_datetime(valid_loads['DELIVERY DATE'], errors='coerce')
        valid_loads = valid_loads[valid_loads['DELIVERY DATE'].notna()].copy()
        
        def get_week_start(date):
            if pd.isna(date):
                return pd.NaT
            # Find the most recent Tuesday (weekday 1)
            days_since_tuesday = (date.weekday() - 1) % 7
            return date - timedelta(days=days_since_tuesday)
        
        valid_loads['WEEK_START'] = valid_loads['DELIVERY DATE'].apply(get_week_start)
        
        # Get available weeks for selection
        available_weeks = sorted(valid_loads['WEEK_START'].unique())
        week_labels = [f"{week.strftime('%m/%d/%Y')} - {(week + timedelta(days=6)).strftime('%m/%d/%Y')}" for week in available_weeks]
        
        # Week selection filter
        st.sidebar.subheader("📅 Week Selection")
        selected_weeks = st.sidebar.multiselect(
            "Select Weeks (Tuesday-Monday)",
            options=week_labels,
            default=week_labels,  # Select all weeks by default
            help="Select specific weeks to analyze. Each week runs from Tuesday to Monday."
        )
        
        # Filter data for selected weeks
        if selected_weeks:
            selected_week_starts = [available_weeks[i] for i, label in enumerate(week_labels) if label in selected_weeks]
            valid_loads = valid_loads[valid_loads['WEEK_START'].isin(selected_week_starts)].copy()
        else:
            st.warning("⚠️ Please select at least one week to analyze.")
            st.stop()
    else:
        # If no delivery date column, create a dummy selected_weeks variable
        selected_weeks = []
    
    # Sidebar filters
    dispatcher_col = find_dispatcher_column(df)
    if dispatcher_col:
        dispatchers = ['All'] + list(df[dispatcher_col].unique())
        selected_dispatcher = st.sidebar.selectbox("Select Dispatcher", dispatchers)
        if selected_dispatcher != 'All':
            valid_loads = valid_loads[valid_loads[dispatcher_col] == selected_dispatcher]
    
    miles_col = find_column(df, 'FULL MILES TOTAL')
    if miles_col:
        df[miles_col] = df[miles_col].apply(parse_number)
    else:
        # Try to find any column with 'miles' in the name
        miles_columns = [col for col in df.columns if 'miles' in col.lower() or 'mile' in col.lower()]
        if miles_columns:
            miles_col = miles_columns[0]  # Use the first one found
        else:
            miles_col = None
    
    trailer_col = find_trailer_column(df)
    if trailer_col:
        trailer_types = ['All'] + [str(t).replace('DryVan', 'Dry Van') for t in df[trailer_col] if isinstance(t, str)]
        selected_trailer = st.sidebar.selectbox("Select Trailer Type", trailer_types)
        if selected_trailer != 'All':
            valid_loads = valid_loads[valid_loads[trailer_col].replace({'DryVan': 'Dry Van'}) == selected_trailer]
    
    if 'LOAD\'S CARRIER COMPANY' in df.columns:
        carriers = ['All'] + list(df['LOAD\'S CARRIER COMPANY'].unique())
        selected_carrier = st.sidebar.selectbox("Select Carrier", carriers)
        if selected_carrier != 'All':
            valid_loads = valid_loads[valid_loads['LOAD\'S CARRIER COMPANY'] == selected_carrier]
    
    # RPM Target Configuration
    st.sidebar.subheader("💰 Revenue per Mile Targets")
    flatbed_rpm_target = st.sidebar.number_input(
        "Flatbed/Stepdeck RPM Target ($)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Target revenue per mile for Flatbed/Stepdeck trailers"
    )
    dryvan_rpm_target = st.sidebar.number_input(
        "Dry Van/Reefer/Power Only RPM Target ($)",
        min_value=0.0,
        max_value=10.0,
        value=1.8,
        step=0.1,
        help="Target revenue per mile for Dry Van/Reefer/Power Only trailers"
    )

    # Main dashboard
    st.header("📊 Key Performance Indicators")
    
    # Show KPI data for each selected week individually
    if len(selected_weeks) > 1:
        st.subheader("📈 KPI by Week")
        
        # Group by week and calculate KPIs
        # Use any available column for counting loads (could be index or any unique identifier)
        count_column = None
        for col in ['LOAD ID', 'LOAD_ID', 'load_id', 'Load ID']:
            if col in valid_loads.columns:
                count_column = col
                break
        
        if count_column is None:
            # If no load ID column found, use the index for counting
            broker_rate_col = find_broker_rate_column(valid_loads)
            driver_rate_col = find_driver_rate_column(valid_loads)
            weekly_kpis = valid_loads.groupby('WEEK_START').agg({
                broker_rate_col: 'sum',
                driver_rate_col: 'sum'
            }).reset_index()
            weekly_kpis['LOAD_COUNT'] = valid_loads.groupby('WEEK_START').size().values
        else:
            broker_rate_col = find_broker_rate_column(valid_loads)
            driver_rate_col = find_driver_rate_column(valid_loads)
            weekly_kpis = valid_loads.groupby('WEEK_START').agg({
                count_column: 'count',
                broker_rate_col: 'sum',
                driver_rate_col: 'sum'
            }).reset_index()
            weekly_kpis = weekly_kpis.rename(columns={count_column: 'LOAD_COUNT'})
        
        weekly_kpis['WEEK_LABEL'] = weekly_kpis['WEEK_START'].apply(
            lambda x: f"{x.strftime('%m/%d/%Y')} - {(x + timedelta(days=6)).strftime('%m/%d/%Y')}"
        )
        broker_rate_col = find_broker_rate_column(valid_loads)
        driver_rate_col = find_driver_rate_column(valid_loads)
        weekly_kpis['GROSS_MARGIN'] = ((weekly_kpis[broker_rate_col] - weekly_kpis[driver_rate_col]) / weekly_kpis[broker_rate_col] * 100).round(1)
        
        # Display weekly KPIs in a table
        broker_rate_col = find_broker_rate_column(valid_loads)
        driver_rate_col = find_driver_rate_column(valid_loads)
        kpi_display = weekly_kpis[['WEEK_LABEL', 'LOAD_COUNT', broker_rate_col, driver_rate_col, 'GROSS_MARGIN']].copy()
        kpi_display.columns = ['Week Period', 'Total Loads', 'Total Billing', 'Total Driver Pay', 'B-Rate %']
        kpi_display['Total Billing'] = kpi_display['Total Billing'].apply(lambda x: f"${x:,.0f}")
        kpi_display['Total Driver Pay'] = kpi_display['Total Driver Pay'].apply(lambda x: f"${x:,.0f}")
        kpi_display['B-Rate %'] = kpi_display['B-Rate %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(kpi_display, use_container_width=True)
        
        # Show overall totals
        st.subheader("📊 Overall Totals (All Selected Weeks)")
    
    # KPI Cards (overall totals)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_loads = len(valid_loads)
        st.metric("Total Valid Loads", f"{total_loads:,}")
    
    with col2:
        broker_rate_col = find_broker_rate_column(valid_loads)
        if broker_rate_col:
            total_billing = valid_loads[broker_rate_col].sum()
            st.metric("Total Billing", f"${total_billing:,.0f}")
        else:
            st.error("❌ Broker rate column not found!")
            st.write("Available columns:", list(valid_loads.columns))
            st.metric("Total Billing", "N/A")
    
    with col3:
        driver_rate_col = find_driver_rate_column(valid_loads)
        if driver_rate_col:
            total_driver_pay = valid_loads[driver_rate_col].sum()
            st.metric("Total Driver Pay", f"${total_driver_pay:,.0f}")
        else:
            st.error("❌ Driver rate column not found!")
            st.metric("Total Driver Pay", "N/A")
    
    with col4:
        if broker_rate_col and driver_rate_col and total_billing > 0:
            margin_percentage = ((total_billing - total_driver_pay) / total_billing * 100)
            st.metric("B-Rate %", f"{margin_percentage:.1f}%")
        else:
            st.metric("B-Rate %", "N/A")

    # Driver Income KPI Section
    st.header("💰 Driver Income Analysis")
    
    # Calculate weekly earnings
    weekly_earnings = calculate_weekly_earnings(valid_loads, miles_col)
    
    # Debug information
    if weekly_earnings.empty:
        st.warning("⚠️ No weekly earnings data available. This might be due to:")
        st.markdown("""
        - Missing or invalid 'DELIVERY DATE' column
        - Date format not recognized
        - No valid delivery dates in the data
        
        **Please check your CSV file has a 'DELIVERY DATE' column with valid dates.**
        """)
    else:
        # Show driver income data for each selected week individually
        if len(selected_weeks) > 1:
            st.subheader("📈 Driver Income by Week")
            
            # Group by week and calculate driver income metrics
            driver_rate_col = find_driver_rate_column(weekly_earnings)
            weekly_driver_metrics = weekly_earnings.groupby('WEEK_START').agg({
                driver_rate_col: ['mean', 'sum'],
                'PERCENTAGE_TO_TARGET': 'mean',
                'DRIVER NAME': 'nunique'
            }).reset_index()
            
            weekly_driver_metrics.columns = ['WEEK_START', 'AVG_WEEKLY_EARNINGS', 'TOTAL_WEEKLY_EARNINGS', 'AVG_PERCENTAGE_TO_TARGET', 'UNIQUE_DRIVERS']
            weekly_driver_metrics['WEEK_LABEL'] = weekly_driver_metrics['WEEK_START'].apply(
                lambda x: f"{x.strftime('%m/%d/%Y')} - {(x + timedelta(days=6)).strftime('%m/%d/%Y')}"
            )
            
            # Calculate drivers above target for each week
            drivers_above_target_by_week = []
            for week_start in weekly_driver_metrics['WEEK_START']:
                week_data = weekly_earnings[weekly_earnings['WEEK_START'] == week_start]
                above_target = len(week_data[week_data['PERCENTAGE_TO_TARGET'] >= 100])
                total_drivers = len(week_data)
                percentage = (above_target / total_drivers * 100) if total_drivers > 0 else 0
                drivers_above_target_by_week.append(percentage)
            
            weekly_driver_metrics['DRIVERS_ABOVE_TARGET_PCT'] = drivers_above_target_by_week
            
            # Display weekly driver metrics in a table
            driver_display = weekly_driver_metrics[['WEEK_LABEL', 'UNIQUE_DRIVERS', 'AVG_WEEKLY_EARNINGS', 'TOTAL_WEEKLY_EARNINGS', 'DRIVERS_ABOVE_TARGET_PCT', 'AVG_PERCENTAGE_TO_TARGET']].copy()
            driver_display.columns = ['Week Period', 'Unique Drivers', 'Avg Weekly Earnings', 'Total Weekly Earnings', 'Drivers Above Target %', 'Avg % to Target']
            driver_display['Avg Weekly Earnings'] = driver_display['Avg Weekly Earnings'].apply(lambda x: f"${x:,.0f}")
            driver_display['Total Weekly Earnings'] = driver_display['Total Weekly Earnings'].apply(lambda x: f"${x:,.0f}")
            driver_display['Drivers Above Target %'] = driver_display['Drivers Above Target %'].apply(lambda x: f"{x:.1f}%")
            driver_display['Avg % to Target'] = driver_display['Avg % to Target'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(driver_display, use_container_width=True)
            
            # Show overall totals
            st.subheader("📊 Overall Driver Income Totals (All Selected Weeks)")
        
        # Driver Income KPI Cards (overall totals)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            driver_rate_col = find_driver_rate_column(weekly_earnings)
            avg_weekly_earnings = weekly_earnings[driver_rate_col].mean()
            st.metric("Avg Weekly Earnings", f"${avg_weekly_earnings:,.0f}")
        
        with col2:
            drivers_above_target = len(weekly_earnings[weekly_earnings['PERCENTAGE_TO_TARGET'] >= 100])
            total_weeks = len(weekly_earnings)
            target_percentage = (drivers_above_target / total_weeks * 100) if total_weeks > 0 else 0
            st.metric("Drivers Above Target", f"{target_percentage:.1f}%")
        
        with col3:
            trailer_col = find_trailer_column(weekly_earnings)
            flatbed_avg = weekly_earnings[weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])][driver_rate_col].mean()
            st.metric("Flatbed/Stepdeck Avg", f"${flatbed_avg:,.0f}")
        
        with col4:
            other_avg = weekly_earnings[~weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])][driver_rate_col].mean()
            st.metric("Other Trailers Avg", f"${other_avg:,.0f}")
        
        # Weekly Earnings Table
        st.subheader("📋 Weekly Driver Earnings by Trailer Type")
        driver_rate_col = find_driver_rate_column(weekly_earnings)
        trailer_col = find_trailer_column(weekly_earnings)
        weekly_summary = weekly_earnings.groupby([trailer_col, 'WEEK_START']).agg({
            driver_rate_col: 'sum',
            'DRIVER NAME': 'count',
            'PERCENTAGE_TO_TARGET': 'mean'
        }).reset_index()
        weekly_summary.columns = ['Trailer Type', 'Week Starting', 'Total Earnings', 'Driver Count', 'Avg % to Target']
        weekly_summary['Trailer Type'] = weekly_summary['Trailer Type'].replace({'DryVan': 'Dry Van'})
        weekly_summary['Week Starting'] = weekly_summary['Week Starting'].dt.strftime('%Y-%m-%d')
        st.dataframe(weekly_summary, use_container_width=True)
        
        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- Driver Performance Summary Table ---
        targets = {
            'Flatbed/Stepdeck': 6000,
            'Dry Van/Reefer/Power Only': 5500
        }
        trailer_groups = [t for t in targets.keys() if t in weekly_earnings['TRAILER GROUP'].unique()]
        summary_md = """### 🚦 Driver Performance Summary by Trailer Type\n"""
        for group in trailer_groups:
            sub = weekly_earnings[weekly_earnings['TRAILER GROUP'] == group].copy()
            total_drivers = len(sub)
            overachievers = len(sub[sub['PERCENTAGE_TO_TARGET'] > 110])
            on_target = len(sub[(sub['PERCENTAGE_TO_TARGET'] >= 100) & (sub['PERCENTAGE_TO_TARGET'] <= 110)])
            watchlist = len(sub[(sub['PERCENTAGE_TO_TARGET'] >= 80) & (sub['PERCENTAGE_TO_TARGET'] < 100)])
            underperformers = len(sub[sub['PERCENTAGE_TO_TARGET'] < 80])
            summary_md += f"\n**{group}**: Total: {total_drivers} drivers\n"
            summary_md += f"- 💚 Overachievers: {overachievers} ({overachievers/total_drivers*100:.1f}%)\n"
            summary_md += f"- 🟢 On Target: {on_target} ({on_target/total_drivers*100:.1f}%)\n"
            summary_md += f"- 🟡 Watchlist: {watchlist} ({watchlist/total_drivers*100:.1f}%)\n"
            summary_md += f"- 🔴 Underperformers: {underperformers} ({underperformers/total_drivers*100:.1f}%)\n"
        st.markdown(summary_md)
        # --- End summary table ---

        # Weekly Driver Performance Charts (Separate Expanders)
        
        # Weekly Driver Earnings vs Target by Trailer Type
        with st.expander("📊 Weekly Driver Earnings vs Target by Trailer Type", expanded=False):
            # Add earnings performance summary above chart
            from enhanced_visualizer import generate_driver_performance_summary
            earnings_summary = generate_driver_performance_summary(weekly_earnings, "earnings")
            st.markdown(f"""
            **🚦 Driver Performance Summary (Earnings):**
            {earnings_summary}
            """)
            
            fig_faceted = plot_weekly_driver_earnings_vs_target_faceted(weekly_earnings)
            st.plotly_chart(fig_faceted, use_container_width=True)

        # Weekly Driver Miles vs Target by Trailer Type
        with st.expander("📊 Weekly Driver Miles vs Target by Trailer Type", expanded=False):
            if miles_col:
                # Add miles performance summary above chart
                miles_summary = generate_driver_performance_summary(weekly_earnings, "miles")
                st.markdown(f"""
                **🚦 Driver Performance Summary (Miles):**
                {miles_summary}
                """)
                
                fig_miles = plot_weekly_driver_miles_vs_target_faceted(weekly_earnings, miles_col=miles_col)
                st.plotly_chart(fig_miles, use_container_width=True)
            else:
                st.warning("⚠️ Miles chart not available - no miles column found in the data")

        # Weekly Driver Revenue per Mile vs Target by Trailer Type
        with st.expander("📊 Weekly Driver Revenue per Mile vs Target by Trailer Type", expanded=False):
            if miles_col:
                # Add revenue per mile performance summary above chart
                revenue_summary = generate_driver_performance_summary(weekly_earnings, "revenue_per_mile", flatbed_rpm_target, dryvan_rpm_target)
                st.markdown(f"""
                **🚦 Driver Performance Summary (Revenue per Mile):**
                {revenue_summary}
                """)
                
                fig_revenue_per_mile = plot_weekly_driver_revenue_per_mile_vs_target_faceted(weekly_earnings, miles_col=miles_col, flatbed_rpm_target=flatbed_rpm_target, dryvan_rpm_target=dryvan_rpm_target)
                st.plotly_chart(fig_revenue_per_mile, use_container_width=True)
            else:
                st.warning("⚠️ Revenue per mile chart not available - no miles column found in the data")

        st.subheader("📊 Top Drivers by Weekly Earnings (Improved)")
        fig_top = plot_top_drivers_by_weekly_earnings_improved(weekly_earnings)
        st.plotly_chart(fig_top, use_container_width=True)

        st.subheader("📊 Target Achievement by Trailer Type (Improved)")
        fig_ach = plot_target_achievement_by_trailer_type_improved(weekly_earnings)
        st.plotly_chart(fig_ach, use_container_width=True)

        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis_income = generate_chart_analysis("driver_income", weekly_earnings)
            st.markdown(analysis_income)
        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

    # Enhanced Visualizations
    st.header("📈 Enhanced Analytics")
    
    # Row 1: Billing Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Billing per Carrier")
        fig1 = plot_total_billing_per_carrier(valid_loads)
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis1 = generate_chart_analysis("carrier_billing", valid_loads)
            st.markdown(analysis1)
    
    with col2:
        st.subheader("Total Billing per Dispatcher")
        fig2 = plot_total_billing_per_dispatcher(valid_loads)
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis2 = generate_chart_analysis("dispatcher_billing", valid_loads)
            st.markdown(analysis2)
    
    # Add spacing between rows
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Row 2: Performance Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dispatcher Performance Matrix")
        fig3 = plot_bd_margin_performance(valid_loads)
        st.plotly_chart(fig3, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis3 = generate_chart_analysis("dispatcher_performance", valid_loads)
            st.markdown(analysis3)
    
    with col2:
        st.subheader("Load Status Distribution per Dispatcher")
        fig4 = plot_loads_per_dispatcher_by_status(df)
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis4 = generate_chart_analysis("load_status", df)
            st.markdown(analysis4)
    
    # Add spacing between rows
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Row 3: Driver Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Drivers by Earnings")
        fig5 = plot_top_drivers_earnings(valid_loads)
        st.plotly_chart(fig5, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis5 = generate_chart_analysis("driver_earnings", valid_loads)
            st.markdown(analysis5)
    
    with col2:
        st.subheader("Billing by Trailer Type")
        fig6 = plot_billing_per_trailer_type(valid_loads)
        st.plotly_chart(fig6, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis6 = generate_chart_analysis("trailer_billing", valid_loads)
            st.markdown(analysis6)
    
    # Add spacing between rows
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Row 4: Revenue per Mile Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue per Mile per Dispatcher")
        fig8 = plot_revenue_per_mile_per_dispatcher(valid_loads)
        st.plotly_chart(fig8, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis8 = generate_chart_analysis("revenue_per_mile_dispatcher", valid_loads)
            st.markdown(analysis8)
    
    with col2:
        st.subheader("Driver Revenue per Mile per Dispatcher")
        fig8b = plot_driver_revenue_per_mile_per_dispatcher(valid_loads)
        st.plotly_chart(fig8b, use_container_width=True)
        with st.expander("📊 Analysis & Insights"):
            analysis8b = generate_chart_analysis("driver_revenue_per_mile_dispatcher", valid_loads)
            st.markdown(analysis8b)
    
    # Add spacing between rows
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Add spacing between rows
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Row 6: Combined Carrier Performance Analysis (Full Width)
    st.subheader("Comprehensive Carrier Performance Analysis")
    st.markdown("""
    This analysis combines billing, miles, and revenue per mile to give you a complete view of carrier efficiency.
    - **Top Left**: Total billing by carrier
    - **Top Right**: Total miles by carrier  
    - **Bottom Left**: Revenue per mile by carrier
    - **Bottom Right**: Efficiency scatter plot (bubble size = total billing, color = revenue per mile)
    """)
    fig11_top, fig11_bottom = plot_carrier_performance_analysis(valid_loads)
    st.plotly_chart(fig11_top, use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.plotly_chart(fig11_bottom, use_container_width=True)
    with st.expander("📊 Analysis & Insights"):
        analysis11 = generate_chart_analysis("carrier_performance", valid_loads)
        st.markdown(analysis11)
    # Add spacing before data tables
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Data Tables Section
    st.header("📋 Detailed Data Tables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dispatcher Billing Summary")
        dispatcher_billing_df = get_dispatcher_billing(valid_loads)
        st.dataframe(dispatcher_billing_df, use_container_width=True)
    
    with col2:
        st.subheader("Carrier Miles Summary")
        carrier_miles_df = get_carrier_miles(valid_loads)
        st.dataframe(carrier_miles_df, use_container_width=True)
    
    # Add spacing before export section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Export functionality
    st.header("💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary Report"):
            # Create summary report
            total_loads = len(valid_loads)
            total_billing = valid_loads[broker_rate_col].sum() if broker_rate_col else 0
            total_driver_pay = valid_loads[driver_rate_col].sum() if driver_rate_col else 0
            margin_percentage = ((total_billing - total_driver_pay) / total_billing * 100) if total_billing > 0 else 0
            
            summary_data = {
                'Metric': ['Total Loads', 'Total Billing', 'Total Driver Pay', 'B-Rate %'],
                'Value': [total_loads, f"${total_billing:,.0f}", f"${total_driver_pay:,.0f}", f"{margin_percentage:.1f}%"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.download_button(
                label="Download Summary CSV",
                data=summary_df.to_csv(index=False),
                file_name="freight_summary_report.csv",
                mime="text/csv"
            )
    
    with col2:
        # Dispatcher Analysis
        dispatcher_col = find_dispatcher_column(valid_loads)
        broker_rate_col = find_broker_rate_column(valid_loads)
        driver_rate_col = find_driver_rate_column(valid_loads)
        bd_margin_col = find_bd_margin_column(valid_loads)
        if dispatcher_col and broker_rate_col and driver_rate_col and bd_margin_col:
            dispatcher_analysis = valid_loads.groupby(dispatcher_col).agg({
                broker_rate_col: 'sum',
                driver_rate_col: 'sum',
                bd_margin_col: 'sum',
                'LOAD ID': 'count'
            }).reset_index()
            dispatcher_analysis.columns = ['Dispatcher', 'Total Billing', 'Total Driver Pay', 'Total BD Margin', 'Load Count']
            st.download_button(
                label="Download Dispatcher Analysis",
                data=dispatcher_analysis.to_csv(index=False),
                file_name="dispatcher_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🚛 Export Driver Analysis"):
            driver_analysis = valid_loads.groupby('DRIVER NAME').agg({
                'DRIVER RATE [$]': 'sum',
                'FULL MILES TOTAL': 'sum',
                'LOAD ID': 'count'
            }).reset_index()
            driver_analysis.columns = ['Driver', 'Total Earnings', 'Total Miles', 'Load Count']
            st.download_button(
                label="Download Driver Analysis",
                data=driver_analysis.to_csv(index=False),
                file_name="driver_analysis.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Please upload a CSV file using the sidebar to begin analysis.")
    
    st.markdown("""
    ### 📋 Expected CSV Format
    
    The application expects CSV files with the following columns:
    - `LOAD ID`: Unique load identifier
    - `DISPATCH NAME` or `FC NAME`: Name of the dispatcher
    - `BROKER RATE (FC) [$` or `BROKER RATE (CFC)`: Broker rate amount
    - `DRIVER RATE [$]` or `DRIVER RATE`: Driver pay amount
    - `DRIVER NAME`: Name of the driver
    - `TRAILER TYPE` or `TRAILER`: Type of trailer used
    - `LOAD STATUS`: Status of the load (Booked, Delivered, etc.)
    - `LOAD'S CARRIER COMPANY`: Carrier company name
    - `FULL MILES TOTAL`: Total miles for the load
    - `BD MARGIN`: BD margin field
    - `DELIVERY DATE`: Delivery date for time series analysis
    """) 