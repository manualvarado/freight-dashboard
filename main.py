import streamlit as st
import pandas as pd
import re
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
    plot_driver_miles_heatmap,
    plot_bd_margin_distribution,
    plot_carrier_performance_analysis,
    generate_chart_analysis,
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

st.set_page_config(
    page_title="Freight Operations Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('🚛 Freight Operations Weekly Report Dashboard')

# Sidebar filters
st.sidebar.header("📊 Filters")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Convert numeric columns from US format to float
    numeric_columns = ['BROKER RATE', 'DRIVER RATE', 'FULL MILES TOTAL']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)
    
    # Convert additional billing columns if they exist
    if 'BROKER RATE (CFC)' in df.columns:
        df['BROKER RATE (CFC)'] = df['BROKER RATE (CFC)'].apply(parse_number)
    
    if 'BD MARGIN' in df.columns:
        df['BD MARGIN'] = df['BD MARGIN'].apply(parse_number)
    
    # Define valid statuses for both billing and driver pay
    valid_statuses = ['Booked', 'Delivered', 'On dispute', 'TONU Received']
    
    # Filter data for valid loads only
    valid_loads = df[df['LOAD STATUS'].isin(valid_statuses)]
    
    # Sidebar filters
    if 'DISPATCH NAME' in df.columns:
        dispatchers = ['All'] + list(df['DISPATCH NAME'].unique())
        selected_dispatcher = st.sidebar.selectbox("Select Dispatcher", dispatchers)
        if selected_dispatcher != 'All':
            valid_loads = valid_loads[valid_loads['DISPATCH NAME'] == selected_dispatcher]
    
    if 'TRAILER TYPE' in df.columns:
        trailer_types = ['All'] + list(df['TRAILER TYPE'].unique())
        selected_trailer = st.sidebar.selectbox("Select Trailer Type", trailer_types)
        if selected_trailer != 'All':
            valid_loads = valid_loads[valid_loads['TRAILER TYPE'] == selected_trailer]
    
    if 'LOAD\'S CARRIER COMPANY' in df.columns:
        carriers = ['All'] + list(df['LOAD\'S CARRIER COMPANY'].unique())
        selected_carrier = st.sidebar.selectbox("Select Carrier", carriers)
        if selected_carrier != 'All':
            valid_loads = valid_loads[valid_loads['LOAD\'S CARRIER COMPANY'] == selected_carrier]

    # Main dashboard
    st.header("📊 Key Performance Indicators")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_loads = len(valid_loads)
        st.metric("Total Valid Loads", f"{total_loads:,}")
    
    with col2:
        total_billing = valid_loads['BROKER RATE (CFC)'].sum()
        st.metric("Total Billing", f"${total_billing:,.0f}")
    
    with col3:
        total_driver_pay = valid_loads['DRIVER RATE'].sum()
        st.metric("Total Driver Pay", f"${total_driver_pay:,.0f}")
    
    with col4:
        if total_billing > 0:
            margin_percentage = ((total_billing - total_driver_pay) / total_billing * 100)
            st.metric("Gross Margin %", f"{margin_percentage:.1f}%")
        else:
            st.metric("Gross Margin %", "N/A")

    # Enhanced Visualizations
    st.header("📈 Enhanced Analytics")
    
    # Row 1: Billing Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Total Billing per Carrier")
        fig1 = plot_total_billing_per_carrier(valid_loads)
        st.plotly_chart(fig1, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis1 = generate_chart_analysis("carrier_billing", valid_loads)
            st.markdown(analysis1)
    
    with col2:
        st.subheader("2. Total Billing per Dispatcher")
        fig2 = plot_total_billing_per_dispatcher(valid_loads)
        st.plotly_chart(fig2, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis2 = generate_chart_analysis("dispatcher_billing", valid_loads)
            st.markdown(analysis2)
    
    # Row 2: Performance Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3. Dispatcher Performance Matrix")
        fig3 = plot_bd_margin_performance(valid_loads)
        st.plotly_chart(fig3, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis3 = generate_chart_analysis("dispatcher_performance", valid_loads)
            st.markdown(analysis3)
    
    with col2:
        st.subheader("4. Load Status Distribution per Dispatcher")
        fig4 = plot_loads_per_dispatcher_by_status(valid_loads)
        st.plotly_chart(fig4, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis4 = generate_chart_analysis("load_status", valid_loads)
            st.markdown(analysis4)
    
    # Row 3: Driver Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("5. Top 20 Drivers by Earnings")
        fig5 = plot_top_drivers_earnings(valid_loads)
        st.plotly_chart(fig5, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis5 = generate_chart_analysis("driver_earnings", valid_loads)
            st.markdown(analysis5)
    
    with col2:
        st.subheader("6. Billing by Trailer Type")
        fig6 = plot_billing_per_trailer_type(valid_loads)
        st.plotly_chart(fig6, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis6 = generate_chart_analysis("trailer_billing", valid_loads)
            st.markdown(analysis6)
    
    # Row 4: Time Series and Miles Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("7. Average Driver Earnings per Week")
        fig7 = plot_average_driver_earnings_weekly(valid_loads)
        st.plotly_chart(fig7, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis7 = generate_chart_analysis("weekly_earnings", valid_loads)
            st.markdown(analysis7)
    
    with col2:
        st.subheader("8. Total Miles per Carrier")
        fig8 = plot_total_miles_per_carrier(valid_loads)
        st.plotly_chart(fig8, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis8 = generate_chart_analysis("carrier_miles", valid_loads)
            st.markdown(analysis8)
    
    # Row 5: Advanced Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("9. Driver Miles Heatmap")
        fig9 = plot_driver_miles_heatmap(valid_loads)
        st.plotly_chart(fig9, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis9 = generate_chart_analysis("driver_heatmap", valid_loads)
            st.markdown(analysis9)
    
    with col2:
        st.subheader("10. BD Margin Distribution Analysis")
        fig10 = plot_bd_margin_distribution(valid_loads)
        st.plotly_chart(fig10, use_container_width=True)
        # Analysis section
        with st.expander("📊 Analysis & Insights"):
            analysis10 = generate_chart_analysis("margin_distribution", valid_loads)
            st.markdown(analysis10)
    
    # Row 6: Combined Carrier Performance Analysis (Full Width)
    st.subheader("11. Comprehensive Carrier Performance Analysis")
    st.markdown("""
    This analysis combines billing, miles, and revenue per mile to give you a complete view of carrier efficiency.
    - **Top Left**: Total billing by carrier
    - **Top Right**: Total miles by carrier  
    - **Bottom Left**: Revenue per mile by carrier
    - **Bottom Right**: Efficiency scatter plot (bubble size = total billing, color = revenue per mile)
    """)
    fig11 = plot_carrier_performance_analysis(valid_loads)
    st.plotly_chart(fig11, use_container_width=True)
    # Analysis section
    with st.expander("📊 Analysis & Insights"):
        analysis11 = generate_chart_analysis("carrier_performance", valid_loads)
        st.markdown(analysis11)
    
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
    
    # Export functionality
    st.header("💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary Report"):
            # Create summary report
            summary_data = {
                'Metric': ['Total Loads', 'Total Billing', 'Total Driver Pay', 'Gross Margin %'],
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
        if st.button("📈 Export Dispatcher Analysis"):
            dispatcher_analysis = valid_loads.groupby('DISPATCH NAME').agg({
                'BROKER RATE (CFC)': 'sum',
                'DRIVER RATE': 'sum',
                'LOAD ID': 'count'
            }).reset_index()
            dispatcher_analysis.columns = ['Dispatcher', 'Total Billing', 'Total Driver Pay', 'Load Count']
            st.download_button(
                label="Download Dispatcher Analysis",
                data=dispatcher_analysis.to_csv(index=False),
                file_name="dispatcher_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🚛 Export Driver Analysis"):
            driver_analysis = valid_loads.groupby('DRIVER NAME').agg({
                'DRIVER RATE': 'sum',
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
    - `DISPATCH NAME`: Name of the dispatcher
    - `BROKER RATE (CFC)`: Broker rate amount
    - `DRIVER RATE`: Driver pay amount
    - `DRIVER NAME`: Name of the driver
    - `TRAILER TYPE`: Type of trailer used
    - `LOAD STATUS`: Status of the load (Booked, Delivered, etc.)
    - `LOAD'S CARRIER COMPANY`: Carrier company name
    - `FULL MILES TOTAL`: Total miles for the load
    - `BD MARGIN`: BD margin field
    - `PICK UP DATE`: Pickup date for time series analysis
    """) 