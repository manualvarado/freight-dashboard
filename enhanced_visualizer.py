import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import colorsys
import random

# Helper functions for dynamic column detection
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
    # Try old format
    if 'DRIVER RATE' in df.columns:
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
    if 'trailer_col' in df.columns:
        return 'trailer_col'
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

def get_week_start(date):
    if pd.isna(date):
        return pd.NaT
    # Find the most recent Tuesday (weekday 1)
    days_since_tuesday = (date.weekday() - 1) % 7
    return date - timedelta(days=days_since_tuesday)

def plot_total_billing_per_carrier(df):
    """1. Total Billing per Carrier - Horizontal Bar Chart (sorted descending)"""
    if df.empty or 'LOAD\'S CARRIER COMPANY' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find broker rate column
    broker_rate_col = find_broker_rate_column(df)
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by carrier and sum billing
    carrier_billing = df.groupby('LOAD\'S CARRIER COMPANY')[broker_rate_col].sum().reset_index()
    carrier_billing = carrier_billing.sort_values(broker_rate_col, ascending=True)  # For horizontal bar, ascending=True shows highest at top
    
    fig = px.bar(
        carrier_billing, 
        x=broker_rate_col, 
        y='LOAD\'S CARRIER COMPANY',
        orientation='h',
        title='Total Billing per Carrier',
        labels={broker_rate_col: 'Total Billing ($)', 'LOAD\'S CARRIER COMPANY': 'Carrier'},
        color=broker_rate_col,
        color_continuous_scale='Blues',
        text=carrier_billing[broker_rate_col].apply(lambda x: f'${x:,.2f}')
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Total Billing: $%{x:,.2f}<extra></extra>',
        textposition='outside'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Total Billing ($)",
        yaxis_title="Carrier"
    )
    
    return fig

def plot_total_billing_per_dispatcher(df):
    """2. Total Billing per Dispatcher - Horizontal Bar Chart + BD Margin Overlay + Data Labels"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find dispatcher, broker rate, and BD margin columns
    dispatcher_col = find_dispatcher_column(df)
    broker_rate_col = find_broker_rate_column(df)
    bd_margin_col = find_bd_margin_column(df)
    
    if not dispatcher_col:
        return go.Figure().add_annotation(text="Dispatcher column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not bd_margin_col:
        return go.Figure().add_annotation(text="BD margin column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by dispatcher and sum both billing and BD margin
    dispatcher_stats = df.groupby(dispatcher_col).agg({
        broker_rate_col: 'sum',
        bd_margin_col: 'sum'
    }).reset_index()
    
    # Sort by billing amount
    dispatcher_stats = dispatcher_stats.sort_values(broker_rate_col, ascending=True)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add billing bars (green)
    fig.add_trace(go.Bar(
        x=dispatcher_stats[broker_rate_col],
        y=dispatcher_stats[dispatcher_col],
        orientation='h',
        name='Total Billing',
        marker_color='lightgreen',
        text=dispatcher_stats[broker_rate_col].apply(lambda x: f'${x:,.2f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Total Billing: $%{x:,.2f}<extra></extra>'
    ))
    
    # Add BD margin bars (red) superimposed
    fig.add_trace(go.Bar(
        x=dispatcher_stats[bd_margin_col],
        y=dispatcher_stats[dispatcher_col],
        orientation='h',
        name='BD Margin',
        marker_color='red',
        opacity=0.7,
        text=dispatcher_stats[bd_margin_col].apply(lambda x: f'${x:,.2f}'),
        textposition='inside',
        textfont=dict(color='white', size=10),
        hovertemplate='<b>%{y}</b><br>BD Margin: $%{x:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='overlay',
        title='Total Billing per Dispatcher',
        xaxis_title='Amount ($)',
        yaxis_title='Dispatcher',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def plot_bd_margin_performance(df):
    """3. BD Margin per Dispatcher - Scatter Plot (X: Total Billing, Y: BD Margin) + Bubble Size = Load Count"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find dispatcher, broker rate, and BD margin columns
    dispatcher_col = find_dispatcher_column(df)
    broker_rate_col = find_broker_rate_column(df)
    bd_margin_col = find_bd_margin_column(df)
    
    if not dispatcher_col:
        return go.Figure().add_annotation(text="Dispatcher column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not bd_margin_col:
        return go.Figure().add_annotation(text="BD margin column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by dispatcher - use size() instead of counting specific column
    dispatcher_stats = df.groupby(dispatcher_col).agg({
        broker_rate_col: 'sum',
        bd_margin_col: 'sum'
    }).reset_index()
    
    # Add load count using size
    load_counts = df.groupby(dispatcher_col).size().reset_index(name='Load Count')
    dispatcher_stats = dispatcher_stats.merge(load_counts, on=dispatcher_col)
    
    fig = px.scatter(
        dispatcher_stats,
        x=broker_rate_col,
        y=bd_margin_col,
        size='Load Count',
        hover_name=dispatcher_col,
        title='Dispatcher Performance: Billing vs Margin vs Load Count',
        labels={broker_rate_col: 'Total Billing ($)', bd_margin_col: 'BD Margin ($)', 'Load Count': 'Number of Loads'},
        color='Load Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Total Billing ($)",
        yaxis_title="BD Margin ($)"
    )
    
    return fig

def plot_loads_per_dispatcher_by_status(df):
    """4. Total Number of Loads per Dispatcher (según status) - Stacked Bar Chart"""
    if df.empty or 'LOAD STATUS' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find dispatcher column
    dispatcher_col = find_dispatcher_column(df)
    if not dispatcher_col:
        return go.Figure().add_annotation(text="Dispatcher column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Create pivot table
    status_pivot = df.groupby([dispatcher_col, 'LOAD STATUS']).size().unstack(fill_value=0)
    
    # Get all unique load statuses from the data
    all_statuses = df['LOAD STATUS'].dropna().unique()
    
    # Create a comprehensive color map that covers all statuses
    # Start with predefined colors for known statuses
    predefined_colors = {
        'Booked': '#2E8B57',      # Green
        'Delivered': '#4169E1',   # Blue
        'On dispute': '#FFD700',  # Gold
        'TONU Received': '#FF6347', # Red
        'Cancelled': '#DC143C',   # Crimson
        'Disputing a TONU': '#FF4500', # Orange Red
        'In Transit': '#32CD32',  # Lime Green
        'Pending': '#FFA500',     # Orange
        'Completed': '#008000',   # Dark Green
        'Failed': '#8B0000',      # Dark Red
        'Rejected': '#FF0000',    # Red
        'Approved': '#00FF00',    # Bright Green
        'Under Review': '#FFFF00', # Yellow
        'On Hold': '#FF8C00',     # Dark Orange
        'Rescheduled': '#9370DB', # Medium Purple
        'No Show': '#FF1493',     # Deep Pink
        'Late': '#FFD700',        # Gold
        'Early': '#00CED1',       # Dark Turquoise
        'Partial': '#FF69B4',     # Hot Pink
        'Full': '#00FA9A'         # Medium Spring Green
    }
    
    # Generate additional colors for any statuses not in predefined list
    import colorsys
    import random
    
    # Set random seed for consistent colors
    random.seed(42)
    
    def generate_color():
        """Generate a random but visually distinct color"""
        hue = random.random()
        saturation = 0.7 + random.random() * 0.3  # 0.7-1.0
        value = 0.8 + random.random() * 0.2       # 0.8-1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
    
    # Create final color map including all statuses
    color_map = {}
    for status in all_statuses:
        if status in predefined_colors:
            color_map[status] = predefined_colors[status]
        else:
            color_map[status] = generate_color()
    
    fig = px.bar(
        status_pivot,
        title='Load Status Distribution per Dispatcher',
        labels={'value': 'Number of Loads'},
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Dispatcher",
        yaxis_title="Number of Loads",
        barmode='stack'
    )
    
    return fig

def plot_top_drivers_earnings(df):
    """5. Top 20 Drivers by Driver Earnings - Vertical Bar Chart (top-down)"""
    if df.empty or 'DRIVER NAME' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find driver rate column
    driver_rate_col = find_driver_rate_column(df)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Get top 20 drivers
    top_drivers = df.groupby('DRIVER NAME')[driver_rate_col].sum().nlargest(20).reset_index()
    top_drivers = top_drivers.sort_values(driver_rate_col, ascending=False)
    
    # Calculate percentage of total
    total_earnings = df[driver_rate_col].sum()
    top_drivers['Percentage'] = (top_drivers[driver_rate_col] / total_earnings * 100).round(1)
    
    fig = px.bar(
        top_drivers,
        x='DRIVER NAME',
        y=driver_rate_col,
        title='Top 20 Drivers by Earnings',
        labels={driver_rate_col: 'Total Earnings ($)', 'DRIVER NAME': 'Driver'},
        text=driver_rate_col,
        color=driver_rate_col,
        color_continuous_scale='Reds'
    )
    
    fig.update_traces(texttemplate='$%{text:,.0f}<br>(%{customdata:.1f}%)', textposition='outside')
    fig.update_layout(
        height=500,
        xaxis_title="Driver",
        yaxis_title="Total Earnings ($)",
        xaxis_tickangle=-45
    )
    
    # Add custom data for percentage
    fig.data[0].customdata = top_drivers['Percentage'].values
    
    return fig

def plot_billing_per_trailer_type(df):
    """6. Total Billing per Trailer Type - Donut Chart"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find trailer and broker rate columns
    trailer_col = find_trailer_column(df)
    broker_rate_col = find_broker_rate_column(df)
    if not trailer_col:
        return go.Figure().add_annotation(text="Trailer column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by trailer type
    trailer_billing = df.groupby(trailer_col)[broker_rate_col].sum().reset_index()
    
    fig = px.pie(
        trailer_billing,
        values=broker_rate_col,
        names=trailer_col,
        title='Total Billing by Trailer Type',
        hole=0.4  # Creates donut chart
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_average_driver_earnings_weekly(df):
    """Create a line chart showing average driver earnings per week."""
    if 'DELIVERY DATE' not in df.columns:
        return go.Figure().add_annotation(text="Required columns not found", x=0.5, y=0.5, showarrow=False)
    
    # Find driver rate column
    driver_rate_col = find_driver_rate_column(df)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", x=0.5, y=0.5, showarrow=False)
    
    # Convert DELIVERY DATE to datetime
    df_copy = df.copy()
    df_copy['DELIVERY DATE'] = pd.to_datetime(df_copy['DELIVERY DATE'], errors='coerce')
    
    # Filter out rows with invalid dates
    df_copy = df_copy[df_copy['DELIVERY DATE'].notna()].copy()
    
    if len(df_copy) == 0:
        return go.Figure().add_annotation(text="No valid delivery dates found", x=0.5, y=0.5, showarrow=False)
    
    # Convert date column if it exists
    if 'DELIVERY DATE' in df.columns:
        df_copy['Week'] = df_copy['DELIVERY DATE'].dt.to_period('W')
        
        # Calculate average earnings per week
        weekly_earnings = df_copy.groupby('Week')[driver_rate_col].mean().reset_index()
        weekly_earnings['Week'] = weekly_earnings['Week'].astype(str)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weekly_earnings['Week'],
            y=weekly_earnings[driver_rate_col],
            mode='lines+markers',
            name='Average Weekly Earnings',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Week: %{x}<br>Avg Earnings: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add target lines
        fig.add_hline(y=6000, line_dash="dash", line_color="red", 
                      annotation_text="Flatbed/Stepdeck Target ($6,000)")
        fig.add_hline(y=5500, line_dash="dash", line_color="orange", 
                      annotation_text="Other Trailers Target ($5,500)")
        
        fig.update_layout(
            title="Average Driver Earnings per Week (Based on Delivery Dates)",
            xaxis_title="Week",
            yaxis_title="Average Weekly Earnings ($)",
            height=500,
            showlegend=True
        )
        
        return fig
    else:
        return go.Figure().add_annotation(text="No date data available", x=0.5, y=0.5, showarrow=False)

def plot_total_miles_per_carrier(df):
    """8. Total Miles per Carrier - Bar Chart (Vertical)"""
    if df.empty or 'LOAD\'S CARRIER COMPANY' not in df.columns or 'FULL MILES TOTAL' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by carrier and sum miles
    carrier_miles = df.groupby('LOAD\'S CARRIER COMPANY')['FULL MILES TOTAL'].sum().reset_index()
    carrier_miles = carrier_miles.sort_values('FULL MILES TOTAL', ascending=False)
    
    fig = px.bar(
        carrier_miles,
        x='LOAD\'S CARRIER COMPANY',
        y='FULL MILES TOTAL',
        title='Total Miles per Carrier',
        labels={'FULL MILES TOTAL': 'Total Miles', 'LOAD\'S CARRIER COMPANY': 'Carrier'},
        color='FULL MILES TOTAL',
        color_continuous_scale='Purples',
        text=carrier_miles['FULL MILES TOTAL'].apply(lambda x: f'{x:,.2f}')
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Total Miles: %{y:,.2f}<extra></extra>',
        textposition='outside'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Carrier",
        yaxis_title="Total Miles",
        xaxis_tickangle=-45
    )
    
    return fig

def plot_driver_miles_heatmap(df):
    """Create a heatmap showing driver miles by week (Tuesday to Monday, based on DELIVERY DATE)."""
    if 'DELIVERY DATE' not in df.columns or 'DRIVER NAME' not in df.columns or 'FULL MILES TOTAL' not in df.columns:
        return go.Figure().add_annotation(text="Required columns not found", x=0.5, y=0.5, showarrow=False)
    
    # Convert DELIVERY DATE to datetime
    df_copy = df.copy()
    df_copy['DELIVERY DATE'] = pd.to_datetime(df_copy['DELIVERY DATE'], errors='coerce')
    df_copy = df_copy[df_copy['DELIVERY DATE'].notna()].copy()
    if len(df_copy) == 0:
        return go.Figure().add_annotation(text="No valid delivery dates found", x=0.5, y=0.5, showarrow=False)
    
    # Assign week start (Tuesday) for each delivery date
    df_copy['WEEK_START'] = df_copy['DELIVERY DATE'].apply(get_week_start)
    
    # Get top 20 drivers by total miles
    top_drivers = df_copy.groupby('DRIVER NAME')['FULL MILES TOTAL'].sum().nlargest(20).index
    df_top_drivers = df_copy[df_copy['DRIVER NAME'].isin(top_drivers)]
    
    # Create pivot table for heatmap
    heatmap_data = df_top_drivers.groupby(['DRIVER NAME', 'WEEK_START'])['FULL MILES TOTAL'].sum().unstack(fill_value=0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(x.date()) for x in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='Viridis',
        hovertemplate='Driver: %{y}<br>Week: %{x}<br>Miles: %{z:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title="Driver Miles Heatmap (Top 20 Drivers by Total Miles, Tuesday-Monday Weeks)",
        xaxis_title="Week Starting (Tuesday)",
        yaxis_title="Driver Name",
        height=600,
        yaxis=dict(autorange='reversed')
    )
    return fig

def plot_bd_margin_distribution(df):
    """10. BD Margin % Distribution - Histogram + Line Chart (promedio móvil)"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find broker rate and BD margin columns
    broker_rate_col = find_broker_rate_column(df)
    bd_margin_col = find_bd_margin_column(df)
    
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not bd_margin_col:
        return go.Figure().add_annotation(text="BD margin column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Calculate margin percentage
    df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df_copy['Margin_Percentage'] = (df_copy[bd_margin_col] / df_copy[broker_rate_col] * 100).fillna(0)
    
    # Remove outliers (above 100% or below -50%)
    margin_data = df_copy[(df_copy['Margin_Percentage'] <= 100) & (df_copy['Margin_Percentage'] >= -50)]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('BD Margin % Distribution', 'BD Margin % Over Time'),
        vertical_spacing=0.1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=margin_data['Margin_Percentage'], nbinsx=30, name='Margin Distribution'),
        row=1, col=1
    )
    
    # Add mean line
    mean_margin = margin_data['Margin_Percentage'].mean()
    fig.add_hline(y=mean_margin, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_margin:.1f}%", row=1, col=1)
    
    # Time series if date available
    if 'PICK UP DATE' in df.columns:
        df_copy['PICK UP DATE'] = pd.to_datetime(df_copy['PICK UP DATE'], errors='coerce')
        time_series = df_copy.groupby(df_copy['PICK UP DATE'].dt.date)['Margin_Percentage'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(x=time_series['PICK UP DATE'], y=time_series['Margin_Percentage'], 
                      mode='lines+markers', name='Daily Average'),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        title_text="BD Margin Analysis",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Margin %", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Margin %", row=2, col=1)
    
    return fig

def plot_carrier_performance_analysis(df):
    """11. Combined Carrier Performance: Billing, Miles, and Revenue per Mile"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if df.empty or 'LOAD\'S CARRIER COMPANY' not in df.columns or 'FULL MILES TOTAL' not in df.columns:
        empty_fig = go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
    
    # Find broker rate column
    broker_rate_col = find_broker_rate_column(df)
    if not broker_rate_col:
        empty_fig = go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
    
    # Group by carrier and calculate metrics
    carrier_stats = df.groupby('LOAD\'S CARRIER COMPANY').agg({
        broker_rate_col: 'sum',
        'FULL MILES TOTAL': 'sum'
    }).reset_index()
    
    # Calculate revenue per mile
    carrier_stats['Revenue_Per_Mile'] = (carrier_stats[broker_rate_col] / carrier_stats['FULL MILES TOTAL']).fillna(0)
    
    # Sort by total billing
    carrier_stats = carrier_stats.sort_values(broker_rate_col, ascending=False)
    
    # Top row: Billing and Miles
    fig_top = make_subplots(rows=1, cols=2, subplot_titles=('Total Billing by Carrier', 'Total Miles by Carrier'))
    
    fig_top.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats[broker_rate_col],
            name='Total Billing',
            marker_color='#1f77b4',
            text=carrier_stats[broker_rate_col].apply(lambda x: f'${x:,.2f}'),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    fig_top.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats['FULL MILES TOTAL'],
            name='Total Miles',
            marker_color='#ff7f0e',
            text=carrier_stats['FULL MILES TOTAL'].apply(lambda x: f'{x:,.2f}'),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig_top.update_layout(
        height=400,
        title_text="Comprehensive Carrier Performance Analysis (Top)",
        showlegend=False,
        margin=dict(t=60)
    )
    
    fig_top.update_xaxes(title_text="Carrier", row=1, col=1, tickangle=-45)
    fig_top.update_yaxes(title_text="Total Billing ($)", row=1, col=1)
    fig_top.update_xaxes(title_text="Carrier", row=1, col=2, tickangle=-45)
    fig_top.update_yaxes(title_text="Total Miles", row=1, col=2)
    
    # Bottom row: Revenue per Mile and Efficiency Scatter
    fig_bottom = make_subplots(rows=1, cols=2, subplot_titles=('Revenue per Mile by Carrier', 'Carrier Efficiency Scatter'))
    
    fig_bottom.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats['Revenue_Per_Mile'],
            name='Revenue/Mile',
            marker_color='#2ca02c',
            text=carrier_stats['Revenue_Per_Mile'].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    fig_bottom.add_trace(
        go.Scatter(
            x=carrier_stats['FULL MILES TOTAL'],
            y=carrier_stats['Revenue_Per_Mile'],
            mode='markers+text',
            name='Efficiency',
            marker=dict(
                size=carrier_stats[broker_rate_col] / carrier_stats[broker_rate_col].max() * 20 + 10,
                color=carrier_stats['Revenue_Per_Mile'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue/Mile ($)")
            ),
            text=carrier_stats['LOAD\'S CARRIER COMPANY'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>Miles: %{x:,.2f}<br>Revenue/Mile: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig_bottom.update_layout(
        height=400,
        title_text="Comprehensive Carrier Performance Analysis (Bottom)",
        showlegend=False,
        margin=dict(t=60)
    )
    
    fig_bottom.update_xaxes(title_text="Carrier", row=1, col=1, tickangle=-45)
    fig_bottom.update_yaxes(title_text="Revenue per Mile ($)", row=1, col=1)
    fig_bottom.update_xaxes(title_text="Total Miles", row=1, col=2)
    fig_bottom.update_yaxes(title_text="Revenue per Mile ($)", row=1, col=2)
    
    return fig_top, fig_bottom

def plot_driver_income_analysis(weekly_earnings):
    """
    Create comprehensive driver income analysis with weekly targets and percentage to target.
    """
    if weekly_earnings.empty:
        return go.Figure().add_annotation(text="No weekly earnings data available", x=0.5, y=0.5, showarrow=False)
    
    # Find the actual column names in the data
    driver_rate_col = find_driver_rate_column(weekly_earnings)
    trailer_col = find_trailer_column(weekly_earnings)
    
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", x=0.5, y=0.5, showarrow=False)
    if not trailer_col:
        return go.Figure().add_annotation(text="Trailer column not found", x=0.5, y=0.5, showarrow=False)
    
    # Check if required columns exist
    required_columns = ['DRIVER NAME', 'WEEK_START', 'TARGET', 'PERCENTAGE_TO_TARGET']
    missing_columns = [col for col in required_columns if col not in weekly_earnings.columns]
    
    if missing_columns:
        return go.Figure().add_annotation(
            text=f"Missing columns: {', '.join(missing_columns)}", 
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Weekly Earnings vs Target by Trailer Type',
            'Percentage to Target Distribution',
            'Top Drivers by Weekly Earnings',
            'Target Achievement by Trailer Type'
        ),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Weekly Earnings vs Target Scatter Plot
    flatbed_data = weekly_earnings[weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])]
    other_data = weekly_earnings[~weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])]
    
    fig.add_trace(
        go.Scatter(
            x=flatbed_data['WEEK_START'],
            y=flatbed_data[driver_rate_col],
            mode='markers',
            name='Flatbed/Stepdeck',
            marker=dict(color='red', size=8),
            hovertemplate='<b>%{text}</b><br>Week: %{x}<br>Earnings: $%{y:,.0f}<br>Target: $6,000<extra></extra>',
            text=flatbed_data['DRIVER NAME']
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=other_data['WEEK_START'],
            y=other_data[driver_rate_col],
            mode='markers',
            name='Other Trailers',
            marker=dict(color='blue', size=8),
            hovertemplate='<b>%{text}</b><br>Week: %{x}<br>Earnings: $%{y:,.0f}<br>Target: $5,500<extra></extra>',
            text=other_data['DRIVER NAME']
        ),
        row=1, col=1
    )
    
    # Add target lines
    fig.add_hline(y=6000, line_dash="dash", line_color="red", 
                  annotation_text="Flatbed/Stepdeck Target ($6,000)", row=1, col=1)
    fig.add_hline(y=5500, line_dash="dash", line_color="blue", 
                  annotation_text="Other Trailers Target ($5,500)", row=1, col=1)
    
    # 2. Percentage to Target Distribution
    fig.add_trace(
        go.Histogram(
            x=weekly_earnings['PERCENTAGE_TO_TARGET'],
            nbinsx=20,
            name='Percentage to Target',
            marker_color='lightgreen',
            hovertemplate='Percentage: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_vline(x=100, line_dash="dash", line_color="red", 
                  annotation_text="100% Target", row=1, col=2)
    
    # 3. Top Drivers by Weekly Earnings
    top_drivers = weekly_earnings.groupby('DRIVER NAME')[driver_rate_col].mean().nlargest(15)
    
    fig.add_trace(
        go.Bar(
            x=top_drivers.values,
            y=top_drivers.index,
            orientation='h',
            name='Avg Weekly Earnings',
            marker_color='orange',
            hovertemplate='<b>%{y}</b><br>Avg Weekly: $%{x:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Target Achievement by Trailer Type
    trailer_performance = weekly_earnings.groupby('trailer_col').agg({
        'PERCENTAGE_TO_TARGET': 'mean',
        driver_rate_col: 'count'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=trailer_performance['trailer_col'],
            y=trailer_performance['PERCENTAGE_TO_TARGET'],
            name='Avg % to Target',
            marker_color='purple',
            hovertemplate='<b>%{x}</b><br>Avg % to Target: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_hline(y=100, line_dash="dash", line_color="red", 
                  annotation_text="100% Target", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title="Driver Income Analysis: Weekly Earnings vs Targets",
        height=800,
        showlegend=True,
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="Week Starting", row=1, col=1)
    fig.update_yaxes(title_text="Weekly Earnings ($)", row=1, col=1)
    
    fig.update_xaxes(title_text="Percentage to Target (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_xaxes(title_text="Average Weekly Earnings ($)", row=2, col=1)
    fig.update_yaxes(title_text="Driver Name", row=2, col=1)
    
    fig.update_xaxes(title_text="Trailer Type", row=2, col=2)
    fig.update_yaxes(title_text="Average % to Target", row=2, col=2)
    
    return fig

def generate_driver_performance_summary(weekly_earnings, metric_type="earnings", flatbed_rpm_target=2.0, dryvan_rpm_target=1.8):
    """Generate driver performance summary for display in charts."""
    targets = {
        'Flatbed/Stepdeck': 6000,
        'Dry Van/Reefer/Power Only': 5500
    }
    
    trailer_groups = [t for t in targets.keys() if t in weekly_earnings['TRAILER GROUP'].unique()]
    summary_text = ""
    
    for group in trailer_groups:
        sub = weekly_earnings[weekly_earnings['TRAILER GROUP'] == group].copy()
        total_drivers = sub['DRIVER NAME'].nunique()
        
        if metric_type == "earnings":
            overachievers = sub[sub['PERCENTAGE_TO_TARGET'] > 110]['DRIVER NAME'].nunique()
            on_target = sub[(sub['PERCENTAGE_TO_TARGET'] >= 100) & (sub['PERCENTAGE_TO_TARGET'] <= 110)]['DRIVER NAME'].nunique()
            watchlist = sub[(sub['PERCENTAGE_TO_TARGET'] >= 80) & (sub['PERCENTAGE_TO_TARGET'] < 100)]['DRIVER NAME'].nunique()
            underperformers = sub[sub['PERCENTAGE_TO_TARGET'] < 80]['DRIVER NAME'].nunique()
        elif metric_type == "miles":
            miles_target = 3000
            sub['MILES_PERCENTAGE'] = (sub['FULL MILES TOTAL'] / miles_target * 100).round(1)
            overachievers = sub[sub['MILES_PERCENTAGE'] > 110]['DRIVER NAME'].nunique()
            on_target = sub[(sub['MILES_PERCENTAGE'] >= 100) & (sub['MILES_PERCENTAGE'] <= 110)]['DRIVER NAME'].nunique()
            watchlist = sub[(sub['MILES_PERCENTAGE'] >= 80) & (sub['MILES_PERCENTAGE'] < 100)]['DRIVER NAME'].nunique()
            underperformers = sub[sub['MILES_PERCENTAGE'] < 80]['DRIVER NAME'].nunique()
        elif metric_type == "revenue_per_mile":
            revenue_targets = {
                'Flatbed/Stepdeck': flatbed_rpm_target,
                'Dry Van/Reefer/Power Only': dryvan_rpm_target
            }
            target = revenue_targets.get(group, dryvan_rpm_target)
            # Find the actual driver rate column in the data
            driver_rate_col = find_driver_rate_column(sub)
            if not driver_rate_col:
                continue
            sub['REVENUE_PER_MILE'] = np.where(sub['FULL MILES TOTAL'] > 0, 
                                             sub[driver_rate_col] / sub['FULL MILES TOTAL'], 0)
            sub['REVENUE_PERCENTAGE'] = (sub['REVENUE_PER_MILE'] / target * 100).round(1)
            overachievers = sub[sub['REVENUE_PERCENTAGE'] > 110]['DRIVER NAME'].nunique()
            on_target = sub[(sub['REVENUE_PERCENTAGE'] >= 100) & (sub['REVENUE_PERCENTAGE'] <= 110)]['DRIVER NAME'].nunique()
            watchlist = sub[(sub['REVENUE_PERCENTAGE'] >= 80) & (sub['REVENUE_PERCENTAGE'] < 100)]['DRIVER NAME'].nunique()
            underperformers = sub[sub['REVENUE_PERCENTAGE'] < 80]['DRIVER NAME'].nunique()
        
        summary_text += f"\n{group}: Total: {total_drivers} drivers\n"
        summary_text += f"💚 Overachievers: {overachievers} ({overachievers/total_drivers*100:.1f}%)\n"
        summary_text += f"🟢 On Target: {on_target} ({on_target/total_drivers*100:.1f}%)\n"
        summary_text += f"🟡 Watchlist: {watchlist} ({watchlist/total_drivers*100:.1f}%)\n"
        summary_text += f"🔴 Underperformers: {underperformers} ({underperformers/total_drivers*100:.1f}%)\n"
    
    return summary_text

def plot_weekly_driver_earnings_vs_target_faceted(weekly_earnings):
    """Faceted horizontal bar chart: one facet per trailer group, y=driver, x=earnings, with color gradient based on target achievement. Fixes y-axis alignment."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Find the actual driver rate column in the data
    driver_rate_col = find_driver_rate_column(weekly_earnings)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    targets = {
        'Flatbed/Stepdeck': 6000,
        'Dry Van/Reefer/Power Only': 5500
    }
    
    # Clean up driver names: drop NaN, convert to string, remove empty
    df = weekly_earnings.copy()
    df = df[df['DRIVER NAME'].notna() & (df['DRIVER NAME'].astype(str).str.strip() != '')].copy()
    df['DRIVER NAME'] = df['DRIVER NAME'].astype(str)
    
    trailer_groups = [t for t in targets.keys() if t in df['TRAILER GROUP'].unique()]
    n = len(trailer_groups)
    if n == 0:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig = make_subplots(rows=1, cols=n, subplot_titles=trailer_groups, shared_yaxes=False)
    for i, group in enumerate(trailer_groups):
        sub = df[df['TRAILER GROUP'] == group].copy()
        def get_color_category(percentage):
            if percentage > 110:
                return '💚 Overachievers (>110%)'
            elif percentage >= 100:
                return '🟢 On Target (100–110%)'
            elif percentage >= 80:
                return '🟡 Watchlist (80–99%)'
            else:
                return '🔴 Underperformers (<80%)'
        sub['Color_Category'] = sub['PERCENTAGE_TO_TARGET'].apply(get_color_category)
        color_map = {
            '💚 Overachievers (>110%)': '#00FF00',
            '🟢 On Target (100–110%)': '#90EE90',
            '🟡 Watchlist (80–99%)': '#FFFF00',
            '🔴 Underperformers (<80%)': '#FF0000'
        }
        # Sort by earnings for consistent y-axis - INVERTED ORDER
        sub = sub.sort_values(driver_rate_col, ascending=True)
        y_order = sub['DRIVER NAME'].tolist()
        for category in color_map.keys():
            category_data = sub[sub['Color_Category'] == category]
            if not category_data.empty:
                fig.add_trace(
                    go.Bar(
                        y=category_data['DRIVER NAME'],
                        x=category_data[driver_rate_col],
                        orientation='h',
                        name=category,
                        marker_color=color_map[category],
                        showlegend=(i == 0),
                        hovertemplate='<b>%{y}</b><br>Earnings: $%{x:,.0f}<br>% to Target: ' + \
                                    category_data['PERCENTAGE_TO_TARGET'].astype(str) + '%<extra></extra>'
                    ),
                    row=1, col=i+1
                )
        # Set y-axis order explicitly - INVERTED
        fig.update_yaxes(categoryorder='array', categoryarray=y_order, row=1, col=i+1)
        fig.add_vline(
            x=targets[group],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target (${targets[group]:,})",
            row=1, col=i+1
        )
        fig.update_xaxes(title_text="Weekly Earnings ($)", row=1, col=i+1)
    
    fig.update_layout(
        title="Weekly Driver Earnings vs Target by Trailer Type",
        height=400 + 30 * max(5, df['DRIVER NAME'].nunique()),
        margin=dict(t=80, l=200),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_top_drivers_by_weekly_earnings_improved(weekly_earnings):
    """Horizontal bar chart of top drivers by weekly earnings (sorted, spaced)."""
    import plotly.graph_objects as go
    
    # Find the actual driver rate column in the data
    driver_rate_col = find_driver_rate_column(weekly_earnings)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    top_df = weekly_earnings.groupby('DRIVER NAME')[driver_rate_col].mean().sort_values(ascending=False).reset_index()
    # Invert the order so top earners appear at the top
    top_df = top_df.sort_values(driver_rate_col, ascending=True)
    
    fig = go.Figure(go.Bar(
        y=top_df['DRIVER NAME'],
        x=top_df[driver_rate_col],
        orientation='h',
        marker_color='orange',
        hovertemplate='<b>%{y}</b><br>Avg Weekly: $%{x:,.0f}<extra></extra>'
    ))
    
    # Add vertical dotted line at $5,500 target
    fig.add_vline(
        x=5500,
        line_dash="dot",
        line_color="red",
        annotation_text="Target ($5,500)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Top Drivers by Weekly Earnings",
        xaxis_title="Average Weekly Earnings ($)",
        yaxis_title="Driver Name",
        height=400 + 30 * min(15, len(top_df)),
        margin=dict(l=120, t=60)
    )
    return fig

def plot_target_achievement_by_trailer_type_improved(weekly_earnings):
    """Bar chart: average % to target by trailer group, 100% line, with rate per mile text on bars."""
    import plotly.graph_objects as go
    
    # Find the actual driver rate column in the data
    driver_rate_col = find_driver_rate_column(weekly_earnings)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    targets = {
        'Flatbed/Stepdeck': 6000,
        'Dry Van/Reefer/Power Only': 5500
    }
    achievement = []
    for group, target in targets.items():
        sub = weekly_earnings[weekly_earnings['TRAILER GROUP'] == group]
        if not sub.empty:
            avg_pct = (sub[driver_rate_col].mean() / target) * 100
            # Calculate rate per mile for this trailer group
            # We need to find the miles column in the original data
            miles_col = None
            for col in sub.columns:
                if 'miles' in col.lower() or 'mile' in col.lower():
                    miles_col = col
                    break
            
            if miles_col and miles_col in sub.columns:
                # Calculate total miles and total driver rate for this group
                total_miles = sub[miles_col].sum()
                total_driver_rate = sub[driver_rate_col].sum()
                if total_miles > 0:
                    rate_per_mile = total_driver_rate / total_miles
                else:
                    rate_per_mile = 0
            else:
                rate_per_mile = 0
            
            achievement.append({
                'Trailer Group': group, 
                'Avg % to Target': avg_pct,
                'Rate per Mile': rate_per_mile
            })
    
    achievement_df = pd.DataFrame(achievement)
    
    # Create the bar chart
    fig = go.Figure(go.Bar(
        x=achievement_df['Trailer Group'],
        y=achievement_df['Avg % to Target'],
        marker_color='purple',
        hovertemplate='<b>%{x}</b><br>Avg % to Target: %{y:.1f}%<br>Rate per Mile: $' + 
                     achievement_df['Rate per Mile'].round(2).astype(str) + '<extra></extra>'
    ))
    
    # Add text annotations on top of bars showing rate per mile
    for i, row in achievement_df.iterrows():
        fig.add_annotation(
            x=row['Trailer Group'],
            y=row['Avg % to Target'],
            text=f"${row['Rate per Mile']:.2f}/mile",
            showarrow=False,
            yshift=10,
            font=dict(size=12, color='white')
        )
    
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Target")
    fig.update_layout(
        title="Target Achievement by Trailer Group",
        yaxis_title="Average % to Target",
        xaxis_title="Trailer Group",
        height=400,
        margin=dict(t=60)
    )
    return fig

def plot_weekly_driver_miles_vs_target_faceted(weekly_earnings, miles_col='FULL MILES TOTAL'):
    """Faceted horizontal bar chart: one facet per trailer group, y=driver, x=total miles, with color gradient based on miles target achievement."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    
    if miles_col not in weekly_earnings.columns:
        return go.Figure().add_annotation(text="No miles data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    miles_target = 3000
    df = weekly_earnings.copy()
    df = df[df['DRIVER NAME'].notna() & (df['DRIVER NAME'].astype(str).str.strip() != '')].copy()
    df['DRIVER NAME'] = df['DRIVER NAME'].astype(str)
    
    def parse_number(value):
        import re
        import pandas as pd
        if pd.isna(value) or value == '':
            return 0.0
        value_str = str(value).strip()
        value_str = re.sub(r'[^\d.,]', '', value_str)
        if ',' in value_str:
            value_str = value_str.replace(',', '')
        try:
            return float(value_str)
        except ValueError:
            return 0.0
    df[miles_col] = df[miles_col].apply(parse_number)
    miles_df = df.groupby(['DRIVER NAME', 'WEEK_START', 'TRAILER GROUP'])[miles_col].sum().reset_index()
    miles_df['PERCENTAGE_TO_TARGET'] = (miles_df[miles_col] / miles_target * 100).round(1)
    trailer_groups = miles_df['TRAILER GROUP'].dropna().unique().tolist()
    n = len(trailer_groups)
    if n == 0:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig = make_subplots(rows=1, cols=n, subplot_titles=trailer_groups, shared_yaxes=False)
    for i, group in enumerate(trailer_groups):
        sub = miles_df[miles_df['TRAILER GROUP'] == group].copy()
        def get_color_category(percentage):
            if percentage > 110:
                return '💚 Overachievers (>110%)'
            elif percentage >= 100:
                return '🟢 On Target (100–110%)'
            elif percentage >= 80:
                return '🟡 Watchlist (80–99%)'
            else:
                return '🔴 Underperformers (<80%)'
        sub['Color_Category'] = sub['PERCENTAGE_TO_TARGET'].apply(get_color_category)
        color_map = {
            '💚 Overachievers (>110%)': '#00FF00',
            '🟢 On Target (100–110%)': '#90EE90',
            '🟡 Watchlist (80–99%)': '#FFFF00',
            '🔴 Underperformers (<80%)': '#FF0000'
        }
        # Sort by miles for consistent y-axis - INVERTED ORDER
        sub = sub.sort_values(miles_col, ascending=True)
        y_order = sub['DRIVER NAME'].tolist()
        for category in color_map.keys():
            category_data = sub[sub['Color_Category'] == category]
            if not category_data.empty:
                fig.add_trace(
                    go.Bar(
                        y=category_data['DRIVER NAME'],
                        x=category_data[miles_col],
                        orientation='h',
                        name=category,
                        marker_color=color_map[category],
                        showlegend=(i == 0),
                        hovertemplate='<b>%{y}</b><br>Miles: %{x:,.0f}<br>% to Target: ' + \
                                    category_data['PERCENTAGE_TO_TARGET'].astype(str) + '%<extra></extra>'
                    ),
                    row=1, col=i+1
                )
        # Set y-axis order explicitly - INVERTED
        fig.update_yaxes(categoryorder='array', categoryarray=y_order, row=1, col=i+1)
        fig.add_vline(
            x=miles_target,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Target ({miles_target:,})",
            row=1, col=i+1
        )
        fig.update_xaxes(title_text="Weekly Miles", row=1, col=i+1)
    
    fig.update_layout(
        title="Weekly Driver Miles vs Target by Trailer Type",
        height=400 + 30 * max(5, miles_df['DRIVER NAME'].nunique()),
        margin=dict(t=80, l=200),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_revenue_per_mile_per_dispatcher(df):
    """Bar chart showing revenue per mile for each dispatcher."""
    import plotly.graph_objects as go
    import pandas as pd
    
    # Find the miles column
    miles_col = None
    for col in df.columns:
        if 'miles' in col.lower() or 'mile' in col.lower():
            miles_col = col
            break
    
    if not miles_col:
        return go.Figure().add_annotation(text="No miles data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find dispatcher and broker rate columns
    dispatcher_col = find_dispatcher_column(df)
    broker_rate_col = find_broker_rate_column(df)
    
    if not dispatcher_col:
        return go.Figure().add_annotation(text="Dispatcher column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not broker_rate_col:
        return go.Figure().add_annotation(text="Broker rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Calculate revenue per mile per dispatcher
    dispatcher_metrics = df.groupby(dispatcher_col).agg({
        broker_rate_col: 'sum',
        miles_col: 'sum'
    }).reset_index()
    
    # Calculate revenue per mile
    dispatcher_metrics['REVENUE_PER_MILE'] = (dispatcher_metrics[broker_rate_col] / dispatcher_metrics[miles_col]).round(2)
    
    # Remove any infinite or NaN values
    dispatcher_metrics = dispatcher_metrics[dispatcher_metrics['REVENUE_PER_MILE'].notna() & (dispatcher_metrics['REVENUE_PER_MILE'] != float('inf'))]
    
    if dispatcher_metrics.empty:
        return go.Figure().add_annotation(text="No valid revenue per mile data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Sort by revenue per mile
    dispatcher_metrics = dispatcher_metrics.sort_values('REVENUE_PER_MILE', ascending=False)
    
    # Create color gradient based on revenue per mile
    max_revenue = dispatcher_metrics['REVENUE_PER_MILE'].max()
    min_revenue = dispatcher_metrics['REVENUE_PER_MILE'].min()
    
    colors = []
    for revenue in dispatcher_metrics['REVENUE_PER_MILE']:
        if max_revenue == min_revenue:
            normalized = 0.5
        else:
            normalized = (revenue - min_revenue) / (max_revenue - min_revenue)
        # Green to red gradient
        colors.append(f'rgb({255 * (1 - normalized)}, {255 * normalized}, 0)')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dispatcher_metrics[dispatcher_col],
        y=dispatcher_metrics['REVENUE_PER_MILE'],
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Revenue per Mile: $%{y:.2f}<br>Total Revenue: $' + 
                     dispatcher_metrics[broker_rate_col].round(0).astype(str) + 
                     '<br>Total Miles: ' + dispatcher_metrics[miles_col].round(0).astype(str) + 
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Revenue per Mile by Dispatcher",
        xaxis_title="Dispatcher",
        yaxis_title="Revenue per Mile ($)",
        height=400,
        showlegend=False,
        margin=dict(t=80, l=80, r=80, b=80)
    )
    
    # Add a horizontal line for average
    avg_revenue = dispatcher_metrics['REVENUE_PER_MILE'].mean()
    fig.add_hline(
        y=avg_revenue,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Average: ${avg_revenue:.2f}",
        annotation_position="top right"
    )
    
    return fig

def plot_driver_revenue_per_mile_per_dispatcher(df):
    """Bar chart showing driver revenue per mile for each dispatcher."""
    import plotly.graph_objects as go
    import pandas as pd
    
    # Find the miles column
    miles_col = None
    for col in df.columns:
        if 'miles' in col.lower() or 'mile' in col.lower():
            miles_col = col
            break
    
    if not miles_col:
        return go.Figure().add_annotation(text="No miles data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Find dispatcher and driver rate columns
    dispatcher_col = find_dispatcher_column(df)
    driver_rate_col = find_driver_rate_column(df)
    
    if not dispatcher_col:
        return go.Figure().add_annotation(text="Dispatcher column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Calculate driver revenue per mile per dispatcher
    dispatcher_metrics = df.groupby(dispatcher_col).agg({
        driver_rate_col: 'sum',
        miles_col: 'sum'
    }).reset_index()
    
    # Calculate driver revenue per mile
    dispatcher_metrics['DRIVER_REVENUE_PER_MILE'] = (dispatcher_metrics[driver_rate_col] / dispatcher_metrics[miles_col]).round(2)
    
    # Remove any infinite or NaN values
    dispatcher_metrics = dispatcher_metrics[dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].notna() & (dispatcher_metrics['DRIVER_REVENUE_PER_MILE'] != float('inf'))]
    
    if dispatcher_metrics.empty:
        return go.Figure().add_annotation(text="No valid driver revenue per mile data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Sort by driver revenue per mile
    dispatcher_metrics = dispatcher_metrics.sort_values('DRIVER_REVENUE_PER_MILE', ascending=False)
    
    # Create color gradient based on driver revenue per mile
    max_revenue = dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].max()
    min_revenue = dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].min()
    
    colors = []
    for revenue in dispatcher_metrics['DRIVER_REVENUE_PER_MILE']:
        if max_revenue == min_revenue:
            normalized = 0.5
        else:
            normalized = (revenue - min_revenue) / (max_revenue - min_revenue)
        # Blue to purple gradient for driver revenue
        colors.append(f'rgb({100 + 155 * normalized}, {100 + 155 * normalized}, {255})')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dispatcher_metrics[dispatcher_col],
        y=dispatcher_metrics['DRIVER_REVENUE_PER_MILE'],
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Driver Revenue per Mile: $%{y:.2f}<br>Total Driver Pay: $' + 
                     dispatcher_metrics[driver_rate_col].round(0).astype(str) + 
                     '<br>Total Miles: ' + dispatcher_metrics[miles_col].round(0).astype(str) + 
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Driver Revenue per Mile by Dispatcher",
        xaxis_title="Dispatcher",
        yaxis_title="Driver Revenue per Mile ($)",
        height=400,
        showlegend=False,
        margin=dict(t=80, l=80, r=80, b=80)
    )
    
    # Add a horizontal line for average
    avg_revenue = dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].mean()
    fig.add_hline(
        y=avg_revenue,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Average: ${avg_revenue:.2f}",
        annotation_position="top right"
    )
    
    return fig

def generate_chart_analysis(chart_type, df, chart_data=None):
    """Generate written analysis and insights for each chart type"""
    
    if df.empty:
        return "No data available for analysis."
    
    # Find required columns
    broker_rate_col = find_broker_rate_column(df)
    driver_rate_col = find_driver_rate_column(df)
    trailer_col = find_trailer_column(df)
    
    analysis = ""
    
    if chart_type == "carrier_billing":
        # Analysis for Total Billing per Carrier
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        total_billing = df[broker_rate_col].sum()
        top_carrier = df.groupby('LOAD\'S CARRIER COMPANY')[broker_rate_col].sum().nlargest(1)
        top_carrier_name = top_carrier.index[0]
        top_carrier_value = top_carrier.values[0]
        top_carrier_pct = (top_carrier_value / total_billing * 100)
        
        analysis = f"""
        **📊 Total Billing per Carrier Analysis**
        
        **Key Findings:**
        - **Total Revenue**: ${total_billing:,.0f}
        - **Top Performer**: {top_carrier_name} (${top_carrier_value:,.0f}, {top_carrier_pct:.1f}% of total)
        - **Carrier Concentration**: {len(df["LOAD'S CARRIER COMPANY"].unique())} active carriers
        
        **Business Insights:**
        - {top_carrier_name} is your highest-revenue carrier, contributing {top_carrier_pct:.1f}% of total billing
        - Consider diversifying if any single carrier exceeds 30% of total revenue
        - Monitor carrier performance trends for strategic partnership decisions
        """
    
    elif chart_type == "dispatcher_billing":
        # Analysis for Total Billing per Dispatcher
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        
        # Find dispatcher column
        dispatcher_col = find_dispatcher_column(df)
        if not dispatcher_col:
            return "Dispatcher column not found for analysis."
        
        total_billing = df[broker_rate_col].sum()
        top_dispatcher = df.groupby(dispatcher_col)[broker_rate_col].sum().nlargest(1)
        top_dispatcher_name = top_dispatcher.index[0]
        top_dispatcher_value = top_dispatcher.values[0]
        top_dispatcher_pct = (top_dispatcher_value / total_billing * 100)
        
        avg_billing_per_dispatcher = total_billing / len(df[dispatcher_col].unique())
        
        analysis = f"""
        **📊 Total Billing per Dispatcher Analysis**
        
        **Key Findings:**
        - **Total Revenue**: ${total_billing:,.0f}
        - **Top Performer**: {top_dispatcher_name} (${top_dispatcher_value:,.0f}, {top_dispatcher_pct:.1f}% of total)
        - **Average per Dispatcher**: ${avg_billing_per_dispatcher:,.0f}
        - **Active Dispatchers**: {len(df[dispatcher_col].unique())}
        
        **Business Insights:**
        - {top_dispatcher_name} is your highest-billing dispatcher
        - Performance gap analysis: Top dispatcher is {top_dispatcher_value/avg_billing_per_dispatcher:.1f}x above average
        - Consider training programs for underperforming dispatchers
        - Monitor workload distribution for optimal team performance
        """
    
    elif chart_type == "dispatcher_performance":
        # Analysis for Dispatcher Performance Matrix
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        
        # Find dispatcher and BD margin columns
        dispatcher_col = find_dispatcher_column(df)
        bd_margin_col = find_bd_margin_column(df)
        if not dispatcher_col:
            return "Dispatcher column not found for analysis."
        if not bd_margin_col:
            return "BD margin column not found for analysis."
        
        dispatcher_stats = df.groupby(dispatcher_col).agg({
            broker_rate_col: 'sum',
            bd_margin_col: 'sum'
        }).reset_index()
        
        # Calculate efficiency metrics
        dispatcher_stats['Efficiency'] = (dispatcher_stats[bd_margin_col] / dispatcher_stats[broker_rate_col] * 100).fillna(0)
        
        top_efficiency = dispatcher_stats.loc[dispatcher_stats['Efficiency'].idxmax()]
        top_billing = dispatcher_stats.loc[dispatcher_stats[broker_rate_col].idxmax()]
        
        analysis = f"""
        **📊 Dispatcher Performance Matrix Analysis**
        
        **Key Findings:**
        - **Most Efficient**: {top_efficiency[dispatcher_col]} ({top_efficiency['Efficiency']:.1f}% margin)
        - **Highest Billing**: {top_billing[dispatcher_col]} (${top_billing[broker_rate_col]:,.0f})
        - **Performance Range**: {dispatcher_stats['Efficiency'].min():.1f}% to {dispatcher_stats['Efficiency'].max():.1f}% margin
        
        **Business Insights:**
        - {top_efficiency[dispatcher_col]} achieves the best margin efficiency
        - {top_billing[dispatcher_col]} generates the most revenue
        - Consider combining high-efficiency strategies with high-volume operations
        - Identify training opportunities for dispatchers below average efficiency
        """
    
    elif chart_type == "load_status":
        # Analysis for Load Status Distribution
        status_counts = df['LOAD STATUS'].value_counts()
        total_loads = len(df)
        delivered_pct = (status_counts.get('Delivered', 0) / total_loads * 100)
        booked_pct = (status_counts.get('Booked', 0) / total_loads * 100)
        dispute_pct = (status_counts.get('On dispute', 0) / total_loads * 100)
        
        analysis = f"""
        **📊 Load Status Distribution Analysis**
        
        **Key Findings:**
        - **Total Loads**: {total_loads:,}
        - **Delivered**: {status_counts.get('Delivered', 0):,} ({delivered_pct:.1f}%)
        - **Booked**: {status_counts.get('Booked', 0):,} ({booked_pct:.1f}%)
        - **On Dispute**: {status_counts.get('On dispute', 0):,} ({dispute_pct:.1f}%)
        
        **Business Insights:**
        - {delivered_pct:.1f}% completion rate indicates operational efficiency
        - {dispute_pct:.1f}% dispute rate may need attention
        - Monitor dispute resolution processes for improvement opportunities
        - Track status transitions to optimize load management
        """
    
    elif chart_type == "driver_earnings":
        # Analysis for Top Drivers by Earnings
        if not driver_rate_col:
            return "Driver rate column not found for analysis."
        driver_earnings = df.groupby('DRIVER NAME')[driver_rate_col].sum().sort_values(ascending=False)
        total_earnings = driver_earnings.sum()
        top_driver_earnings = driver_earnings.iloc[0]
        top_driver_name = driver_earnings.index[0]
        top_driver_pct = (top_driver_earnings / total_earnings * 100)
        
        # Calculate earnings distribution
        top_20_pct = (driver_earnings.head(20).sum() / total_earnings * 100)
        
        analysis = f"""
        **📊 Top Drivers by Earnings Analysis**
        
        **Key Findings:**
        - **Total Driver Pay**: ${total_earnings:,.0f}
        - **Top Earner**: {top_driver_name} (${top_driver_earnings:,.0f}, {top_driver_pct:.1f}% of total)
        - **Top 20 Drivers**: {top_20_pct:.1f}% of total earnings
        - **Active Drivers**: {len(driver_earnings)} drivers
        
        **Business Insights:**
        - {top_driver_name} is your highest-earning driver
        - Top 20 drivers account for {top_20_pct:.1f}% of total driver pay
        - Consider driver retention strategies for top performers
        - Monitor earnings distribution for fair compensation practices
        """
    
    elif chart_type == "trailer_billing":
        # Analysis for Billing by Trailer Type
        if not trailer_col or not broker_rate_col:
            return "Trailer or broker rate column not found for analysis."
        trailer_billing = df.groupby(trailer_col)[broker_rate_col].sum().sort_values(ascending=False)
        total_billing = trailer_billing.sum()
        top_trailer = trailer_billing.index[0]
        top_trailer_value = trailer_billing.iloc[0]
        top_trailer_pct = (top_trailer_value / total_billing * 100)
        
        analysis = f"""
        **📊 Billing by Trailer Type Analysis**
        
        **Key Findings:**
        - **Total Revenue**: ${total_billing:,.0f}
        - **Top Trailer Type**: {top_trailer} (${top_trailer_value:,.0f}, {top_trailer_pct:.1f}% of total)
        - **Trailer Types**: {len(trailer_billing)} different types
        - **Revenue Concentration**: {top_trailer} dominates with {top_trailer_pct:.1f}% share
        
        **Business Insights:**
        - {top_trailer} is your most profitable trailer type
        - Consider expanding {top_trailer} capacity if demand exists
        - Diversify trailer mix to reduce dependency on single type
        - Monitor market rates for each trailer type
        """
    
    elif chart_type == "weekly_earnings":
        # Analysis for Weekly Driver Earnings
        if not driver_rate_col:
            return "Driver rate column not found for analysis."
        if 'DELIVERY DATE' in df.columns:
            df_copy = df.copy()
            df_copy['DELIVERY DATE'] = pd.to_datetime(df_copy['DELIVERY DATE'], errors='coerce')
            df_copy['Week'] = df_copy['DELIVERY DATE'].dt.to_period('W')
            
            weekly_earnings = df_copy.groupby('Week')[driver_rate_col].mean()
            avg_weekly_earnings = weekly_earnings.mean()
            max_weekly_earnings = weekly_earnings.max()
            min_weekly_earnings = weekly_earnings.min()
            
            analysis = f"""
            **📊 Weekly Driver Earnings Analysis (Based on Delivery Dates)**
            
            **Key Findings:**
            - **Average Weekly Earnings**: ${avg_weekly_earnings:,.0f}
            - **Peak Week**: ${max_weekly_earnings:,.0f}
            - **Lowest Week**: ${min_weekly_earnings:,.0f}
            - **Earnings Range**: ${max_weekly_earnings - min_weekly_earnings:,.0f}
            - **Weeks Analyzed**: {len(weekly_earnings)}
            
            **Business Insights:**
            - Weekly earnings vary by ${max_weekly_earnings - min_weekly_earnings:,.0f}
            - Peak week is {max_weekly_earnings/avg_weekly_earnings:.1f}x above average
            - Monitor seasonal patterns for capacity planning
            - Consider driver incentives during low-earning weeks
            """
        else:
            analysis = "**📊 Weekly Driver Earnings Analysis**\n\nDelivery date data not available for weekly analysis."
    
    elif chart_type == "carrier_miles":
        # Analysis for Total Miles per Carrier
        carrier_miles = df.groupby('LOAD\'S CARRIER COMPANY')['FULL MILES TOTAL'].sum().sort_values(ascending=False)
        total_miles = carrier_miles.sum()
        top_carrier_miles = carrier_miles.iloc[0]
        top_carrier_name = carrier_miles.index[0]
        top_carrier_pct = (top_carrier_miles / total_miles * 100)
        
        analysis = f"""
        **📊 Total Miles per Carrier Analysis**
        
        **Key Findings:**
        - **Total Miles**: {total_miles:,.0f} miles
        - **Top Carrier**: {top_carrier_name} ({top_carrier_miles:,.0f} miles, {top_carrier_pct:.1f}% of total)
        - **Active Carriers**: {len(carrier_miles)} carriers
        - **Average Miles per Carrier**: {total_miles/len(carrier_miles):,.0f} miles
        
        **Business Insights:**
        - {top_carrier_name} handles {top_carrier_pct:.1f}% of total miles
        - Consider capacity planning based on carrier mile distribution
        - Monitor carrier reliability and on-time performance
        - Balance workload across carriers for optimal efficiency
        """
    
    elif chart_type == "margin_distribution":
        # Analysis for BD Margin Distribution
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        
        # Find BD margin column
        bd_margin_col = find_bd_margin_column(df)
        if not bd_margin_col:
            return "BD margin column not found for analysis."
        
        df_copy = df.copy()
        df_copy['Margin_Percentage'] = (df_copy[bd_margin_col] / df_copy[broker_rate_col] * 100).fillna(0)
        margin_data = df_copy[(df_copy['Margin_Percentage'] <= 100) & (df_copy['Margin_Percentage'] >= -50)]
        
        avg_margin = margin_data['Margin_Percentage'].mean()
        median_margin = margin_data['Margin_Percentage'].median()
        std_margin = margin_data['Margin_Percentage'].std()
        
        analysis = f"""
        **📊 BD Margin Distribution Analysis**
        
        **Key Findings:**
        - **Average Margin**: {avg_margin:.1f}%
        - **Median Margin**: {median_margin:.1f}%
        - **Margin Standard Deviation**: {std_margin:.1f}%
        - **Margin Range**: {margin_data['Margin_Percentage'].min():.1f}% to {margin_data['Margin_Percentage'].max():.1f}%
        - **Loads Analyzed**: {len(margin_data):,}
        
        **Business Insights:**
        - {avg_margin:.1f}% average margin indicates overall profitability
        - Margin variability ({std_margin:.1f}% std dev) suggests pricing opportunities
        - {median_margin:.1f}% median shows typical load profitability
        - Monitor margin trends for pricing strategy adjustments
        """
    
    elif chart_type == "carrier_performance":
        # Analysis for Combined Carrier Performance
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        carrier_stats = df.groupby('LOAD\'S CARRIER COMPANY').agg({
            broker_rate_col: 'sum',
            'FULL MILES TOTAL': 'sum'
        }).reset_index()
        carrier_stats['Revenue_Per_Mile'] = (carrier_stats[broker_rate_col] / carrier_stats['FULL MILES TOTAL']).fillna(0)
        
        top_revenue = carrier_stats.loc[carrier_stats[broker_rate_col].idxmax()]
        top_miles = carrier_stats.loc[carrier_stats['FULL MILES TOTAL'].idxmax()]
        top_efficiency = carrier_stats.loc[carrier_stats['Revenue_Per_Mile'].idxmax()]
        
        analysis = f"""
        **📊 Comprehensive Carrier Performance Analysis**
        
        **Key Findings:**
        - **Highest Revenue**: {top_revenue["LOAD'S CARRIER COMPANY"]} (${top_revenue[broker_rate_col]:,.0f})
        - **Most Miles**: {top_miles["LOAD'S CARRIER COMPANY"]} ({top_miles['FULL MILES TOTAL']:,.0f} miles)
        - **Best Efficiency**: {top_efficiency["LOAD'S CARRIER COMPANY"]} (${top_efficiency['Revenue_Per_Mile']:.2f}/mile)
        - **Active Carriers**: {len(carrier_stats)} carriers
        
        **Business Insights:**
        - {top_revenue["LOAD'S CARRIER COMPANY"]} generates the most revenue
        - {top_miles["LOAD'S CARRIER COMPANY"]} handles the most miles
        - {top_efficiency["LOAD'S CARRIER COMPANY"]} provides the best revenue per mile
        - Consider strategic partnerships with high-efficiency carriers
        - Balance volume and efficiency for optimal carrier mix
        """
    
    elif chart_type == "driver_income":
        # Analysis for Driver Income Analysis
        if 'DELIVERY DATE' not in df.columns or 'DRIVER NAME' not in df.columns:
            analysis = "**📊 Driver Income Analysis**\n\nRequired delivery date and driver data not available for analysis."
            return analysis
        
        # Find the actual column names in the data
        driver_rate_col = find_driver_rate_column(df)
        trailer_col = find_trailer_column(df)
        
        if not driver_rate_col:
            return "Driver rate column not found for analysis."
        if not trailer_col:
            return "Trailer column not found for analysis."
        
        # Import datetime here to avoid circular imports
        from datetime import datetime, timedelta
        
        # Convert DELIVERY DATE to datetime
        df_copy = df.copy()
        df_copy['DELIVERY DATE'] = pd.to_datetime(df_copy['DELIVERY DATE'], errors='coerce')
        
        # Calculate week start (Tuesday) for each delivery date
        def get_week_start(date):
            if pd.isna(date):
                return pd.NaT
            # Find the most recent Tuesday
            days_since_tuesday = (date.weekday() - 1) % 7
            return date - timedelta(days=days_since_tuesday)
        
        df_copy['WEEK_START'] = df_copy['DELIVERY DATE'].apply(get_week_start)
        
        # Group by driver and week
        weekly_earnings = df_copy.groupby(['DRIVER NAME', 'WEEK_START', trailer_col])[driver_rate_col].sum().reset_index()
        
        # Calculate target and percentage
        def get_weekly_target(trailer_type):
            if trailer_type in ['Flatbed', 'Stepdeck']:
                return 6000
            else:
                return 5500
        
        weekly_earnings['TARGET'] = weekly_earnings[trailer_col].apply(get_weekly_target)
        weekly_earnings['PERCENTAGE_TO_TARGET'] = (weekly_earnings[driver_rate_col] / weekly_earnings['TARGET'] * 100).round(1)
        
        # Calculate statistics
        avg_weekly_earnings = weekly_earnings[driver_rate_col].mean()
        weeks_above_target = len(weekly_earnings[weekly_earnings['PERCENTAGE_TO_TARGET'] >= 100])
        total_weeks = len(weekly_earnings)
        target_achievement_rate = (weeks_above_target / total_weeks * 100) if total_weeks > 0 else 0
        
        # Trailer type analysis
        flatbed_avg = weekly_earnings[weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])][driver_rate_col].mean()
        other_avg = weekly_earnings[~weekly_earnings[trailer_col].isin(['Flatbed', 'Stepdeck'])][driver_rate_col].mean()
        
        # Top performers
        top_driver = weekly_earnings.groupby('DRIVER NAME')[driver_rate_col].mean().idxmax()
        top_driver_avg = weekly_earnings.groupby('DRIVER NAME')[driver_rate_col].mean().max()
        
        analysis = f"""
        **📊 Driver Income Analysis: Weekly Earnings vs Targets (Based on Delivery Dates)**
        
        **Key Findings:**
        - **Average Weekly Earnings**: ${avg_weekly_earnings:,.0f}
        - **Target Achievement Rate**: {target_achievement_rate:.1f}% of weeks meet or exceed target
        - **Flatbed/Stepdeck Average**: ${flatbed_avg:,.0f} (Target: $6,000)
        - **Other Trailers Average**: ${other_avg:,.0f} (Target: $5,500)
        - **Top Performer**: {top_driver} (${top_driver_avg:,.0f} avg weekly)
        - **Total Weeks Analyzed**: {total_weeks}
        
        **Target Analysis:**
        - **Flatbed/Stepdeck Target**: $6,000/week
        - **Other Trailers Target**: $5,500/week
        - **Weeks Above Target**: {weeks_above_target} out of {total_weeks}
        
        **Business Insights:**
        - {target_achievement_rate:.1f}% target achievement indicates driver performance
        - Flatbed/Stepdeck drivers average {flatbed_avg/6000:.1%} of their target
        - Other trailer drivers average {other_avg/5500:.1%} of their target
        - {top_driver} is the highest-earning driver with ${top_driver_avg:,.0f} average weekly earnings
        - Consider driver incentives for those consistently below targets
        - Monitor trailer type performance for capacity planning
        """
    
    elif chart_type == "revenue_per_mile_dispatcher":
        # Analysis for Revenue per Mile per Dispatcher
        # Find the miles column
        miles_col = None
        for col in df.columns:
            if 'miles' in col.lower() or 'mile' in col.lower():
                miles_col = col
                break
        
        if not miles_col:
            analysis = "**📊 Revenue per Mile per Dispatcher Analysis**\n\nMiles data not available for analysis."
            return analysis
        
        # Find dispatcher column
        dispatcher_col = find_dispatcher_column(df)
        if not dispatcher_col:
            analysis = "**📊 Revenue per Mile per Dispatcher Analysis**\n\nDispatcher data not available for analysis."
            return analysis
        
        if not broker_rate_col:
            return "Broker rate column not found for analysis."
        dispatcher_metrics = df.groupby(dispatcher_col).agg({
            broker_rate_col: 'sum',
            miles_col: 'sum'
        }).reset_index()
        
        dispatcher_metrics['REVENUE_PER_MILE'] = (dispatcher_metrics[broker_rate_col] / dispatcher_metrics[miles_col]).round(2)
        dispatcher_metrics = dispatcher_metrics[dispatcher_metrics['REVENUE_PER_MILE'].notna() & (dispatcher_metrics['REVENUE_PER_MILE'] != float('inf'))]
        
        if dispatcher_metrics.empty:
            analysis = "**📊 Revenue per Mile per Dispatcher Analysis**\n\nNo valid revenue per mile data available for analysis."
            return analysis
        
        top_dispatcher = dispatcher_metrics.loc[dispatcher_metrics['REVENUE_PER_MILE'].idxmax()]
        avg_revenue_per_mile = dispatcher_metrics['REVENUE_PER_MILE'].mean()
        total_revenue = dispatcher_metrics[broker_rate_col].sum()
        total_miles = dispatcher_metrics[miles_col].sum()
        
        analysis = f"""
        **📊 Revenue per Mile per Dispatcher Analysis**
        
        **Key Findings:**
        - **Top Performer**: {top_dispatcher[dispatcher_col]} (${top_dispatcher['REVENUE_PER_MILE']:.2f}/mile)
        - **Average Revenue per Mile**: ${avg_revenue_per_mile:.2f}
        - **Total Revenue**: ${total_revenue:,.0f}
        - **Total Miles**: {total_miles:,.0f} miles
        - **Active Dispatchers**: {len(dispatcher_metrics)} dispatchers
        
        **Business Insights:**
        - {top_dispatcher[dispatcher_col]} generates the highest revenue per mile
        - Average revenue per mile is ${avg_revenue_per_mile:.2f}
        - Consider sharing best practices from top performers
        - Monitor dispatcher efficiency and route optimization
        - Focus on high-value loads and optimal pricing strategies
        """
    
    elif chart_type == "driver_revenue_per_mile_dispatcher":
        # Analysis for Driver Revenue per Mile per Dispatcher
        # Find the miles column
        miles_col = None
        for col in df.columns:
            if 'miles' in col.lower() or 'mile' in col.lower():
                miles_col = col
                break
        
        if not miles_col:
            analysis = "**📊 Driver Revenue per Mile per Dispatcher Analysis**\n\nMiles data not available for analysis."
            return analysis
        
        # Find dispatcher column
        dispatcher_col = find_dispatcher_column(df)
        if not dispatcher_col:
            analysis = "**📊 Driver Revenue per Mile per Dispatcher Analysis**\n\nDispatcher data not available for analysis."
            return analysis
        
        if not driver_rate_col:
            return "Driver rate column not found for analysis."
        dispatcher_metrics = df.groupby(dispatcher_col).agg({
            driver_rate_col: 'sum',
            miles_col: 'sum'
        }).reset_index()
        
        dispatcher_metrics['DRIVER_REVENUE_PER_MILE'] = (dispatcher_metrics[driver_rate_col] / dispatcher_metrics[miles_col]).round(2)
        dispatcher_metrics = dispatcher_metrics[dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].notna() & (dispatcher_metrics['DRIVER_REVENUE_PER_MILE'] != float('inf'))]
        
        if dispatcher_metrics.empty:
            analysis = "**📊 Driver Revenue per Mile per Dispatcher Analysis**\n\nNo valid driver revenue per mile data available for analysis."
            return analysis
        
        top_dispatcher = dispatcher_metrics.loc[dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].idxmax()]
        avg_driver_revenue_per_mile = dispatcher_metrics['DRIVER_REVENUE_PER_MILE'].mean()
        total_driver_pay = dispatcher_metrics[driver_rate_col].sum()
        total_miles = dispatcher_metrics[miles_col].sum()
        
        analysis = f"""
        **📊 Driver Revenue per Mile per Dispatcher Analysis**
        
        **Key Findings:**
        - **Top Performer**: {top_dispatcher[dispatcher_col]} (${top_dispatcher['DRIVER_REVENUE_PER_MILE']:.2f}/mile)
        - **Average Driver Revenue per Mile**: ${avg_driver_revenue_per_mile:.2f}
        - **Total Driver Pay**: ${total_driver_pay:,.0f}
        - **Total Miles**: {total_miles:,.0f} miles
        - **Active Dispatchers**: {len(dispatcher_metrics)} dispatchers
        
        **Business Insights:**
        - {top_dispatcher[dispatcher_col]} pays drivers the highest rate per mile
        - Average driver revenue per mile is ${avg_driver_revenue_per_mile:.2f}
        - Monitor driver compensation fairness across dispatchers
        - Consider driver retention strategies for high-paying dispatchers
        - Balance driver pay with company profitability
        """
    
    return analysis 

def plot_weekly_driver_revenue_per_mile_vs_target_faceted(weekly_earnings, miles_col='FULL MILES TOTAL', flatbed_rpm_target=2.0, dryvan_rpm_target=1.8):
    """Faceted horizontal bar chart: one facet per trailer group, y=driver, x=revenue per mile, with color gradient based on target achievement."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    
    if miles_col not in weekly_earnings.columns:
        return go.Figure().add_annotation(text="No miles data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Revenue per mile targets (configurable)
    revenue_per_mile_targets = {
        'Flatbed/Stepdeck': flatbed_rpm_target,  # Configurable target
        'Dry Van/Reefer/Power Only': dryvan_rpm_target  # Configurable target
    }
    
    df = weekly_earnings.copy()
    df = df[df['DRIVER NAME'].notna() & (df['DRIVER NAME'].astype(str).str.strip() != '')].copy()
    df['DRIVER NAME'] = df['DRIVER NAME'].astype(str)
    
    def parse_number(value):
        import re
        import pandas as pd
        if pd.isna(value) or value == '':
            return 0.0
        value_str = str(value).strip()
        value_str = re.sub(r'[^\d.,]', '', value_str)
        if ',' in value_str:
            value_str = value_str.replace(',', '')
        try:
            return float(value_str)
        except ValueError:
            return 0.0
    
    df[miles_col] = df[miles_col].apply(parse_number)
    
    # Find the actual driver rate column in the data
    driver_rate_col = find_driver_rate_column(df)
    if not driver_rate_col:
        return go.Figure().add_annotation(text="Driver rate column not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Calculate revenue per mile
    df['REVENUE_PER_MILE'] = np.where(df[miles_col] > 0, df[driver_rate_col] / df[miles_col], 0)
    
    # Group by driver, week, and trailer group
    revenue_df = df.groupby(['DRIVER NAME', 'WEEK_START', 'TRAILER GROUP']).agg({
        driver_rate_col: 'sum',
        miles_col: 'sum'
    }).reset_index()
    
    # Calculate revenue per mile for grouped data
    revenue_df['REVENUE_PER_MILE'] = np.where(revenue_df[miles_col] > 0, 
                                             revenue_df[driver_rate_col] / revenue_df[miles_col], 0)
    
    # Calculate percentage to target
    def get_target_for_group(group):
        return revenue_per_mile_targets.get(group, 1.8)  # Default to 1.8 if group not found
    
    revenue_df['TARGET'] = revenue_df['TRAILER GROUP'].apply(get_target_for_group)
    revenue_df['PERCENTAGE_TO_TARGET'] = (revenue_df['REVENUE_PER_MILE'] / revenue_df['TARGET'] * 100).round(1)
    
    trailer_groups = revenue_df['TRAILER GROUP'].dropna().unique().tolist()
    n = len(trailer_groups)
    if n == 0:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig = make_subplots(rows=1, cols=n, subplot_titles=trailer_groups, shared_yaxes=False)
    
    for i, group in enumerate(trailer_groups):
        sub = revenue_df[revenue_df['TRAILER GROUP'] == group].copy()
        
        def get_color_category(percentage):
            if percentage > 110:
                return '💚 Overachievers (>110%)'
            elif percentage >= 100:
                return '🟢 On Target (100–110%)'
            elif percentage >= 80:
                return '🟡 Watchlist (80–99%)'
            else:
                return '🔴 Underperformers (<80%)'
        
        sub['Color_Category'] = sub['PERCENTAGE_TO_TARGET'].apply(get_color_category)
        color_map = {
            '💚 Overachievers (>110%)': '#00FF00',
            '🟢 On Target (100–110%)': '#90EE90',
            '🟡 Watchlist (80–99%)': '#FFFF00',
            '🔴 Underperformers (<80%)': '#FF0000'
        }
        
        # Sort by revenue per mile for consistent y-axis - INVERTED ORDER
        sub = sub.sort_values('REVENUE_PER_MILE', ascending=True)
        y_order = sub['DRIVER NAME'].tolist()
        
        for category in color_map.keys():
            category_data = sub[sub['Color_Category'] == category]
            if not category_data.empty:
                fig.add_trace(
                    go.Bar(
                        y=category_data['DRIVER NAME'],
                        x=category_data['REVENUE_PER_MILE'],
                        orientation='h',
                        name=category,
                        marker_color=color_map[category],
                        showlegend=(i == 0),
                        hovertemplate='<b>%{y}</b><br>Revenue per Mile: $%{x:.2f}<br>% to Target: ' + \
                                    category_data['PERCENTAGE_TO_TARGET'].astype(str) + '%<extra></extra>'
                    ),
                    row=1, col=i+1
                )
        
        # Set y-axis order explicitly - INVERTED
        fig.update_yaxes(categoryorder='array', categoryarray=y_order, row=1, col=i+1)
        
        # Add target line
        target_value = revenue_per_mile_targets.get(group, 1.8)
        fig.add_vline(
            x=target_value,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Target (${target_value:.2f})",
            row=1, col=i+1
        )
        fig.update_xaxes(title_text="Revenue per Mile ($)", row=1, col=i+1)
    
    fig.update_layout(
        title="Weekly Driver Revenue per Mile vs Target by Trailer Type",
        height=400 + 30 * max(5, revenue_df['DRIVER NAME'].nunique()),
        margin=dict(t=80, l=200),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

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

def create_week_over_week_comparison(weekly_kpis, current_week=None):
    """
    Create a side-by-side week-over-week comparison dashboard.
    
    Args:
        weekly_kpis: DataFrame with weekly KPI data
        current_week: Current week to compare against (defaults to most recent)
    
    Returns:
        Plotly figure with side-by-side comparison
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    if weekly_kpis.empty or len(weekly_kpis) < 2:
        return go.Figure().add_annotation(
            text="Insufficient data for week-over-week comparison (need at least 2 weeks)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Sort by week start date
    weekly_kpis = weekly_kpis.sort_values('WEEK_START').reset_index(drop=True)
    
    # Determine current and previous week
    if current_week is None:
        current_week = weekly_kpis['WEEK_START'].iloc[-1]
    
    current_data = weekly_kpis[weekly_kpis['WEEK_START'] == current_week]
    if current_data.empty:
        current_week = weekly_kpis['WEEK_START'].iloc[-1]
        current_data = weekly_kpis[weekly_kpis['WEEK_START'] == current_week]
    
    # Find previous week
    previous_week = weekly_kpis[weekly_kpis['WEEK_START'] < current_week]['WEEK_START'].iloc[-1] if len(weekly_kpis[weekly_kpis['WEEK_START'] < current_week]) > 0 else None
    
    if previous_week is None:
        return go.Figure().add_annotation(
            text="No previous week data available for comparison",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    previous_data = weekly_kpis[weekly_kpis['WEEK_START'] == previous_week]
    
    # Get column names
    broker_rate_col = find_broker_rate_column(weekly_kpis)
    driver_rate_col = find_driver_rate_column(weekly_kpis)
    
    if not broker_rate_col or not driver_rate_col:
        return go.Figure().add_annotation(
            text="Required rate columns not found for comparison",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Extract values
    current_loads = current_data['LOAD_COUNT'].iloc[0] if 'LOAD_COUNT' in current_data.columns else 0
    current_billing = current_data[broker_rate_col].iloc[0]
    current_driver_pay = current_data[driver_rate_col].iloc[0]
    current_margin = current_data['GROSS_MARGIN'].iloc[0] if 'GROSS_MARGIN' in current_data.columns else 0
    
    previous_loads = previous_data['LOAD_COUNT'].iloc[0] if 'LOAD_COUNT' in previous_data.columns else 0
    previous_billing = previous_data[broker_rate_col].iloc[0]
    previous_driver_pay = previous_data[driver_rate_col].iloc[0]
    previous_margin = previous_data['GROSS_MARGIN'].iloc[0] if 'GROSS_MARGIN' in previous_data.columns else 0
    
    # Calculate changes
    loads_change = ((current_loads - previous_loads) / previous_loads * 100) if previous_loads > 0 else 0
    billing_change = ((current_billing - previous_billing) / previous_billing * 100) if previous_billing > 0 else 0
    driver_pay_change = ((current_driver_pay - previous_driver_pay) / previous_driver_pay * 100) if previous_driver_pay > 0 else 0
    margin_change = current_margin - previous_margin
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Load Count: {current_loads} vs {previous_loads}',
            f'Total Billing: ${current_billing:,.0f} vs ${previous_billing:,.0f}',
            f'Driver Pay: ${current_driver_pay:,.0f} vs ${previous_driver_pay:,.0f}',
            f'B-Rate %: {current_margin:.1f}% vs {previous_margin:.1f}%'
        ),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Define colors based on performance
    def get_color(value, is_margin=False):
        if is_margin:
            return 'green' if value >= 0 else 'red'
        return 'green' if value >= 0 else 'red'
    
    # Load Count Comparison
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current_loads,
        delta={'reference': previous_loads, 'relative': True, 'valueformat': '.1%'},
        gauge={'axis': {'range': [None, max(current_loads, previous_loads) * 1.2]},
               'bar': {'color': get_color(loads_change)},
               'steps': [{'range': [0, previous_loads], 'color': "lightgray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': previous_loads}},
        title={'text': "Load Count"},
        domain={'row': 0, 'column': 0}
    ))
    
    # Billing Comparison
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current_billing,
        delta={'reference': previous_billing, 'relative': True, 'valueformat': '.1%'},
        gauge={'axis': {'range': [None, max(current_billing, previous_billing) * 1.2]},
               'bar': {'color': get_color(billing_change)},
               'steps': [{'range': [0, previous_billing], 'color': "lightgray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': previous_billing}},
        title={'text': "Total Billing ($)"},
        domain={'row': 0, 'column': 1}
    ))
    
    # Driver Pay Comparison
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current_driver_pay,
        delta={'reference': previous_driver_pay, 'relative': True, 'valueformat': '.1%'},
        gauge={'axis': {'range': [None, max(current_driver_pay, previous_driver_pay) * 1.2]},
               'bar': {'color': get_color(driver_pay_change)},
               'steps': [{'range': [0, previous_driver_pay], 'color': "lightgray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': previous_driver_pay}},
        title={'text': "Driver Pay ($)"},
        domain={'row': 1, 'column': 0}
    ))
    
    # Margin Comparison
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current_margin,
        delta={'reference': previous_margin, 'relative': False, 'valueformat': '.1f'},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': get_color(margin_change, True)},
               'steps': [{'range': [0, previous_margin], 'color': "lightgray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': previous_margin}},
        title={'text': "B-Rate %"},
        domain={'row': 1, 'column': 1}
    ))
    
    # Update layout
    current_week_label = current_data['WEEK_LABEL'].iloc[0] if 'WEEK_LABEL' in current_data.columns else current_week.strftime('%m/%d/%Y')
    previous_week_label = previous_data['WEEK_LABEL'].iloc[0] if 'WEEK_LABEL' in previous_data.columns else previous_week.strftime('%m/%d/%Y')
    
    fig.update_layout(
        title=f"Week-over-Week Comparison: {current_week_label} vs {previous_week_label}",
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_sparkline_trends(weekly_kpis, kpi_columns=None, num_weeks=8):
    """
    Create sparkline trend charts for KPIs showing recent performance.
    
    Args:
        weekly_kpis: DataFrame with weekly KPI data
        kpi_columns: List of KPI columns to show trends for
        num_weeks: Number of recent weeks to show in sparklines
    
    Returns:
        Plotly figure with sparkline trends
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    if weekly_kpis.empty or len(weekly_kpis) < 2:
        return go.Figure().add_annotation(
            text="Insufficient data for trend analysis (need at least 2 weeks)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Sort by week start date
    weekly_kpis = weekly_kpis.sort_values('WEEK_START').reset_index(drop=True)
    
    # Define default KPI columns if not provided
    if kpi_columns is None:
        broker_rate_col = find_broker_rate_column(weekly_kpis)
        driver_rate_col = find_driver_rate_column(weekly_kpis)
        kpi_columns = ['LOAD_COUNT', broker_rate_col, driver_rate_col, 'GROSS_MARGIN']
        kpi_columns = [col for col in kpi_columns if col in weekly_kpis.columns]
    
    # Limit to specified number of weeks for sparklines
    recent_data = weekly_kpis.tail(num_weeks).copy()
    
    if len(recent_data) < 2:
        return go.Figure().add_annotation(
            text="Insufficient recent data for trend analysis",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Create subplots
    n_cols = len(kpi_columns)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[col.replace('_', ' ').title() for col in kpi_columns],
        specs=[[{"type": "scatter"} for _ in range(n_cols)]]
    )
    
    # Define colors for trends
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, col in enumerate(kpi_columns):
        if col not in recent_data.columns:
            continue
            
        # Calculate trend
        values = recent_data[col].values
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        # Determine color based on trend
        if trend > 0:
            line_color = 'green'
        elif trend < 0:
            line_color = 'red'
        else:
            line_color = 'gray'
        
        # Create sparkline
        fig.add_trace(go.Scatter(
            x=recent_data['WEEK_LABEL'] if 'WEEK_LABEL' in recent_data.columns else range(len(values)),
            y=values,
            mode='lines+markers',
            name=col.replace('_', ' ').title(),
            line=dict(color=line_color, width=2),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate=f'{col.replace("_", " ").title()}: %{{y:,.0f}}<extra></extra>'
        ), row=1, col=i+1)
        
        # Add trend indicator
        current_value = values[-1]
        previous_value = values[-2] if len(values) > 1 else values[0]
        change_pct = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        
        # Add annotation for trend
        fig.add_annotation(
            x=len(values)-1,  # Use index position instead of label
            y=values[-1],
            xref=f'x{i+1}',
            yref=f'y{i+1}',
            text=f"{change_pct:+.1f}%",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            font=dict(color=line_color, size=12, weight='bold'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=line_color,
            borderwidth=1
        )
        
        # Update subplot layout
        fig.update_xaxes(showticklabels=False, row=1, col=i+1)
        fig.update_yaxes(showticklabels=False, row=1, col=i+1)
    
    fig.update_layout(
        title="Recent KPI Trends (Last 8 Weeks)",
        height=200,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_weekly_comparison_table(weekly_kpis, num_weeks=8):
    """
    Create an interactive table showing weekly comparisons with drill-down capability.
    
    Args:
        weekly_kpis: DataFrame with weekly KPI data
        num_weeks: Number of recent weeks to show
    
    Returns:
        DataFrame formatted for display
    """
    if weekly_kpis.empty:
        return pd.DataFrame()
    
    # Sort by week start date and get recent weeks
    weekly_kpis = weekly_kpis.sort_values('WEEK_START').reset_index(drop=True)
    recent_data = weekly_kpis.tail(num_weeks).copy()
    
    # Get column names
    broker_rate_col = find_broker_rate_column(recent_data)
    driver_rate_col = find_driver_rate_column(recent_data)
    
    if not broker_rate_col or not driver_rate_col:
        return pd.DataFrame()
    
    # Create comparison table
    comparison_table = recent_data[['WEEK_LABEL', 'LOAD_COUNT', broker_rate_col, driver_rate_col, 'GROSS_MARGIN']].copy()
    
    # Calculate week-over-week changes
    for col in ['LOAD_COUNT', broker_rate_col, driver_rate_col, 'GROSS_MARGIN']:
        if col in comparison_table.columns:
            comparison_table[f'{col}_Change'] = comparison_table[col].pct_change() * 100
            comparison_table[f'{col}_Change'] = comparison_table[f'{col}_Change'].fillna(0)
    
    # Format columns for display
    comparison_table['Week'] = comparison_table['WEEK_LABEL']
    comparison_table['Loads'] = comparison_table['LOAD_COUNT'].astype(int)
    comparison_table['Billing'] = comparison_table[broker_rate_col].apply(lambda x: f"${x:,.0f}")
    comparison_table['Driver_Pay'] = comparison_table[driver_rate_col].apply(lambda x: f"${x:,.0f}")
    comparison_table['B_Rate_%'] = comparison_table['GROSS_MARGIN'].apply(lambda x: f"{x:.1f}%")
    
    # Format change columns
    comparison_table['Loads_Change'] = comparison_table['LOAD_COUNT_Change'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
    comparison_table['Billing_Change'] = comparison_table[f'{broker_rate_col}_Change'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
    comparison_table['Driver_Pay_Change'] = comparison_table[f'{driver_rate_col}_Change'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
    comparison_table['B_Rate_Change'] = comparison_table['GROSS_MARGIN_Change'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
    
    # Select display columns
    display_columns = ['Week', 'Loads', 'Loads_Change', 'Billing', 'Billing_Change', 
                      'Driver_Pay', 'Driver_Pay_Change', 'B_Rate_%', 'B_Rate_Change']
    
    return comparison_table[display_columns]

def generate_week_over_week_insights(weekly_kpis):
    """
    Generate insights and recommendations based on week-over-week performance.
    
    Args:
        weekly_kpis: DataFrame with weekly KPI data
    
    Returns:
        String with insights and recommendations
    """
    if weekly_kpis.empty or len(weekly_kpis) < 2:
        return "Insufficient data for week-over-week analysis."
    
    # Sort by week start date
    weekly_kpis = weekly_kpis.sort_values('WEEK_START').reset_index(drop=True)
    
    # Get column names
    broker_rate_col = find_broker_rate_column(weekly_kpis)
    driver_rate_col = find_driver_rate_column(weekly_kpis)
    
    if not broker_rate_col or not driver_rate_col:
        return "Required rate columns not found for analysis."
    
    # Get current and previous week
    current_week = weekly_kpis.iloc[-1]
    previous_week = weekly_kpis.iloc[-2]
    
    # Calculate changes
    loads_change = ((current_week['LOAD_COUNT'] - previous_week['LOAD_COUNT']) / previous_week['LOAD_COUNT'] * 100) if previous_week['LOAD_COUNT'] > 0 else 0
    billing_change = ((current_week[broker_rate_col] - previous_week[broker_rate_col]) / previous_week[broker_rate_col] * 100) if previous_week[broker_rate_col] > 0 else 0
    driver_pay_change = ((current_week[driver_rate_col] - previous_week[driver_rate_col]) / previous_week[driver_rate_col] * 100) if previous_week[driver_rate_col] > 0 else 0
    margin_change = current_week['GROSS_MARGIN'] - previous_week['GROSS_MARGIN']
    
    # Generate insights
    insights = []
    
    # Load count insights
    if loads_change > 10:
        insights.append(f"🚀 **Load volume increased significantly** (+{loads_change:.1f}%) - Consider capacity planning for sustained growth")
    elif loads_change < -10:
        insights.append(f"⚠️ **Load volume decreased** ({loads_change:.1f}%) - Investigate market conditions and sales pipeline")
    elif abs(loads_change) <= 5:
        insights.append(f"📊 **Load volume stable** ({loads_change:+.1f}%) - Consistent operations maintained")
    
    # Billing insights
    if billing_change > 10:
        insights.append(f"💰 **Revenue growth strong** (+{billing_change:.1f}%) - Excellent market positioning")
    elif billing_change < -10:
        insights.append(f"📉 **Revenue decline** ({billing_change:.1f}%) - Review pricing strategy and market conditions")
    
    # Driver pay insights
    if driver_pay_change > 10:
        insights.append(f"👥 **Driver compensation increased** (+{driver_pay_change:.1f}%) - Monitor driver satisfaction and retention")
    elif driver_pay_change < -10:
        insights.append(f"⚠️ **Driver compensation decreased** ({driver_pay_change:.1f}%) - Check driver morale and retention risk")
    
    # Margin insights
    if margin_change > 2:
        insights.append(f"✅ **Margin improved** (+{margin_change:.1f}%) - Strong operational efficiency")
    elif margin_change < -2:
        insights.append(f"🔴 **Margin declined** ({margin_change:.1f}%) - Review cost structure and pricing")
    
    # Overall performance assessment
    positive_changes = sum([1 for change in [loads_change, billing_change, margin_change] if change > 0])
    total_metrics = 3
    
    if positive_changes >= 2:
        overall_sentiment = "🎯 **Overall Performance: Strong** - Most metrics showing positive trends"
    elif positive_changes == 1:
        overall_sentiment = "📊 **Overall Performance: Mixed** - Some areas need attention"
    else:
        overall_sentiment = "⚠️ **Overall Performance: Concerning** - Multiple metrics declining"
    
    insights.insert(0, overall_sentiment)
    
    return "\n\n".join(insights)