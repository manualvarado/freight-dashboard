import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_total_billing_per_carrier(df):
    """1. Total Billing per Carrier - Horizontal Bar Chart (sorted descending)"""
    if df.empty or 'LOAD\'S CARRIER COMPANY' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by carrier and sum billing
    carrier_billing = df.groupby('LOAD\'S CARRIER COMPANY')['BROKER RATE (CFC)'].sum().reset_index()
    carrier_billing = carrier_billing.sort_values('BROKER RATE (CFC)', ascending=True)  # For horizontal bar, ascending=True shows highest at top
    
    fig = px.bar(
        carrier_billing, 
        x='BROKER RATE (CFC)', 
        y='LOAD\'S CARRIER COMPANY',
        orientation='h',
        title='Total Billing per Carrier',
        labels={'BROKER RATE (CFC)': 'Total Billing ($)', 'LOAD\'S CARRIER COMPANY': 'Carrier'},
        color='BROKER RATE (CFC)',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Total Billing ($)",
        yaxis_title="Carrier"
    )
    
    return fig

def plot_total_billing_per_dispatcher(df):
    """2. Total Billing per Dispatcher - Horizontal Bar Chart + Data Labels"""
    if df.empty or 'DISPATCH NAME' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by dispatcher and sum billing
    dispatcher_billing = df.groupby('DISPATCH NAME')['BROKER RATE (CFC)'].sum().reset_index()
    dispatcher_billing = dispatcher_billing.sort_values('BROKER RATE (CFC)', ascending=True)
    
    fig = px.bar(
        dispatcher_billing,
        x='BROKER RATE (CFC)',
        y='DISPATCH NAME',
        orientation='h',
        title='Total Billing per Dispatcher',
        labels={'BROKER RATE (CFC)': 'Total Billing ($)', 'DISPATCH NAME': 'Dispatcher'},
        text='BROKER RATE (CFC)',  # Data labels
        color='BROKER RATE (CFC)',
        color_continuous_scale='Greens'
    )
    
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Total Billing ($)",
        yaxis_title="Dispatcher"
    )
    
    return fig

def plot_bd_margin_performance(df):
    """3. BD Margin per Dispatcher - Scatter Plot (X: Total Billing, Y: BD Margin) + Bubble Size = Load Count"""
    if df.empty or 'DISPATCH NAME' not in df.columns or 'BD MARGIN' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by dispatcher - use size() instead of counting specific column
    dispatcher_stats = df.groupby('DISPATCH NAME').agg({
        'BROKER RATE (CFC)': 'sum',
        'BD MARGIN': 'sum'
    }).reset_index()
    
    # Add load count using size
    load_counts = df.groupby('DISPATCH NAME').size().reset_index(name='Load Count')
    dispatcher_stats = dispatcher_stats.merge(load_counts, on='DISPATCH NAME')
    
    fig = px.scatter(
        dispatcher_stats,
        x='BROKER RATE (CFC)',
        y='BD MARGIN',
        size='Load Count',
        hover_name='DISPATCH NAME',
        title='Dispatcher Performance: Billing vs Margin vs Load Count',
        labels={'BROKER RATE (CFC)': 'Total Billing ($)', 'BD MARGIN': 'BD Margin ($)', 'Load Count': 'Number of Loads'},
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
    if df.empty or 'DISPATCH NAME' not in df.columns or 'LOAD STATUS' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Create pivot table
    status_pivot = df.groupby(['DISPATCH NAME', 'LOAD STATUS']).size().unstack(fill_value=0)
    
    fig = px.bar(
        status_pivot,
        title='Load Status Distribution per Dispatcher',
        labels={'value': 'Number of Loads', 'DISPATCH NAME': 'Dispatcher'},
        color_discrete_map={
            'Booked': '#2E8B57',      # Green
            'Delivered': '#4169E1',   # Blue
            'On dispute': '#FFD700',  # Gold
            'TONU Received': '#FF6347', # Red
            'Cancelled': '#DC143C',   # Crimson
            'Disputing a TONU': '#FF4500' # Orange Red
        }
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
    if df.empty or 'DRIVER NAME' not in df.columns or 'DRIVER RATE' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Get top 20 drivers
    top_drivers = df.groupby('DRIVER NAME')['DRIVER RATE'].sum().nlargest(20).reset_index()
    top_drivers = top_drivers.sort_values('DRIVER RATE', ascending=False)
    
    # Calculate percentage of total
    total_earnings = df['DRIVER RATE'].sum()
    top_drivers['Percentage'] = (top_drivers['DRIVER RATE'] / total_earnings * 100).round(1)
    
    fig = px.bar(
        top_drivers,
        x='DRIVER NAME',
        y='DRIVER RATE',
        title='Top 20 Drivers by Earnings',
        labels={'DRIVER RATE': 'Total Earnings ($)', 'DRIVER NAME': 'Driver'},
        text='DRIVER RATE',
        color='DRIVER RATE',
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
    if df.empty or 'TRAILER TYPE' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by trailer type
    trailer_billing = df.groupby('TRAILER TYPE')['BROKER RATE (CFC)'].sum().reset_index()
    
    fig = px.pie(
        trailer_billing,
        values='BROKER RATE (CFC)',
        names='TRAILER TYPE',
        title='Total Billing by Trailer Type',
        hole=0.4  # Creates donut chart
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_average_driver_earnings_weekly(df):
    """7. Average Driver Earnings per Week - Line Chart"""
    if df.empty or 'DRIVER RATE' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Convert date column if it exists
    if 'PICK UP DATE' in df.columns:
        df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df_copy['PICK UP DATE'] = pd.to_datetime(df_copy['PICK UP DATE'], errors='coerce')
        df_copy['Week'] = df_copy['PICK UP DATE'].dt.to_period('W')
        
        weekly_earnings = df_copy.groupby('Week')['DRIVER RATE'].mean().reset_index()
        weekly_earnings['Week'] = weekly_earnings['Week'].astype(str)
        
        fig = px.line(
            weekly_earnings,
            x='Week',
            y='DRIVER RATE',
            title='Average Driver Earnings per Week',
            labels={'DRIVER RATE': 'Average Earnings ($)', 'Week': 'Week'},
            markers=True
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Week",
            yaxis_title="Average Earnings ($)"
        )
    else:
        fig = go.Figure().add_annotation(
            text="Date column not available for weekly analysis", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    return fig

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
        color_continuous_scale='Purples'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Carrier",
        yaxis_title="Total Miles",
        xaxis_tickangle=-45
    )
    
    return fig

def plot_driver_miles_heatmap(df):
    """9. Total Miles per Driver - Heatmap with Drivers on Y-axis and weeks on X-axis"""
    if df.empty or 'DRIVER NAME' not in df.columns or 'FULL MILES TOTAL' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Convert date column if it exists
    if 'PICK UP DATE' in df.columns:
        df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df_copy['PICK UP DATE'] = pd.to_datetime(df_copy['PICK UP DATE'], errors='coerce')
        df_copy['Week'] = df_copy['PICK UP DATE'].dt.to_period('W')
        
        # Get top 20 drivers by total miles
        top_drivers = df_copy.groupby('DRIVER NAME')['FULL MILES TOTAL'].sum().nlargest(20).index
        
        # Filter for top drivers and create pivot
        driver_weekly = df_copy[df_copy['DRIVER NAME'].isin(top_drivers)].groupby(['DRIVER NAME', 'Week'])['FULL MILES TOTAL'].sum().unstack(fill_value=0)
        
        # Convert Period index to string to avoid JSON serialization issues
        driver_weekly.columns = driver_weekly.columns.astype(str)
        
        fig = px.imshow(
            driver_weekly,
            title='Driver Miles Heatmap (Top 20 Drivers)',
            labels={'x': 'Week', 'y': 'Driver', 'color': 'Miles'},
            aspect='auto',
            color_continuous_scale='YlOrRd'
        )
        
        fig.update_layout(height=500)
    else:
        fig = go.Figure().add_annotation(
            text="Date column not available for weekly heatmap", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    return fig

def plot_bd_margin_distribution(df):
    """10. BD Margin % Distribution - Histogram + Line Chart (promedio móvil)"""
    if df.empty or 'BD MARGIN' not in df.columns or 'BROKER RATE (CFC)' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Calculate margin percentage
    df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df_copy['Margin_Percentage'] = (df_copy['BD MARGIN'] / df_copy['BROKER RATE (CFC)'] * 100).fillna(0)
    
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
    if df.empty or 'LOAD\'S CARRIER COMPANY' not in df.columns or 'BROKER RATE (CFC)' not in df.columns or 'FULL MILES TOTAL' not in df.columns:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Group by carrier and calculate metrics
    carrier_stats = df.groupby('LOAD\'S CARRIER COMPANY').agg({
        'BROKER RATE (CFC)': 'sum',
        'FULL MILES TOTAL': 'sum'
    }).reset_index()
    
    # Calculate revenue per mile
    carrier_stats['Revenue_Per_Mile'] = (carrier_stats['BROKER RATE (CFC)'] / carrier_stats['FULL MILES TOTAL']).fillna(0)
    
    # Sort by total billing
    carrier_stats = carrier_stats.sort_values('BROKER RATE (CFC)', ascending=False)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Billing by Carrier', 'Total Miles by Carrier', 'Revenue per Mile by Carrier', 'Carrier Efficiency Scatter'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Total Billing Bar Chart
    fig.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats['BROKER RATE (CFC)'],
            name='Total Billing',
            marker_color='#1f77b4',
            text=carrier_stats['BROKER RATE (CFC)'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # 2. Total Miles Bar Chart
    fig.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats['FULL MILES TOTAL'],
            name='Total Miles',
            marker_color='#ff7f0e',
            text=carrier_stats['FULL MILES TOTAL'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # 3. Revenue per Mile Bar Chart
    fig.add_trace(
        go.Bar(
            x=carrier_stats['LOAD\'S CARRIER COMPANY'],
            y=carrier_stats['Revenue_Per_Mile'],
            name='Revenue/Mile',
            marker_color='#2ca02c',
            text=carrier_stats['Revenue_Per_Mile'].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Efficiency Scatter Plot (Miles vs Revenue per Mile)
    fig.add_trace(
        go.Scatter(
            x=carrier_stats['FULL MILES TOTAL'],
            y=carrier_stats['Revenue_Per_Mile'],
            mode='markers+text',
            name='Efficiency',
            marker=dict(
                size=carrier_stats['BROKER RATE (CFC)'] / carrier_stats['BROKER RATE (CFC)'].max() * 20 + 10,
                color=carrier_stats['Revenue_Per_Mile'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue/Mile ($)")
            ),
            text=carrier_stats['LOAD\'S CARRIER COMPANY'],
            textposition='top center',
            textfont=dict(size=8)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Comprehensive Carrier Performance Analysis",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Carrier", row=1, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Total Billing ($)", row=1, col=1)
    
    fig.update_xaxes(title_text="Carrier", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Total Miles", row=1, col=2)
    
    fig.update_xaxes(title_text="Carrier", row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="Revenue per Mile ($)", row=2, col=1)
    
    fig.update_xaxes(title_text="Total Miles", row=2, col=2)
    fig.update_yaxes(title_text="Revenue per Mile ($)", row=2, col=2)
    
    return fig

def generate_chart_analysis(chart_type, df, chart_data=None):
    """Generate written analysis and insights for each chart type"""
    
    if df.empty:
        return "No data available for analysis."
    
    analysis = ""
    
    if chart_type == "carrier_billing":
        # Analysis for Total Billing per Carrier
        total_billing = df['BROKER RATE (CFC)'].sum()
        top_carrier = df.groupby('LOAD\'S CARRIER COMPANY')['BROKER RATE (CFC)'].sum().nlargest(1)
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
        total_billing = df['BROKER RATE (CFC)'].sum()
        top_dispatcher = df.groupby('DISPATCH NAME')['BROKER RATE (CFC)'].sum().nlargest(1)
        top_dispatcher_name = top_dispatcher.index[0]
        top_dispatcher_value = top_dispatcher.values[0]
        top_dispatcher_pct = (top_dispatcher_value / total_billing * 100)
        
        avg_billing_per_dispatcher = total_billing / len(df['DISPATCH NAME'].unique())
        
        analysis = f"""
        **📊 Total Billing per Dispatcher Analysis**
        
        **Key Findings:**
        - **Total Revenue**: ${total_billing:,.0f}
        - **Top Performer**: {top_dispatcher_name} (${top_dispatcher_value:,.0f}, {top_dispatcher_pct:.1f}% of total)
        - **Average per Dispatcher**: ${avg_billing_per_dispatcher:,.0f}
        - **Active Dispatchers**: {len(df['DISPATCH NAME'].unique())}
        
        **Business Insights:**
        - {top_dispatcher_name} is your highest-billing dispatcher
        - Performance gap analysis: Top dispatcher is {top_dispatcher_value/avg_billing_per_dispatcher:.1f}x above average
        - Consider training programs for underperforming dispatchers
        - Monitor workload distribution for optimal team performance
        """
    
    elif chart_type == "dispatcher_performance":
        # Analysis for Dispatcher Performance Matrix
        dispatcher_stats = df.groupby('DISPATCH NAME').agg({
            'BROKER RATE (CFC)': 'sum',
            'BD MARGIN': 'sum'
        }).reset_index()
        
        # Calculate efficiency metrics
        dispatcher_stats['Efficiency'] = (dispatcher_stats['BD MARGIN'] / dispatcher_stats['BROKER RATE (CFC)'] * 100).fillna(0)
        
        top_efficiency = dispatcher_stats.loc[dispatcher_stats['Efficiency'].idxmax()]
        top_billing = dispatcher_stats.loc[dispatcher_stats['BROKER RATE (CFC)'].idxmax()]
        
        analysis = f"""
        **📊 Dispatcher Performance Matrix Analysis**
        
        **Key Findings:**
        - **Most Efficient**: {top_efficiency['DISPATCH NAME']} ({top_efficiency['Efficiency']:.1f}% margin)
        - **Highest Billing**: {top_billing['DISPATCH NAME']} (${top_billing['BROKER RATE (CFC)']:,.0f})
        - **Performance Range**: {dispatcher_stats['Efficiency'].min():.1f}% to {dispatcher_stats['Efficiency'].max():.1f}% margin
        
        **Business Insights:**
        - {top_efficiency['DISPATCH NAME']} achieves the best margin efficiency
        - {top_billing['DISPATCH NAME']} generates the most revenue
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
        driver_earnings = df.groupby('DRIVER NAME')['DRIVER RATE'].sum().sort_values(ascending=False)
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
        trailer_billing = df.groupby('TRAILER TYPE')['BROKER RATE (CFC)'].sum().sort_values(ascending=False)
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
        if 'PICK UP DATE' in df.columns:
            df_copy = df.copy()
            df_copy['PICK UP DATE'] = pd.to_datetime(df_copy['PICK UP DATE'], errors='coerce')
            df_copy['Week'] = df_copy['PICK UP DATE'].dt.to_period('W')
            
            weekly_earnings = df_copy.groupby('Week')['DRIVER RATE'].mean()
            avg_weekly_earnings = weekly_earnings.mean()
            max_weekly_earnings = weekly_earnings.max()
            min_weekly_earnings = weekly_earnings.min()
            
            analysis = f"""
            **📊 Weekly Driver Earnings Analysis**
            
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
            analysis = "**📊 Weekly Driver Earnings Analysis**\n\nDate data not available for weekly analysis."
    
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
    
    elif chart_type == "driver_heatmap":
        # Analysis for Driver Miles Heatmap
        if 'PICK UP DATE' in df.columns:
            df_copy = df.copy()
            df_copy['PICK UP DATE'] = pd.to_datetime(df_copy['PICK UP DATE'], errors='coerce')
            df_copy['Week'] = df_copy['PICK UP DATE'].dt.to_period('W')
            
            driver_weekly = df_copy.groupby(['DRIVER NAME', 'Week'])['FULL MILES TOTAL'].sum()
            top_drivers = df_copy.groupby('DRIVER NAME')['FULL MILES TOTAL'].sum().nlargest(20)
            
            analysis = f"""
            **📊 Driver Miles Heatmap Analysis**
            
            **Key Findings:**
            - **Top 20 Drivers**: Analyzed for weekly mile patterns
            - **Time Period**: {len(df_copy['Week'].unique())} weeks of data
            - **Mile Distribution**: Visualized across drivers and time
            - **Consistency Patterns**: Identified through heatmap intensity
            
            **Business Insights:**
            - Heatmap reveals driver consistency and availability patterns
            - Darker areas indicate higher mile weeks
            - Identify drivers with consistent vs. variable performance
            - Use patterns for capacity planning and driver scheduling
            """
        else:
            analysis = "**📊 Driver Miles Heatmap Analysis**\n\nDate data not available for weekly heatmap analysis."
    
    elif chart_type == "margin_distribution":
        # Analysis for BD Margin Distribution
        df_copy = df.copy()
        df_copy['Margin_Percentage'] = (df_copy['BD MARGIN'] / df_copy['BROKER RATE (CFC)'] * 100).fillna(0)
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
        carrier_stats = df.groupby('LOAD\'S CARRIER COMPANY').agg({
            'BROKER RATE (CFC)': 'sum',
            'FULL MILES TOTAL': 'sum'
        }).reset_index()
        carrier_stats['Revenue_Per_Mile'] = (carrier_stats['BROKER RATE (CFC)'] / carrier_stats['FULL MILES TOTAL']).fillna(0)
        
        top_revenue = carrier_stats.loc[carrier_stats['BROKER RATE (CFC)'].idxmax()]
        top_miles = carrier_stats.loc[carrier_stats['FULL MILES TOTAL'].idxmax()]
        top_efficiency = carrier_stats.loc[carrier_stats['Revenue_Per_Mile'].idxmax()]
        
        analysis = f"""
        **📊 Comprehensive Carrier Performance Analysis**
        
        **Key Findings:**
        - **Highest Revenue**: {top_revenue["LOAD'S CARRIER COMPANY"]} (${top_revenue['BROKER RATE (CFC)']:,.0f})
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
    
    return analysis 