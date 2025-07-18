import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
import streamlit as st
import warnings
import base64
import os
warnings.filterwarnings('ignore')

class MonthlyOpsReview:
    def __init__(self, df):
        """
        Initialize the Monthly Operations Review dashboard.
        
        Args:
            df (pd.DataFrame): Input dataframe with freight data
        """
        self.df = df.copy()
        self.setup_data()
        
    def setup_data(self):
        """Prepare data for monthly analysis."""
        # Ensure DELIVERY DATE is datetime
        if 'DELIVERY DATE' in self.df.columns:
            self.df['DELIVERY DATE'] = pd.to_datetime(self.df['DELIVERY DATE'], errors='coerce')
        
        # Add month and year columns
        if 'DELIVERY DATE' in self.df.columns:
            self.df['Month'] = self.df['DELIVERY DATE'].dt.month
            self.df['Year'] = self.df['DELIVERY DATE'].dt.year
            self.df['Month_Name'] = self.df['DELIVERY DATE'].dt.strftime('%B')
            self.df['Month_Year'] = self.df['DELIVERY DATE'].dt.strftime('%Y-%m')
        
        # Parse numeric columns
        self.parse_numeric_columns()
        
        # Clean dispatcher names globally
        self.clean_dispatcher_names()
        
    def parse_numeric_columns(self):
        """Parse numeric columns that might have formatting issues."""
        numeric_columns = ['BROKER RATE (FC) [$', 'BROKER RATE', 'BD MARGIN', 'DRIVER RATE [$] [$]']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(',', '').str.replace('$', ''), 
                    errors='coerce'
                )
    
    def clean_dispatcher_names(self):
        """Clean dispatcher names to ensure they are valid strings."""
        if 'DISPATCH NAME' in self.df.columns:
            # Remove NaN values
            self.df = self.df.dropna(subset=['DISPATCH NAME'])
            
            # Convert to string and handle any remaining issues
            self.df['DISPATCH NAME'] = self.df['DISPATCH NAME'].astype(str)
            
            # Remove empty strings and 'nan' values
            self.df = self.df[self.df['DISPATCH NAME'].str.strip() != '']
            self.df = self.df[self.df['DISPATCH NAME'] != 'nan']
            self.df = self.df[self.df['DISPATCH NAME'] != 'None']
            
            # Reset index after cleaning
            self.df = self.df.reset_index(drop=True)
    
    def get_current_month_data(self):
        """Get data for the current month (assuming report is generated on 1st of month for previous month)."""
        current_date = datetime.now()
        report_month = current_date.replace(day=1) - timedelta(days=1)
        report_month = report_month.replace(day=1)
        
        current_month_data = self.df[
            (self.df['Year'] == report_month.year) & 
            (self.df['Month'] == report_month.month)
        ].copy()
        
        # Exclude CANCELLED, N/A loads, and Unpaid - Claim billing status
        if 'LOAD STATUS' in current_month_data.columns:
            current_month_data = current_month_data[
                ~current_month_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in current_month_data.columns:
            current_month_data = current_month_data[
                ~current_month_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        return current_month_data, report_month
    
    def get_yoy_comparison_data(self, current_month, current_year):
        """Get Year-over-Year comparison data."""
        # Current year data for the month
        current_year_data = self.df[
            (self.df['Year'] == current_year) & 
            (self.df['Month'] == current_month)
        ].copy()
        
        # Previous year data for the same month
        previous_year_data = self.df[
            (self.df['Year'] == current_year - 1) & 
            (self.df['Month'] == current_month)
        ].copy()
        
        # Exclude CANCELLED, N/A loads, and Unpaid - Claim billing status from both datasets
        if 'LOAD STATUS' in current_year_data.columns:
            current_year_data = current_year_data[
                ~current_year_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in current_year_data.columns:
            current_year_data = current_year_data[
                ~current_year_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        if 'LOAD STATUS' in previous_year_data.columns:
            previous_year_data = previous_year_data[
                ~previous_year_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in previous_year_data.columns:
            previous_year_data = previous_year_data[
                ~previous_year_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        return current_year_data, previous_year_data
    
    def plot_yoy_billing_margin_comparison(self):
        """1. YoY Billing & Margin Comparison - Side-by-side bar charts."""
        current_month_data, report_month = self.get_current_month_data()
        current_year_data, previous_year_data = self.get_yoy_comparison_data(
            report_month.month, report_month.year
        )
        
        # Calculate metrics for ALL dispatchers (not just common ones)
        current_billing = current_year_data['BROKER RATE (FC) [$'].sum() if 'BROKER RATE (FC) [$' in current_year_data.columns else 0
        current_margin = current_year_data['BD MARGIN'].sum() if 'BD MARGIN' in current_year_data.columns else 0
        current_loads = len(current_year_data)
        current_avg_per_load = current_billing / current_loads if current_loads > 0 else 0
        
        previous_billing = previous_year_data['BROKER RATE (FC) [$'].sum() if 'BROKER RATE (FC) [$' in previous_year_data.columns else 0
        previous_margin = previous_year_data['BD MARGIN'].sum() if 'BD MARGIN' in previous_year_data.columns else 0
        previous_loads = len(previous_year_data)
        previous_avg_per_load = previous_billing / previous_loads if previous_loads > 0 else 0
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Billing Comparison', 'Margin Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Billing comparison with average per load
        fig.add_trace(
            go.Bar(
                x=[f'{report_month.year-1}', f'{report_month.year}'],
                y=[previous_billing, current_billing],
                name='Billing',
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[
                    f'${previous_billing:,.0f} ({previous_loads} loads)<br>Avg. $/Load: ${previous_avg_per_load:,.2f}', 
                    f'${current_billing:,.0f} ({current_loads} loads)<br>Avg. $/Load: ${current_avg_per_load:,.2f}'
                ],
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Margin comparison
        fig.add_trace(
            go.Bar(
                x=[f'{report_month.year-1}', f'{report_month.year}'],
                y=[previous_margin, current_margin],
                name='Margin',
                marker_color=['#FFE66D', '#95E1D3'],
                text=[f'${previous_margin:,.0f}', f'${current_margin:,.0f}'],
                textposition='outside',
                textfont=dict(size=10),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Year-over-Year Comparison: {report_month.strftime("%B")} {report_month.year} vs {report_month.strftime("%B")} {report_month.year-1}',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Billing ($)", row=1, col=1)
        fig.update_yaxes(title_text="Margin ($)", row=1, col=2)
        
        return fig
    
    def get_dispatcher_evolution_data(self):
        """Get dispatcher evolution data for filtering purposes."""
        current_month_data, report_month = self.get_current_month_data()
        
        # Filter data for current year up to report month
        evolution_data = self.df[
            (self.df['Year'] == report_month.year) & 
            (self.df['Month'] <= report_month.month)
        ].copy()
        
        # Exclude CANCELLED, N/A loads, and Unpaid - Claim billing status
        if 'LOAD STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        if evolution_data.empty:
            return pd.DataFrame(), report_month
        
        # Group by dispatcher and month
        monthly_stats = evolution_data.groupby(['DISPATCH NAME', 'Month']).agg({
            'BROKER RATE (FC) [$': 'sum',
            'BD MARGIN': 'sum'
        }).reset_index()
        
        return monthly_stats, report_month
    
    def plot_dispatcher_monthly_evolution(self, selected_dispatchers=None):
        """2. Dispatcher Monthly Billing and Margin Evolution (Jan–Current Month) with grouped+stacked bars."""
        monthly_stats, report_month = self.get_dispatcher_evolution_data()
        
        if monthly_stats.empty:
            return go.Figure().add_annotation(
                text="No data available for monthly evolution", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        all_dispatchers = monthly_stats['DISPATCH NAME'].unique().tolist()
        if selected_dispatchers is None:
            dispatchers_to_show = all_dispatchers
        else:
            dispatchers_to_show = [d for d in selected_dispatchers if d in all_dispatchers]
        filtered_stats = monthly_stats[monthly_stats['DISPATCH NAME'].isin(dispatchers_to_show)].copy()
        months = sorted(filtered_stats['Month'].unique())
        colors = px.colors.qualitative.Set3
        fig = go.Figure()
        
        for i, dispatcher in enumerate(dispatchers_to_show):
            dispatcher_data = filtered_stats[filtered_stats['DISPATCH NAME'] == dispatcher].copy()
            if not dispatcher_data.empty:
                dispatcher_data['BillingMinusMargin'] = dispatcher_data['BROKER RATE (FC) [$'] - dispatcher_data['BD MARGIN']
                # X-axis: month, but use offsetgroup to group bars by dispatcher
                fig.add_trace(
                    go.Bar(
                        x=dispatcher_data['Month'],
                        y=dispatcher_data['BD MARGIN'],
                        name=f'{dispatcher} - Margin',
                        marker_color='red',
                        text=dispatcher_data['BD MARGIN'].apply(lambda x: f'${x:,.0f}'),
                        textposition='inside',
                        textfont=dict(size=10),
                        opacity=0.9,
                        showlegend=(i==0),
                        offsetgroup=dispatcher,
                        base=None
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=dispatcher_data['Month'],
                        y=dispatcher_data['BillingMinusMargin'],
                        name=f'{dispatcher} - Billing',
                        marker_color=colors[i % len(colors)],
                        text=dispatcher_data['BROKER RATE (FC) [$'].apply(lambda x: f'${x:,.0f}'),
                        textposition='outside',
                        textfont=dict(size=10),
                        opacity=0.8,
                        showlegend=True,
                        offsetgroup=dispatcher,
                        base=None
                    )
                )
        fig.update_layout(
            title=f'Dispatcher Monthly Evolution: January - {report_month.strftime("%B %Y")}<br><sub>Showing {len(dispatchers_to_show)} of {len(all_dispatchers)} dispatchers</sub>',
            height=600,
            template='plotly_white',
            barmode='relative',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Month",
                ticktext=[calendar.month_abbr[m] for m in months],
                tickvals=months
            ),
            yaxis=dict(title="Amount ($)")
        )
        return fig
    
    def plot_dispatcher_billing_vs_load_volume(self):
        """3. Dispatcher Billing vs Load Volume – Current Month."""
        current_month_data, report_month = self.get_current_month_data()
        
        if current_month_data.empty:
            return go.Figure().add_annotation(
                text="No data available for current month", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Group by dispatcher
        dispatcher_stats = current_month_data.groupby('DISPATCH NAME').agg({
            'BROKER RATE (FC) [$': 'sum',
            'BD MARGIN': 'sum'
        }).reset_index()
        
        # Add load count
        load_counts = current_month_data.groupby('DISPATCH NAME').size().reset_index(name='Load Count')
        dispatcher_stats = dispatcher_stats.merge(load_counts, on='DISPATCH NAME')
        
        # Sort by billing
        dispatcher_stats = dispatcher_stats.sort_values('BROKER RATE (FC) [$', ascending=False)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add BD Margin bars (stacked under billing)
        fig.add_trace(
            go.Bar(
                x=dispatcher_stats['DISPATCH NAME'],
                y=dispatcher_stats['BD MARGIN'],
                name='BD Margin',
                marker_color='#C70039',
                text=dispatcher_stats['BD MARGIN'].apply(lambda x: f'${x:,.0f}'),
                textposition='inside',
                textfont=dict(color='white', size=10),
                yaxis='y'
            )
        )
        
        # Add billing bars (remaining amount after BD Margin)
        remaining_billing = dispatcher_stats['BROKER RATE (FC) [$'] - dispatcher_stats['BD MARGIN']
        fig.add_trace(
            go.Bar(
                x=dispatcher_stats['DISPATCH NAME'],
                y=remaining_billing,
                name='Billing (Net)',
                marker_color='#FFC300',
                text=dispatcher_stats['BROKER RATE (FC) [$'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                textfont=dict(size=10),
                yaxis='y'
            )
        )
        
        # Add load volume line
        fig.add_trace(
            go.Scatter(
                x=dispatcher_stats['DISPATCH NAME'],
                y=dispatcher_stats['Load Count'],
                mode='lines+markers',
                name='Load Count',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                text=dispatcher_stats['Load Count'],
                textposition='top center',
                yaxis='y2'
            )
        )
        
        fig.update_layout(
            title=f'Dispatcher Billing vs Load Volume: {report_month.strftime("%B %Y")}',
            height=500,
            yaxis=dict(title="Billing ($)", side="left"),
            yaxis2=dict(title="Load Count", side="right", overlaying="y"),
            template='plotly_white',
            showlegend=True,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_load_carrier_monthly_evolution(self):
        """4. Load Carrier Monthly Billing Evolution (Jan–Current Month)."""
        current_month_data, report_month = self.get_current_month_data()
        
        # Filter data for current year up to report month
        evolution_data = self.df[
            (self.df['Year'] == report_month.year) & 
            (self.df['Month'] <= report_month.month)
        ].copy()
        
        # Exclude CANCELLED, N/A loads, and Unpaid - Claim billing status
        if 'LOAD STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        if evolution_data.empty or 'LOAD\'S CARRIER COMPANY' not in evolution_data.columns:
            return go.Figure().add_annotation(
                text="No data available for load carrier evolution", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Group by load carrier and month
        monthly_stats = evolution_data.groupby(['LOAD\'S CARRIER COMPANY', 'Month']).agg({
            'BROKER RATE (FC) [$': 'sum'
        }).reset_index()
        
        # Get top 10 carriers by total billing
        top_carriers = evolution_data.groupby('LOAD\'S CARRIER COMPANY')['BROKER RATE (FC) [$'].sum().nlargest(10).index
        
        # Filter for top carriers
        monthly_stats = monthly_stats[monthly_stats['LOAD\'S CARRIER COMPANY'].isin(top_carriers)]
        
        # Calculate total billing per month for annotations
        monthly_totals = evolution_data.groupby('Month')['BROKER RATE (FC) [$'].sum().reset_index()
        monthly_totals = monthly_totals.rename(columns={'BROKER RATE (FC) [$': 'Total_Billing'})
        
        # Calculate total billing up to date (sum of all months)
        total_billing_up_to_date = monthly_totals['Total_Billing'].sum()
        
        # Create stacked bar chart
        fig = px.bar(
            monthly_stats,
            x='Month',
            y='BROKER RATE (FC) [$',
            color='LOAD\'S CARRIER COMPANY',
            title=f'Load Carrier Monthly Billing Evolution: January - {report_month.strftime("%B %Y")}',
            labels={'BROKER RATE (FC) [$': 'Billing ($)', 'LOAD\'S CARRIER COMPANY': 'Load Carrier'},
            barmode='stack'
        )
        
        # Add total billing annotations on top of each bar
        for month in monthly_totals['Month']:
            total_billing = monthly_totals[monthly_totals['Month'] == month]['Total_Billing'].iloc[0]
            fig.add_annotation(
                x=month,
                y=total_billing,
                text=f'${total_billing:,.0f}',
                showarrow=False,
                font=dict(color='white', size=12, weight='bold'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1,
                yshift=10
            )
        
        # Add total billing up to date annotation below the X-axis
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref='paper',
            yref='paper',
            text=f'<b>Total Billing Up to Date: ${total_billing_up_to_date:,.0f}</b>',
            showarrow=False,
            font=dict(color='white', size=14, weight='bold'),
            bgcolor='black',
            align='center'
        )
        
        fig.update_layout(
            height=550,  # Increased height to accommodate the annotation
            template='plotly_white',
            xaxis=dict(
                ticktext=[calendar.month_abbr[m] for m in sorted(monthly_stats['Month'].unique())],
                tickvals=sorted(monthly_stats['Month'].unique())
            ),
            margin=dict(b=80)  # Add bottom margin for the annotation
        )
        
        return fig
    
    def plot_load_carrier_weekly_evolution(self):
        """4.5. Load Carrier Weekly Billing Evolution (Current Month)."""
        current_month_data, report_month = self.get_current_month_data()
        
        if current_month_data.empty or 'LOAD\'S CARRIER COMPANY' not in current_month_data.columns:
            return go.Figure().add_annotation(
                text="No data available for load carrier weekly evolution", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Define week boundaries (Tuesday to Monday)
        def get_week_start_end(date):
            """Get the Tuesday-Monday week boundaries for a given date."""
            # Find the most recent Tuesday (week starts on Tuesday)
            days_since_tuesday = (date.weekday() - 1) % 7  # Tuesday is 1
            week_start = date - timedelta(days=days_since_tuesday)
            week_end = week_start + timedelta(days=6)  # Monday is 6 days after Tuesday
            return week_start, week_end
        
        # Get the first and last day of the current month
        first_day = report_month.replace(day=1)
        last_day = (report_month.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        # Get the first Tuesday of the month (or the Tuesday of the week containing the first day)
        first_week_start, first_week_end = get_week_start_end(first_day)
        
        # Get the last Monday of the month (or the Monday of the week containing the last day)
        last_week_start, last_week_end = get_week_start_end(last_day)
        
        # Create weekly periods for the current month
        weekly_periods = []
        current_week_start = first_week_start
        
        while current_week_start <= last_week_end:
            current_week_end = current_week_start + timedelta(days=6)
            weekly_periods.append({
                'week_start': current_week_start,
                'week_end': current_week_end,
                'week_label': f"{current_week_start.strftime('%m/%d')} - {current_week_end.strftime('%m/%d')}"
            })
            current_week_start += timedelta(days=7)
        
        # Filter data for loads delivered within the weekly periods
        weekly_data = []
        for period in weekly_periods:
            week_data = current_month_data[
                (current_month_data['DELIVERY DATE'] >= period['week_start']) &
                (current_month_data['DELIVERY DATE'] <= period['week_end'])
            ].copy()
            
            if not week_data.empty:
                # Group by load carrier for this week
                week_stats = week_data.groupby('LOAD\'S CARRIER COMPANY').agg({
                    'BROKER RATE (FC) [$': 'sum'
                }).reset_index()
                week_stats['week_label'] = period['week_label']
                week_stats['week_start'] = period['week_start']
                weekly_data.append(week_stats)
        
        if not weekly_data:
            return go.Figure().add_annotation(
                text="No data available for the specified weekly periods", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Combine all weekly data
        all_weekly_data = pd.concat(weekly_data, ignore_index=True)
        
        # Get top 8 carriers by total billing across all weeks
        top_carriers = all_weekly_data.groupby('LOAD\'S CARRIER COMPANY')['BROKER RATE (FC) [$'].sum().nlargest(8).index
        
        # Filter for top carriers
        filtered_weekly_data = all_weekly_data[all_weekly_data['LOAD\'S CARRIER COMPANY'].isin(top_carriers)]
        
        # Calculate total billing per week for annotations
        weekly_totals = all_weekly_data.groupby('week_label')['BROKER RATE (FC) [$'].sum().reset_index()
        weekly_totals = weekly_totals.rename(columns={'BROKER RATE (FC) [$': 'Total_Billing'})
        
        # Create stacked bar chart
        fig = px.bar(
            filtered_weekly_data,
            x='week_label',
            y='BROKER RATE (FC) [$',
            color='LOAD\'S CARRIER COMPANY',
            title=f'Load Carrier Weekly Billing Evolution: {report_month.strftime("%B %Y")} (Tuesday-Monday Weeks)',
            labels={'BROKER RATE (FC) [$': 'Billing ($)', 'LOAD\'S CARRIER COMPANY': 'Load Carrier', 'week_label': 'Week Period'},
            barmode='stack'
        )
        
        # Add total billing annotations on top of each bar
        for week_label in weekly_totals['week_label']:
            total_billing = weekly_totals[weekly_totals['week_label'] == week_label]['Total_Billing'].iloc[0]
            fig.add_annotation(
                x=week_label,
                y=total_billing,
                text=f'${total_billing:,.0f}',
                showarrow=False,
                font=dict(color='white', size=10, weight='bold'),
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor='white',
                borderwidth=1,
                yshift=8
            )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            xaxis=dict(
                title="Week Period (Tuesday-Monday)",
                tickangle=45
            ),
            yaxis=dict(title="Billing ($)")
        )
        
        return fig
    
    def plot_franchise_carrier_monthly_evolution(self):
        """5. Franchise Carrier Monthly Billing Evolution (Jan–Current Month)."""
        current_month_data, report_month = self.get_current_month_data()
        
        # Filter data for current year up to report month
        evolution_data = self.df[
            (self.df['Year'] == report_month.year) & 
            (self.df['Month'] <= report_month.month)
        ].copy()
        
        # Exclude CANCELLED, N/A loads, and Unpaid - Claim billing status
        if 'LOAD STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['LOAD STATUS'].str.upper().isin(['CANCELLED', 'N/A', 'NA'])
            ].copy()
        
        if 'BILLING STATUS' in evolution_data.columns:
            evolution_data = evolution_data[
                ~evolution_data['BILLING STATUS'].str.contains('Unpaid - Claim', case=False, na=False)
            ].copy()
        
        if evolution_data.empty or 'DRIVER\'S CARRIER COMPANY' not in evolution_data.columns:
            return go.Figure().add_annotation(
                text="No data available for franchise carrier evolution", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Group by franchise carrier and month
        monthly_stats = evolution_data.groupby(['DRIVER\'S CARRIER COMPANY', 'Month']).agg({
            'BROKER RATE (FC) [$': 'sum'
        }).reset_index()
        
        # Get top 10 carriers by total billing
        top_carriers = evolution_data.groupby('DRIVER\'S CARRIER COMPANY')['BROKER RATE (FC) [$'].sum().nlargest(10).index
        
        # Filter for top carriers
        monthly_stats = monthly_stats[monthly_stats['DRIVER\'S CARRIER COMPANY'].isin(top_carriers)]
        
        # Create grouped bar chart with consistent colors
        fig = px.bar(
            monthly_stats,
            x='Month',
            y='BROKER RATE (FC) [$',
            color='DRIVER\'S CARRIER COMPANY',
            title=f'Franchise Carrier Monthly Billing Evolution: January - {report_month.strftime("%B %Y")}',
            labels={'BROKER RATE (FC) [$': 'Billing ($)', 'DRIVER\'S CARRIER COMPANY': 'Franchise Carrier'},
            barmode='group'
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            xaxis=dict(
                ticktext=[calendar.month_abbr[m] for m in sorted(monthly_stats['Month'].unique())],
                tickvals=sorted(monthly_stats['Month'].unique())
            )
        )
        
        return fig
    
    def plot_trailer_type_performance(self):
        """6. Trailer Type Performance – Current Month."""
        current_month_data, report_month = self.get_current_month_data()
        
        if current_month_data.empty or 'TRAILER TYPE' not in current_month_data.columns:
            return go.Figure().add_annotation(
                text="No data available for trailer type performance", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Group by trailer type
        trailer_stats = current_month_data.groupby('TRAILER TYPE').agg({
            'BROKER RATE (FC) [$': 'sum'
        }).reset_index()
        
        # Add load count
        load_counts = current_month_data.groupby('TRAILER TYPE').size().reset_index(name='Load Count')
        trailer_stats = trailer_stats.merge(load_counts, on='TRAILER TYPE')
        
        # Sort by billing
        trailer_stats = trailer_stats.sort_values('BROKER RATE (FC) [$', ascending=False)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add billing bars
        fig.add_trace(
            go.Bar(
                x=trailer_stats['TRAILER TYPE'],
                y=trailer_stats['BROKER RATE (FC) [$'],
                name='Billing',
                marker_color='lightgreen',
                text=trailer_stats['BROKER RATE (FC) [$'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                yaxis='y'
            )
        )
        
        # Add load count line
        fig.add_trace(
            go.Scatter(
                x=trailer_stats['TRAILER TYPE'],
                y=trailer_stats['Load Count'],
                mode='lines+markers',
                name='Load Count',
                line=dict(color='orange', width=3),
                marker=dict(size=8),
                text=trailer_stats['Load Count'],
                textposition='top center',
                yaxis='y2'
            )
        )
        
        fig.update_layout(
            title=f'Trailer Type Performance: {report_month.strftime("%B %Y")}',
            height=500,
            yaxis=dict(title="Billing ($)", side="left"),
            yaxis2=dict(title="Load Count", side="right", overlaying="y"),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def generate_monthly_summary(self):
        """Generate a comprehensive monthly summary with trends."""
        current_month_data, report_month = self.get_current_month_data()
        
        if current_month_data.empty:
            return "No data available for the current month."
        
        # Calculate key metrics
        total_billing = current_month_data['BROKER RATE (FC) [$'].sum() if 'BROKER RATE (FC) [$' in current_month_data.columns else 0
        total_margin = current_month_data['BD MARGIN'].sum() if 'BD MARGIN' in current_month_data.columns else 0
        total_loads = len(current_month_data)
        margin_percentage = (total_margin / total_billing * 100) if total_billing > 0 else 0
        
        # Get previous month for comparison
        prev_month = report_month - timedelta(days=1)
        prev_month = prev_month.replace(day=1)
        
        prev_month_data = self.df[
            (self.df['Year'] == prev_month.year) & 
            (self.df['Month'] == prev_month.month)
        ].copy()
        
        prev_billing = prev_month_data['BROKER RATE (FC) [$'].sum() if 'BROKER RATE (FC) [$' in prev_month_data.columns else 0
        prev_margin = prev_month_data['BD MARGIN'].sum() if 'BD MARGIN' in prev_month_data.columns else 0
        prev_loads = len(prev_month_data)
        
        # Calculate month-over-month changes
        billing_change = ((total_billing - prev_billing) / prev_billing * 100) if prev_billing > 0 else 0
        margin_change = ((total_margin - prev_margin) / prev_margin * 100) if prev_margin > 0 else 0
        load_change = ((total_loads - prev_loads) / prev_loads * 100) if prev_loads > 0 else 0
        
        # Top performers
        top_dispatcher = current_month_data.groupby('DISPATCH NAME')['BROKER RATE (FC) [$'].sum().idxmax() if 'DISPATCH NAME' in current_month_data.columns else "N/A"
        top_carrier = current_month_data.groupby('LOAD\'S CARRIER COMPANY')['BROKER RATE (FC) [$'].sum().idxmax() if 'LOAD\'S CARRIER COMPANY' in current_month_data.columns else "N/A"
        
        summary = f"""
## 📊 Monthly Operations Review Summary: {report_month.strftime("%B %Y")}

### 🎯 Key Performance Indicators
- **Total Billing**: ${total_billing:,.2f} ({billing_change:+.1f}% vs previous month)
- **Total Margin**: ${total_margin:,.2f} ({margin_change:+.1f}% vs previous month)
- **Margin Percentage**: {margin_percentage:.1f}%
- **Total Loads**: {total_loads:,} ({load_change:+.1f}% vs previous month)

### 🏆 Top Performers
- **Top Dispatcher**: {top_dispatcher}
- **Top Load Carrier**: {top_carrier}

### 📈 Month-over-Month Trends
- **Billing Trend**: {'📈 Increasing' if billing_change > 0 else '📉 Decreasing' if billing_change < 0 else '➡️ Stable'}
- **Margin Trend**: {'📈 Increasing' if margin_change > 0 else '📉 Decreasing' if margin_change < 0 else '➡️ Stable'}
- **Load Volume Trend**: {'📈 Increasing' if load_change > 0 else '📉 Decreasing' if load_change < 0 else '➡️ Stable'}

### 🔍 Strategic Insights
- **Margin Health**: {'🟢 Strong' if margin_percentage > 15 else '🟡 Moderate' if margin_percentage > 10 else '🔴 Needs Attention'}
- **Operational Efficiency**: {'🟢 High' if total_loads > 100 else '🟡 Moderate' if total_loads > 50 else '🔴 Low'}
        """
        
        return summary
    
    def generate_all_charts(self, selected_dispatchers=None):
        """Generate all charts for the monthly operations review."""
        charts = {
            'yoy_comparison': self.plot_yoy_billing_margin_comparison(),
            'dispatcher_evolution': self.plot_dispatcher_monthly_evolution(selected_dispatchers),
            'dispatcher_billing_loads': self.plot_dispatcher_billing_vs_load_volume(),
            'load_carrier_evolution': self.plot_load_carrier_monthly_evolution(),
            'load_carrier_weekly_evolution': self.plot_load_carrier_weekly_evolution(),
            'franchise_carrier_evolution': self.plot_franchise_carrier_monthly_evolution(),
            'trailer_performance': self.plot_trailer_type_performance()
        }
        
        return charts
    
    def save_charts_as_png(self, output_dir='monthly_reports'):
        """Save all charts as PNG files."""
        os.makedirs(output_dir, exist_ok=True)
        
        current_month_data, report_month = self.get_current_month_data()
        report_date = report_month.strftime("%Y_%m")
        
        charts = self.generate_all_charts()
        
        for chart_name, fig in charts.items():
            filename = f"{output_dir}/monthly_ops_review_{chart_name}_{report_date}.png"
            fig.write_image(filename, width=1200, height=800)
        
        return f"Charts saved to {output_dir}/"
    
    def generate_html_dashboard(self, output_file='monthly_ops_dashboard.html'):
        """Generate a complete HTML dashboard."""
        charts = self.generate_all_charts()
        summary = self.generate_monthly_summary()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Monthly Operations Review Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container {{ background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-title {{ color: #2c3e50; font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
        h1 {{ margin: 0; }}
        h2 {{ color: #34495e; }}
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <a href="http://localhost:8501/" target="_self" style="font-size: 1.1em; text-decoration: none; color: #007bff; font-weight: bold;">🔄 Switch to Weekly Dashboard</a>
            <img src="jc_logo.png" alt="JC Global Logo" style="height: 50px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
        </div>
        <h1>📊 Monthly Operations Review Dashboard</h1>
        <p>Comprehensive freight operations analysis and performance metrics</p>
    </div>
    
    <div class="summary">
        {summary.replace(chr(10), '<br>')}
    </div>
"""
        
        # Add each chart
        chart_titles = {
            'yoy_comparison': '📊 Year-over-Year Billing & Margin Comparison',
            'dispatcher_evolution': '📈 Dispatcher Monthly Billing and Margin Evolution',
            'dispatcher_billing_loads': '📉 Dispatcher Billing vs Load Volume',
            'load_carrier_evolution': '📦 Load Carrier Monthly Billing Evolution',
            'load_carrier_weekly_evolution': '📅 Load Carrier Weekly Billing Evolution',
            'franchise_carrier_evolution': '🚛 Franchise Carrier Monthly Billing Evolution',
            'trailer_performance': '🔄 Trailer Type Performance Analysis'
        }
        
        for chart_name, fig in charts.items():
            html_content += f"""
    <div class="chart-container">
        <div class="chart-title">{chart_titles[chart_name]}</div>
        <div id="chart_{chart_name}"></div>
    </div>
"""
        
        html_content += """
</body>
<script>
"""
        
        # Add JavaScript for each chart
        for chart_name, fig in charts.items():
            html_content += f"""
    var chart_{chart_name} = {fig.to_json()};
    Plotly.newPlot('chart_{chart_name}', chart_{chart_name}.data, chart_{chart_name}.layout);
"""
        
        html_content += """
</script>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return f"HTML dashboard saved as {output_file}"


def get_logo_base64():
    """Get base64 encoded logo for display."""
    logo_path = "jc_logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def create_monthly_ops_review_app():
    """Create a Streamlit app for the Monthly Operations Review."""
    st.set_page_config(
        page_title="Monthly Operations Review",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Display header with logo
    logo_base64 = get_logo_base64()
    if logo_base64:
        st.markdown(
            f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <a href="http://localhost:8501/" target="_self" style="font-size: 1.1em; text-decoration: none; color: #007bff; font-weight: bold;">🔄 Switch to Weekly Dashboard</a>
                <img src="data:image/png;base64,{logo_base64}" alt="JC Global Logo" style="height: 50px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"/>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <a href="http://localhost:8501/" target="_self" style="font-size: 1.1em; text-decoration: none; color: #007bff; font-weight: bold;">🔄 Switch to Weekly Dashboard</a>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 8px; font-weight: bold; font-size: 20px; letter-spacing: 2px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">JC</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    st.title("📊 Monthly Operations Review Dashboard")
    st.markdown("Comprehensive freight operations analysis with YoY comparisons and monthly evolution tracking")

    # File upload in sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your freight data CSV file",
        type=['csv'],
        help="Upload a CSV file with columns like DISPATCH NAME, BROKER RATE (FC) [$, BD MARGIN, etc."
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
            # Initialize dashboard
            dashboard = MonthlyOpsReview(df)
            
            # Get dispatcher data for filtering
            dispatcher_data, report_month = dashboard.get_dispatcher_evolution_data()
            all_dispatchers = dispatcher_data['DISPATCH NAME'].unique().tolist() if not dispatcher_data.empty else []
            
            # Display summary
            st.markdown("---")
            st.markdown("## 📋 Executive Summary")
            summary = dashboard.generate_monthly_summary()
            st.markdown(summary)
            
            # Generate charts
            st.markdown("---")
            st.markdown("## 📈 Performance Analytics")
            
            # YoY Comparison
            st.markdown("### 📊 Year-over-Year Billing & Margin Comparison")
            yoy_fig = dashboard.plot_yoy_billing_margin_comparison()
            st.plotly_chart(yoy_fig, use_container_width=True)
            
            # Dispatcher Evolution with filter
            st.markdown("### 📈 Dispatcher Monthly Billing and Margin Evolution")
            
            # Dispatcher filter
            if all_dispatchers:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_dispatchers = st.multiselect(
                        "Select dispatchers to display (leave empty to show all):",
                        options=all_dispatchers,
                        default=all_dispatchers[:min(8, len(all_dispatchers))],  # Default to first 8
                        help="Choose which dispatchers to include in the evolution chart"
                    )
                with col2:
                    if st.button("Show All"):
                        selected_dispatchers = all_dispatchers
                    if st.button("Clear All"):
                        selected_dispatchers = []
                
                dispatcher_evolution_fig = dashboard.plot_dispatcher_monthly_evolution(selected_dispatchers)
                st.plotly_chart(dispatcher_evolution_fig, use_container_width=True)
            else:
                st.info("No dispatcher data available for evolution chart")
                dispatcher_evolution_fig = dashboard.plot_dispatcher_monthly_evolution()
                st.plotly_chart(dispatcher_evolution_fig, use_container_width=True)
            
            # Dispatcher Billing vs Loads
            st.markdown("### 📉 Dispatcher Billing vs Load Volume")
            dispatcher_loads_fig = dashboard.plot_dispatcher_billing_vs_load_volume()
            st.plotly_chart(dispatcher_loads_fig, use_container_width=True)
            
            # Load Carrier Evolution
            st.markdown("### 📦 Load Carrier Monthly Billing Evolution")
            load_carrier_fig = dashboard.plot_load_carrier_monthly_evolution()
            st.plotly_chart(load_carrier_fig, use_container_width=True)
            
            # Load Carrier Weekly Evolution
            st.markdown("### 📦 Load Carrier Weekly Billing Evolution")
            load_carrier_weekly_fig = dashboard.plot_load_carrier_weekly_evolution()
            st.plotly_chart(load_carrier_weekly_fig, use_container_width=True)
            
            # Franchise Carrier Evolution
            st.markdown("### 🚛 Franchise Carrier Monthly Billing Evolution")
            franchise_carrier_fig = dashboard.plot_franchise_carrier_monthly_evolution()
            st.plotly_chart(franchise_carrier_fig, use_container_width=True)
            
            # Trailer Performance
            st.markdown("### 🔄 Trailer Type Performance Analysis")
            trailer_fig = dashboard.plot_trailer_type_performance()
            st.plotly_chart(trailer_fig, use_container_width=True)
            
            # Export options
            st.markdown("---")
            st.markdown("## 💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export All Charts as PNG"):
                    result = dashboard.save_charts_as_png()
                    st.success(result)
            
            with col2:
                if st.button("🌐 Generate HTML Dashboard"):
                    result = dashboard.generate_html_dashboard()
                    st.success(result)
            
            with col3:
                if st.button("📄 Download Summary Report"):
                    # Create a text report
                    report_text = dashboard.generate_monthly_summary()
                    st.download_button(
                        label="📄 Download Report",
                        data=report_text,
                        file_name=f"monthly_ops_summary_{datetime.now().strftime('%Y_%m')}.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns: DISPATCH NAME, BROKER RATE (FC) [$, BD MARGIN, DELIVERY DATE, etc.")
    else:
        st.info("👆 Please upload a CSV file using the sidebar to begin analysis.")
        st.markdown("""
        ### 📋 Expected CSV Format
        The application expects CSV files with the following columns:
        - `DISPATCH NAME`
        - `BROKER RATE (FC) [$`
        - `BD MARGIN`
        - `DELIVERY DATE`
        - `LOAD'S CARRIER COMPANY`
        - `DRIVER'S CARRIER COMPANY`
        - `TRAILER TYPE`
        - `LOAD ID`
        """)


if __name__ == "__main__":
    create_monthly_ops_review_app()
