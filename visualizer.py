import matplotlib.pyplot as plt

def plot_dispatcher_billing(df):
    """Generates a bar chart for dispatcher billing."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dispatcher Billing - No Data')
        return fig
    
    # Determine which column to use for billing
    billing_col = 'BROKER RATE (FC) [$' if 'BROKER RATE (FC) [$' in df.columns else 'BROKER RATE'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['DISPATCH NAME'], df[billing_col])
    ax.set_title('Total Billing per Dispatcher')
    ax.set_xlabel('Dispatcher')
    ax.set_ylabel('Total Billing ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_cancellation_stats(df):
    """Generates a stacked bar chart for cancellation stats."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Load Status - No Data')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 7))
    # Get all columns except 'DISPATCH NAME'
    status_columns = [col for col in df.columns if col != 'DISPATCH NAME']
    if status_columns:
        df.plot(x='DISPATCH NAME', y=status_columns, kind='bar', stacked=True, ax=ax)
    ax.set_title('Load Status per Dispatcher')
    ax.set_xlabel('Dispatcher')
    ax.set_ylabel('Number of Loads')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_trailer_type_billing(df):
    """Generates a bar chart for trailer type billing."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trailer Type Billing - No Data')
        return fig
    
    # Determine which column to use for billing
    billing_col = 'BROKER RATE (FC) [$' if 'BROKER RATE (FC) [$' in df.columns else 'BROKER RATE'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['TRAILER TYPE'], df[billing_col])
    ax.set_title('Total Billing per Trailer Type')
    ax.set_xlabel('Trailer Type')
    ax.set_ylabel('Total Billing ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_top_drivers(df):
    """Generates a horizontal bar chart for top drivers."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Top Drivers - No Data')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df['DRIVER NAME'], df['DRIVER RATE [$]'])
    ax.set_title('Top 20 Drivers by Earnings')
    ax.set_xlabel('Total Pay ($)')
    ax.set_ylabel('Driver')
    plt.tight_layout()
    return fig

def plot_carrier_miles(df):
    """Generates a bar chart for miles per carrier."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Carrier Miles - No Data')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['LOAD\'S CARRIER COMPANY'], df['FULL MILES TOTAL'])
    ax.set_title('Total Miles per Carrier')
    ax.set_xlabel('Carrier')
    ax.set_ylabel('Total Miles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_broker_rate_cfc(df):
    """Generates a bar chart for BROKER RATE (FC) [$ per dispatcher."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('BROKER RATE (FC) [$ - No Data')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['DISPATCH NAME'], df['BROKER RATE (FC) [$'])
    ax.set_title('BROKER RATE (FC) [$ per Dispatcher')
    ax.set_xlabel('Dispatcher')
    ax.set_ylabel('BROKER RATE (FC) [$ ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_bd_margin(df):
    """Generates a bar chart for BD MARGIN per dispatcher."""
    if df.empty or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('BD MARGIN - No Data')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['DISPATCH NAME'], df['BD MARGIN'])
    ax.set_title('BD MARGIN per Dispatcher')
    ax.set_xlabel('Dispatcher')
    ax.set_ylabel('BD MARGIN ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig 