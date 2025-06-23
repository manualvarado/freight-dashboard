import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def get_dispatcher_billing(df):
    """Calculates total billing per dispatcher using BROKER RATE (CFC)."""
    # Use BROKER RATE (CFC) if available, otherwise fall back to BROKER RATE
    if 'BROKER RATE (CFC)' in df.columns:
        return df.groupby('DISPATCH NAME')['BROKER RATE (CFC)'].sum().reset_index()
    elif 'BROKER RATE' in df.columns:
        return df.groupby('DISPATCH NAME')['BROKER RATE'].sum().reset_index()
    else:
        # Return empty DataFrame if neither column exists
        return pd.DataFrame(columns=['DISPATCH NAME', 'BROKER RATE (CFC)'])

def get_cancellation_stats(df):
    """Calculates load count and cancellations per dispatcher."""
    # Using 'LOAD STATUS' and 'DISPATCH NAME' columns
    if 'LOAD STATUS' in df.columns and 'DISPATCH NAME' in df.columns:
        status_counts = df.groupby(['DISPATCH NAME', 'LOAD STATUS']).size().unstack(fill_value=0)
        return status_counts.reset_index()
    else:
        return pd.DataFrame()

def get_trailer_type_billing(df):
    """Calculates total billing per trailer type."""
    # Use BROKER RATE (CFC) if available, otherwise fall back to BROKER RATE
    if 'TRAILER TYPE' in df.columns:
        if 'BROKER RATE (CFC)' in df.columns:
            return df.groupby('TRAILER TYPE')['BROKER RATE (CFC)'].sum().reset_index()
        elif 'BROKER RATE' in df.columns:
            return df.groupby('TRAILER TYPE')['BROKER RATE'].sum().reset_index()
        else:
            return pd.DataFrame(columns=['TRAILER TYPE', 'BROKER RATE (CFC)'])
    else:
        return pd.DataFrame()

def get_top_drivers(df, n=20):
    """Calculates earnings for top N drivers using DRIVER RATE."""
    # Using 'DRIVER NAME' and 'DRIVER RATE' columns
    if 'DRIVER NAME' in df.columns and 'DRIVER RATE' in df.columns:
        return df.groupby('DRIVER NAME')['DRIVER RATE'].sum().nlargest(n).reset_index()
    else:
        return pd.DataFrame(columns=['DRIVER NAME', 'DRIVER RATE'])

def get_carrier_miles(df):
    """Calculates miles per carrier."""
    # Using 'LOAD\'S CARRIER COMPANY' and 'FULL MILES TOTAL' columns
    if 'LOAD\'S CARRIER COMPANY' in df.columns and 'FULL MILES TOTAL' in df.columns:
        return df.groupby('LOAD\'S CARRIER COMPANY')['FULL MILES TOTAL'].sum().reset_index()
    else:
        return pd.DataFrame()

def get_broker_rate_cfc_stats(df):
    """Calculates stats using BROKER RATE (CFC) column."""
    # This would be used for more detailed broker rate analysis
    if 'BROKER RATE (CFC)' in df.columns and 'DISPATCH NAME' in df.columns:
        return df.groupby('DISPATCH NAME')['BROKER RATE (CFC)'].sum().reset_index()
    else:
        return pd.DataFrame(columns=['DISPATCH NAME', 'BROKER RATE (CFC)'])

def get_bd_margin_stats(df):
    """Calculates stats using BD MARGIN column."""
    # This would be used for margin analysis
    if 'BD MARGIN' in df.columns and 'DISPATCH NAME' in df.columns:
        return df.groupby('DISPATCH NAME')['BD MARGIN'].sum().reset_index()
    else:
        return pd.DataFrame(columns=['DISPATCH NAME', 'BD MARGIN']) 