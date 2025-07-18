import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def get_dispatcher_billing(df):
    """Calculates total billing per dispatcher using dynamic column detection."""
    # Find dispatcher and broker rate columns
    dispatcher_col = None
    broker_rate_col = None
    
    # Find dispatcher column
    if 'DISPATCH NAME' in df.columns:
        dispatcher_col = 'DISPATCH NAME'
    elif 'FC NAME' in df.columns:
        dispatcher_col = 'FC NAME'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'dispatch' in col.lower() or 'fc' in col.lower():
                dispatcher_col = col
                break
    
    # Find broker rate column
    if 'BROKER RATE (FC) [$' in df.columns:
        broker_rate_col = 'BROKER RATE (FC) [$'
    elif 'BROKER RATE (CFC)' in df.columns:
        broker_rate_col = 'BROKER RATE (CFC)'
    elif 'BROKER RATE' in df.columns:
        broker_rate_col = 'BROKER RATE'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'broker' in col.lower() and 'rate' in col.lower():
                broker_rate_col = col
                break
    
    if dispatcher_col and broker_rate_col:
        return df.groupby(dispatcher_col)[broker_rate_col].sum().reset_index()
    else:
        # Return empty DataFrame if columns not found
        return pd.DataFrame(columns=[dispatcher_col or 'Dispatcher', broker_rate_col or 'Broker Rate'])

def get_cancellation_stats(df):
    """Calculates load count and cancellations per dispatcher."""
    # Find dispatcher column
    dispatcher_col = None
    if 'DISPATCH NAME' in df.columns:
        dispatcher_col = 'DISPATCH NAME'
    elif 'FC NAME' in df.columns:
        dispatcher_col = 'FC NAME'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'dispatch' in col.lower() or 'fc' in col.lower():
                dispatcher_col = col
                break
    
    # Using 'LOAD STATUS' and dispatcher columns
    if 'LOAD STATUS' in df.columns and dispatcher_col:
        status_counts = df.groupby([dispatcher_col, 'LOAD STATUS']).size().unstack(fill_value=0)
        return status_counts.reset_index()
    else:
        return pd.DataFrame()

def get_trailer_type_billing(df):
    """Calculates total billing per trailer type."""
    # Find trailer column
    trailer_col = None
    if 'TRAILER TYPE' in df.columns:
        trailer_col = 'TRAILER TYPE'
    elif 'TRAILER' in df.columns:
        trailer_col = 'TRAILER'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'trailer' in col.lower():
                trailer_col = col
                break
    
    # Find broker rate column
    broker_rate_col = None
    if 'BROKER RATE (FC) [$' in df.columns:
        broker_rate_col = 'BROKER RATE (FC) [$'
    elif 'BROKER RATE (CFC)' in df.columns:
        broker_rate_col = 'BROKER RATE (CFC)'
    elif 'BROKER RATE' in df.columns:
        broker_rate_col = 'BROKER RATE'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'broker' in col.lower() and 'rate' in col.lower():
                broker_rate_col = col
                break
    
    if trailer_col and broker_rate_col:
        return df.groupby(trailer_col)[broker_rate_col].sum().reset_index()
    else:
        return pd.DataFrame(columns=[trailer_col or 'Trailer Type', broker_rate_col or 'Broker Rate'])

def get_top_drivers(df, n=20):
    """Calculates earnings for top N drivers using dynamic column detection."""
    # Find driver name and driver rate columns
    driver_name_col = None
    driver_rate_col = None
    
    # Find driver name column
    if 'DRIVER NAME' in df.columns:
        driver_name_col = 'DRIVER NAME'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'driver' in col.lower() and 'name' in col.lower():
                driver_name_col = col
                break
    
    # Find driver rate column
    if 'DRIVER RATE [$]' in df.columns:
        driver_rate_col = 'DRIVER RATE [$]'
    elif 'DRIVER RATE' in df.columns:
        driver_rate_col = 'DRIVER RATE'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'driver' in col.lower() and 'rate' in col.lower():
                driver_rate_col = col
                break
    
    if driver_name_col and driver_rate_col:
        return df.groupby(driver_name_col)[driver_rate_col].sum().nlargest(n).reset_index()
    else:
        return pd.DataFrame(columns=[driver_name_col or 'Driver Name', driver_rate_col or 'Driver Rate'])

def get_carrier_miles(df):
    """Calculates miles per carrier."""
    # Using 'LOAD\'S CARRIER COMPANY' and 'FULL MILES TOTAL' columns
    if 'LOAD\'S CARRIER COMPANY' in df.columns and 'FULL MILES TOTAL' in df.columns:
        return df.groupby('LOAD\'S CARRIER COMPANY')['FULL MILES TOTAL'].sum().reset_index()
    else:
        return pd.DataFrame()

def get_broker_rate_cfc_stats(df):
    """Calculates stats using dynamic broker rate column detection."""
    # Find dispatcher and broker rate columns
    dispatcher_col = None
    broker_rate_col = None
    
    # Find dispatcher column
    if 'DISPATCH NAME' in df.columns:
        dispatcher_col = 'DISPATCH NAME'
    elif 'FC NAME' in df.columns:
        dispatcher_col = 'FC NAME'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'dispatch' in col.lower() or 'fc' in col.lower():
                dispatcher_col = col
                break
    
    # Find broker rate column
    if 'BROKER RATE (FC) [$' in df.columns:
        broker_rate_col = 'BROKER RATE (FC) [$'
    elif 'BROKER RATE (CFC)' in df.columns:
        broker_rate_col = 'BROKER RATE (CFC)'
    elif 'BROKER RATE' in df.columns:
        broker_rate_col = 'BROKER RATE'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'broker' in col.lower() and 'rate' in col.lower():
                broker_rate_col = col
                break
    
    if dispatcher_col and broker_rate_col:
        return df.groupby(dispatcher_col)[broker_rate_col].sum().reset_index()
    else:
        return pd.DataFrame(columns=[dispatcher_col or 'Dispatcher', broker_rate_col or 'Broker Rate'])

def get_bd_margin_stats(df):
    """Calculates stats using dynamic BD margin column detection."""
    # Find dispatcher and BD margin columns
    dispatcher_col = None
    bd_margin_col = None
    
    # Find dispatcher column
    if 'DISPATCH NAME' in df.columns:
        dispatcher_col = 'DISPATCH NAME'
    elif 'FC NAME' in df.columns:
        dispatcher_col = 'FC NAME'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'dispatch' in col.lower() or 'fc' in col.lower():
                dispatcher_col = col
                break
    
    # Find BD margin column
    if 'BD MARGIN [$]' in df.columns:
        bd_margin_col = 'BD MARGIN [$]'
    elif 'BD MARGIN' in df.columns:
        bd_margin_col = 'BD MARGIN'
    else:
        # Try case-insensitive search
        for col in df.columns:
            if 'bd' in col.lower() and 'margin' in col.lower():
                bd_margin_col = col
                break
    
    if dispatcher_col and bd_margin_col:
        return df.groupby(dispatcher_col)[bd_margin_col].sum().reset_index()
    else:
        return pd.DataFrame(columns=[dispatcher_col or 'Dispatcher', bd_margin_col or 'BD Margin']) 