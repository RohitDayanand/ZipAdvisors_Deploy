import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
import io
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pymysql
import xgboost as xgb
from difflib import get_close_matches
from difflib import SequenceMatcher
import random 
pymysql.install_as_MySQLdb()

model = xgb.XGBClassifier()
model = model.load_model('Models/two_player_xgb.json')

table_map = {

}
market_name_map = {

}

load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME =os.getenv('DB_NAME')

conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
    user=f'{DB_USER}',
    password=f'{DB_PASS}',
    host = 'jsedocc7.scrc.nyu.edu',
    port     = 3306,
    encoding = 'utf-8',
    db = f'{DB_NAME}'
)
engine = create_engine(conn_string)

def get_polymarket_df(P_prediction, P_market):
    try:
        query_polymarket = f"""
            SELECT yes_price, no_price, timestamp
            FROM {P_prediction}
            WHERE market_name = '{P_market}'
            ORDER BY timestamp
        """
        df_polymarket = pd.read_sql(query_polymarket, engine)
        df_polymarket['timestamp'] = pd.to_datetime(df_polymarket['timestamp'])
        return df_polymarket
    except Exception as e:
        print(f"❌ Error loading Polymarket data for {P_market}: {e}")

def get_kalshi_df(K_prediction, K_market):
    try:
        query_kalshi = f"""
            SELECT yes_price, no_price, timestamp
            FROM {K_prediction}
            WHERE market_name = '{K_market}'
            ORDER BY timestamp
        """
        df_kalshi = pd.read_sql(query_kalshi, engine)
        df_kalshi['timestamp'] = pd.to_datetime(df_kalshi['timestamp'])
        return df_kalshi
    except Exception as e:
        print(f"❌ Error loading Polymarket data for {K_market}: {e}")

def get_table_names():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return table_names

def get_market_names(table_name):
    market_names = pd.read_sql(f'SELECT DISTINCT market_name FROM {table_name}', con=engine)['market_name'].tolist()
    return market_names

def suggest_market_table_mapping(polymarket_tables, kalshi_tables):
    mapping = {}

    def clean(name):
        return name.split('_', 1)[1].lower() if '_' in name else name.lower()

    for p_table in polymarket_tables:
        p_clean = clean(p_table)
        best_k_match = get_close_matches(p_clean, [clean(k) for k in kalshi_tables], n=1, cutoff=0.5)
        if best_k_match:
            original_k_table = next(k for k in kalshi_tables if clean(k) == best_k_match[0])
            mapping[p_table] = original_k_table

    return mapping

def generate_nested_market_name_map(table_pairs, cutoff=0.6):
    nested_map = {}

    for p_table, k_table in table_pairs.items():
        query_p = f"SELECT DISTINCT market_name FROM {p_table}"
        query_k = f"SELECT DISTINCT market_name FROM {k_table}"

        p_markets = pd.read_sql(query_p, con=engine)['market_name'].dropna().unique()
        k_markets = pd.read_sql(query_k, con=engine)['market_name'].dropna().unique()

        k_cleaned = {name: name.lower() for name in k_markets}

        submap = {}
        for p_market in p_markets:
            p_lower = p_market.lower()
            match = get_close_matches(p_lower, k_cleaned.values(), n=1, cutoff=cutoff)
            if match:
                matched_k = next(k for k, v in k_cleaned.items() if v == match[0])
                submap[p_lower] = {'polymarket': p_market, 'kalshi': matched_k}
                submap[matched_k.lower()] = {'polymarket': p_market, 'kalshi': matched_k}

        nested_map[p_table] = submap
        nested_map[k_table] = submap

    return nested_map

def setup_maps():
    global table_map
    global market_name_map
    table_names = get_table_names()
    polymarket_tables = [name for name in table_names if name.startswith('P_')]
    kalshi_tables = [name for name in table_names if name.startswith('K_')]
    table_map = suggest_market_table_mapping(polymarket_tables, kalshi_tables)
    market_name_map = generate_nested_market_name_map(table_map)


def get_team_market_data_for_xgb(input_table, input_market):
    global market_name_map
    global table_map

    input_market_lower = input_market.lower()
    is_polymarket = input_table.startswith('P_')
    paired_table = table_map[input_table]

    submap = market_name_map.get(input_table)
    if not submap or input_market_lower not in submap:
        raise ValueError(f"Market '{input_market}' not found for table '{input_table}'.")

    base_market = submap[input_market_lower]
    polymarket_table = input_table if is_polymarket else paired_table
    kalshi_table = paired_table if is_polymarket else input_table

    polymarket_market = base_market['polymarket']
    kalshi_market = base_market['kalshi']

    # Polymarket query
    query_p = f"""
        SELECT market_name, yes_price, no_price, timestamp 
        FROM {polymarket_table}
        WHERE market_name LIKE %s 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    df_p = pd.read_sql(query_p, con=engine, params=(f'%{polymarket_market}%',))
    df_p['timestamp'] = pd.to_datetime(df_p['timestamp'])
    df_p = df_p.rename(columns={
        'yes_price': f'{input_market}_polymarket_yes_price',
        'no_price': f'{input_market}_polymarket_no_price'
    }).sort_values('timestamp')
    df_p = df_p.tail(30).reset_index(drop=True)

    # Kalshi query
    query_k = f"""
        SELECT market_name, yes_price, no_price, timestamp 
        FROM {kalshi_table}
        WHERE market_name LIKE %s 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    df_k = pd.read_sql(query_k, con=engine, params=(f'%{kalshi_market}%',))
    df_k['timestamp'] = pd.to_datetime(df_k['timestamp'])
    df_k = df_k.rename(columns={
        'yes_price': f'{input_market}_kalshi_yes_price',
        'no_price': f'{input_market}_kalshi_no_price'
    }).sort_values('timestamp')
    df_k = df_k.tail(30).reset_index(drop=True)

    print(f"✅ Polymarket shape: {df_p.shape}, Kalshi shape: {df_k.shape}")
    return df_p, df_k

def align_and_generate_features(df, lags=3, player=None):
    print("DEBUG: Entered align_and_generate_features")
    prefix = f"{player}_" if player else ""
    print(f"DEBUG: Using prefix: '{prefix}'")

    # Compute base delta logs (1-step returns)
    df[f'{prefix}delta_log_kalshi_yes'] = np.log(df[f'{prefix}kalshi_yes_price']) - np.log(df[f'{prefix}kalshi_yes_price'].shift(1))
    df[f'{prefix}delta_log_kalshi_no'] = np.log(df[f'{prefix}kalshi_no_price']) - np.log(df[f'{prefix}kalshi_no_price'].shift(1))
    df[f'{prefix}delta_log_polymarket_yes'] = np.log(df[f'{prefix}polymarket_yes_price']) - np.log(df[f'{prefix}polymarket_yes_price'].shift(1))
    df[f'{prefix}delta_log_polymarket_no'] = np.log(df[f'{prefix}polymarket_no_price']) - np.log(df[f'{prefix}polymarket_no_price'].shift(1))
    print("DEBUG: Computed delta logs")

    # Compute spreads
    df[f'{prefix}kalshi_spread'] = df[f'{prefix}kalshi_yes_price'] - df[f'{prefix}kalshi_no_price']
    df[f'{prefix}polymarket_spread'] = df[f'{prefix}polymarket_yes_price'] - df[f'{prefix}polymarket_no_price']
    print("DEBUG: Computed spreads")

    # Momentum features
    df[f'{prefix}lag_momentum_5']  = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).sum().shift(1)
    df[f'{prefix}lag_momentum_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).sum().shift(1)
    print("DEBUG: Computed momentum features")

    # Volatility features
    df[f'{prefix}lag_volatility_5']  = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).std().shift(1)
    df[f'{prefix}lag_volatility_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std().shift(1)
    print("DEBUG: Computed volatility features")

    # Z-score features
    z = (df[f'{prefix}delta_log_polymarket_yes'] - df[f'{prefix}delta_log_polymarket_yes'].rolling(10).mean()) / df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std()
    df[f'{prefix}lag_zscore_10'] = z.shift(1)
    print("DEBUG: Computed z-score feature")

    # Create classic lag features (short-term memory)
    lagged_features = []
    lagged_columns = []
    for i in range(1, lags + 1):
        for col_base in [
            f'{prefix}delta_log_kalshi_yes',
            f'{prefix}delta_log_kalshi_no',
            f'{prefix}delta_log_polymarket_yes',
            f'{prefix}delta_log_polymarket_no'
        ]:
            lagged_col = df[col_base].shift(i-1)
            lagged_features.append(lagged_col)
            lagged_columns.append(f'{prefix}lag_{i}_' + col_base.split(prefix)[-1])
    print("DEBUG: Created lag features for columns:", lagged_columns)

    lagged_df = pd.concat(lagged_features, axis=1)
    lagged_df.columns = lagged_columns

    # Drop rows with missing values in required columns
    required_cols = [
        f'{prefix}delta_log_kalshi_yes',
        f'{prefix}delta_log_kalshi_no',
        f'{prefix}delta_log_polymarket_yes',
        f'{prefix}delta_log_polymarket_no'
    ]
    df = df.dropna(subset=required_cols)
    print("DEBUG: Dropped rows with NA in required columns, new shape:", df.shape)

    # Align with lagged_df
    lagged_df = lagged_df.loc[df.index].dropna()
    print("DEBUG: Aligned lagged_df shape:", lagged_df.shape)

    # Merge everything into one DataFrame
    final_df = pd.concat([df, lagged_df], axis=1).dropna()
    print("DEBUG: Final feature dataframe shape:", final_df.shape)

    return final_df

def xgb_predict(final_df, key_players):
    print("DEBUG: Entered xgb_predict")
    df_features = final_df.copy()
    # === Select only lagged features
    lag_cols = [col for col in df_features.columns if 'lag_' in col]
    print("DEBUG: Initial lag feature columns:", lag_cols)
    X = df_features[lag_cols]

    # Normalize column names for model consistency
    rename_mapping = {}
    for idx, player in enumerate(key_players, start=1):
        for col in df_features.columns:
            if (col.startswith(player + '_')):
                new_col = col.replace(player + '_', f'Player{idx}_')
                rename_mapping[col] = new_col
    print("DEBUG: Rename mapping:", rename_mapping)
    X = X.rename(columns=rename_mapping)
    print("DEBUG: Renamed feature columns:", X.columns.tolist())

    try:
        y_pred = model.predict(X)[0]
        print("DEBUG: Model prediction successful, y_pred:", y_pred)
        return y_pred
    except Exception as e:
        print("DEBUG: Error during model prediction:", e)
        raise e

def get_random_dissimilar_market(input_table, input_market, threshold=0.6):
    print("DEBUG: Entered get_random_dissimilar_market with input_market:", input_market)
    input_market_lower = input_market.lower()
    all_markets = market_name_map.get(input_table, {}).keys()
    dissimilar_markets = [
        market for market in all_markets
        if SequenceMatcher(None, market, input_market_lower).ratio() < threshold
    ]
    print("DEBUG: Found dissimilar markets:", dissimilar_markets)
    if not dissimilar_markets:
        raise ValueError("No sufficiently dissimilar markets found.")
    chosen = random.choice(dissimilar_markets)
    print("DEBUG: Chosen dissimilar market:", chosen)
    return chosen

def xgb_algorithm(input_table, input_market):
    print("DEBUG: Entered xgb_algorithm with input_table:", input_table, "and input_market:", input_market)
    second_market = get_random_dissimilar_market(input_table, input_market)
    two_players = [input_market, second_market]
    print("DEBUG: Two players markets:", two_players)

    mergedP, mergedK = get_team_market_data_for_xgb(input_table, input_market)
    merged2P, merged2K = get_team_market_data_for_xgb(input_table, two_players[1])
    print("DEBUG: Fetched team market data")

    # Convert timestamps
    for df in [mergedP, mergedK, merged2P, merged2K]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    print("DEBUG: Converted and set timestamps as index")

    # Create a common time index
    start = max(df.index.min() for df in [mergedP, mergedK, merged2P, merged2K])
    end = min(df.index.max() for df in [mergedP, mergedK, merged2P, merged2K])
    common_index = pd.date_range(start=start, end=end, freq='20S')
    print("DEBUG: Constructed common index from", start, "to", end)

    # Reindex and forward fill
    dfs = [df.reindex(common_index).ffill() for df in [mergedP, mergedK, merged2P, merged2K]]
    print("DEBUG: Reindexed dataframes with common index")

    # Combine player 1 data (Polymarket + Kalshi)
    player1_df = pd.concat([mergedP, mergedK], axis=1)
    player1_df = player1_df.loc[~player1_df.index.duplicated(keep='first')]
    print("DEBUG: Player 1 dataframe shape:", player1_df.shape)

    # Combine player 2 data (Polymarket + Kalshi)
    player2_df = pd.concat([merged2P, merged2K], axis=1)
    player2_df = player2_df.loc[~player2_df.index.duplicated(keep='first')]
    print("DEBUG: Player 2 dataframe shape:", player2_df.shape)

    # Create a common time index for both players
    start = max(player1_df.index.min(), player2_df.index.min())
    end = min(player1_df.index.max(), player2_df.index.max())
    common_index = pd.date_range(start=start, end=end, freq='20S')
    print("DEBUG: New common index for players from", start, "to", end)

    # Reindex and forward fill
    player1_df = player1_df.reindex(common_index).ffill()
    player2_df = player2_df.reindex(common_index).ffill()
    print("DEBUG: Reindexed both player dataframes")

    final_df = pd.DataFrame()

    features_player_1 = align_and_generate_features(player1_df, 3, input_market)
    if features_player_1 is not None:
        features_player_1.set_index('timestamp', inplace=True)
        final_df = pd.concat([final_df, features_player_1], axis=1)
        print("DEBUG: Added features for player 1, shape:", features_player_1.shape)

    features_player_2 = align_and_generate_features(player2_df, 3, two_players[1])
    if features_player_2 is not None:
        features_player_2.set_index('timestamp', inplace=True)
        final_df = pd.concat([final_df, features_player_2], axis=1)
        print("DEBUG: Added features for player 2, shape:", features_player_2.shape)

    final_df = final_df.dropna()
    print("DEBUG: Final combined dataframe shape after dropna:", final_df.shape)
    
    # Predict using the model
    try:
        y_pred = xgb_predict(final_df, two_players)
        print("DEBUG: xgb_predict executed successfully")
        return y_pred
    except Exception as e:
        print("DEBUG: Error in xgb_predict:", e)

def xgb_merge_databases(P_prediction, P_market, K_prediction, K_market):
    df_polymarket = get_polymarket_df(P_prediction, P_market)
    df_kalshi = get_kalshi_df(K_prediction, K_market)

    #rename each column to it's relevant polymarket_kalshi_prefix (except timestamp)
    df_polymarket = df_polymarket.rename(columns={col: f'polymarket_{col}' for col in df_polymarket.columns if col != 'timestamp'})
    df_kalshi = df_kalshi.rename(columns={col: f'kalshi_{col}' for col in df_kalshi.columns if col != 'timestamp'})

    df_polymarket['timestamp'] = pd.to_datetime(df_polymarket['timestamp'])
    df_kalshi['timestamp'] = pd.to_datetime(df_kalshi['timestamp'])

    df_polymarket.set_index('timestamp', inplace=True)
    df_kalshi.set_index('timestamp', inplace=True)

    # Create a common time index (e.g., 1-minute intervals)
    common_index = pd.date_range(
      start=max(df_polymarket.index.min(), df_kalshi.index.min()),
      end=min(df_polymarket.index.max(), df_kalshi.index.max()),
      freq='21S'  # or whatever granularity you want
    )

    # Reindex both DataFrames and interpolate
    df_polymarket = df_polymarket.reindex(common_index).ffill()
    df_kalshi = df_kalshi.reindex(common_index).ffill()
    df_combined = pd.concat([df_polymarket, df_kalshi], axis=1)
    df_combined = df_combined.reset_index().rename(columns={'index': 'timestamp'})
    df_combined = df_combined.dropna()

    return df_combined

def plot_polymarket_data(P_prediction, P_market, choice):
    if choice == 'yes':
        choice = 'yes_price'
    else:
        choice = 'no_price'
    df_polymarket = get_polymarket_df(P_prediction, P_market)
    if choice == 'yes':
        choice = 'yes_price'

    if df_polymarket is None or not isinstance(df_polymarket, pd.DataFrame) or df_polymarket.empty:
        raise ValueError("Polymarket DataFrame is invalid or empty.")
    
    if choice not in df_polymarket.columns:
        raise ValueError(f"Column '{choice}' not found in Polymarket DataFrame.")
    
    else:
        choice = 'no_price'
    fig = px.line(df_polymarket, x='timestamp', y=f'{choice}', title=f'Polymarket Graph for {P_prediction}')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30s", step="second", stepmode="backward"), 
                dict(count=1, label="1m", step="minute", stepmode="backward"), 
                dict(count=1, label="1h", step="hour", stepmode="backward"), 
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all", label="To Date"),
            ])
        )
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_kalshi_data(K_prediction, K_market, choice):
    if choice == 'yes':
        choice = 'yes_price'
    else:
        choice = 'no_price'
    df_kalshi = get_kalshi_df(K_prediction, K_market)
    if df_kalshi is None or not isinstance(df_kalshi, pd.DataFrame) or df_kalshi.empty:
        raise ValueError("Kalshi DataFrame is invalid or empty.")
    
    if choice not in df_kalshi.columns:
        raise ValueError(f"Column '{choice}' not found in Kalshi DataFrame.")

    prediction_label = convert_table_name_to_clean(K_prediction)
    market_label = convert_table_name_to_clean(K_market)

    fig = px.line(df_kalshi, x='timestamp', y=choice, title=f'Kalshi Graph for {prediction_label} and {market_label}')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=30, label="30s", step="second", stepmode="backward"), 
                dict(count=1, label="1m", step="minute", stepmode="backward"), 
                dict(count=1, label="1h", step="hour", stepmode="backward"), 
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all", label="To Date"),
            ])
        )
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', include_mathjax=False)

def convert_table_name_to_clean(name):
    return name.replace("P_", "").replace("K_", "").replace("_", " ").title()

def prediction_dropdowns():
    table_names = get_table_names()
    predictions = {convert_table_name_to_clean(name): name for name in table_names}
    return predictions

def prediction_dropdowns(prefix):
    table_names = get_table_names()
    filtered_tables = [name for name in table_names if name.startswith(prefix)]
    predictions = {convert_table_name_to_clean(name): name for name in filtered_tables}
    return predictions

def market_dropdowns(table_name):
    market_names = get_market_names(table_name)
    markets = {convert_table_name_to_clean(name): name for name in market_names}
    return markets

def plot_kalshi_volatility(K_prediction, K_market, window=12):
    """
    Overlay Kalshi YES price with rolling volatility on dual y-axes.

    Args:
        K_prediction (str): Table or identifier for the prediction.
        K_market (str): Table or identifier for the market.
        window (int): Rolling window size for volatility.

    Returns:
        str: HTML representation of the Plotly figure.
    """
    # Load and prepare data
    df_kalshi = get_kalshi_df(K_prediction, K_market)
    df_kalshi = df_kalshi.sort_values("timestamp")
    df_kalshi['volatility'] = df_kalshi['yes_price'].rolling(window=window).std()

    # Labels
    prediction_label = convert_table_name_to_clean(K_prediction)
    market_label = convert_table_name_to_clean(K_market)

    # Build figure
    fig = go.Figure()

    # YES price (primary axis)
    fig.add_trace(go.Scatter(
        x=df_kalshi['timestamp'],
        y=df_kalshi['yes_price'],
        mode='lines',
        name='YES Price',
        line=dict(color='blue')
    ))

    # Volatility (secondary axis, faint color)
    fig.add_trace(go.Scatter(
        x=df_kalshi['timestamp'],
        y=df_kalshi['volatility'],
        mode='lines',
        name=f'Volatility (Rolling {window})',
        line=dict(color='orange', dash='dot', width=2),
        opacity=0.5,
        yaxis='y2'
    ))

    # Layout with dual y-axes
    fig.update_layout(
        title=f"Kalshi YES Price and Volatility — {prediction_label} / {market_label}",
        xaxis=dict(
            title='Timestamp',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(title='YES Price', side='left'),
        yaxis2=dict(title='Volatility', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0, y=1),
        height=600,
        template='plotly_white'
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def plot_polymarket_volatility(P_prediction, P_market, window=12):
    # Load and prepare data
    df_polymarket = get_kalshi_df(P_prediction, P_market)
    df_polymarket = df_polymarket.sort_values("timestamp")
    df_polymarket['volatility'] = df_polymarket['yes_price'].rolling(window=window).std()

    # Labels
    prediction_label = convert_table_name_to_clean(P_prediction)
    market_label = convert_table_name_to_clean(P_market)

    # Build figure
    fig = go.Figure()

    # YES price (primary axis)
    fig.add_trace(go.Scatter(
        x=df_polymarket['timestamp'],
        y=df_polymarket['yes_price'],
        mode='lines',
        name='YES Price',
        line=dict(color='blue')
    ))

    # Volatility (secondary axis, faint color)
    fig.add_trace(go.Scatter(
        x=df_polymarket['timestamp'],
        y=df_polymarket['volatility'],
        mode='lines',
        name=f'Volatility (Rolling {window})',
        line=dict(color='orange', dash='dot', width=2),
        opacity=0.5,
        yaxis='y2'
    ))

    # Layout with dual y-axes
    fig.update_layout(
        title=f"Kalshi YES Price and Volatility — {prediction_label} / {market_label}",
        xaxis=dict(
            title='Timestamp',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        yaxis=dict(title='YES Price', side='left'),
        yaxis2=dict(title='Volatility', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0, y=1),
        height=600,
        template='plotly_white'
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

'''
# --- Apply to market YES contracts only ---
market = 'temp'#P_market
df_features = f#df_market.copy()

# Label using the filtered method
yes_series = df_features[f'{market}_delta_log_polymarket_yes']
df_features['target'] = label_with_volatility_filter(yes_series, volatility_window=5, multiplier=0.1)
df_features['contract_type'] = 'yes'

# Drop rows where the target or lag features are missing
lag_features = [col for col in df_features.columns if 'lag_' in col]

df_filtered = df_features.dropna(subset=['target'] + lag_features)

# Encode contract_type (still useful as dummy if you reintroduce NO contracts later)
df_filtered = pd.get_dummies(df_filtered, columns=['contract_type'])

# ✅ Print result
print("\nTarget class counts:")
print(df_filtered["target"].value_counts())
print("\nFinal shape:", df_filtered.shape)

def make_trend_plot(df_plot):
    # Use your timestamp index (or adjust if needed)
    df_plot = df_features.copy()
    df_plot = df_plot.dropna(subset=['target', f'{market}_polymarket_yes_price'])

    # Extract time series and signals
    price_series = df_plot[f'{market}_polymarket_yes_price']
    timestamps = df_plot.index
    buy_signals = df_plot[df_plot['target'] == 1]
    sell_signals = df_plot[df_plot['target'] == 0]

    # Plot the YES price over time
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, price_series, label='Polymarket YES Price', color='blue')

    # Mark buy/sell signals
    plt.scatter(buy_signals.index, buy_signals[f'{market}_polymarket_yes_price'], color='green', label='Buy Signal', marker='^', s=80)
    plt.scatter(sell_signals.index, sell_signals[f'{market}_polymarket_yes_price'], color='red', label='Sell Signal', marker='v', s=80)

    # Plot formatting
    plt.title(f"{market.replace('_', ' ').title()} - YES Price with Buy/Sell Signals")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Set up your params
delta_col = f'{market}_delta_log_polymarket_yes'
window = 10
multiplier = 1.5

# Copy and compute threshold
df_plot = df_features.copy()
df_plot['rolling_std'] = df_plot[delta_col].rolling(window).std()
df_plot['threshold'] = df_plot['rolling_std'] * multiplier
df_plot['-threshold'] = -df_plot['threshold']

# Plot delta vs threshold lines
plt.figure(figsize=(12, 6))
plt.plot(df_plot.index, df_plot[delta_col], label='Delta Log Price', color='blue')
plt.plot(df_plot.index, df_plot['threshold'], label='+Threshold', color='green', linestyle='--')
plt.plot(df_plot.index, df_plot['-threshold'], label='–Threshold', color='red', linestyle='--')

plt.title(f"{market.replace('_', ' ').title()} – ΔLog(Price) vs Volatility Threshold")
plt.xlabel("Timestamp")
plt.ylabel("ΔLog Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

def rebalance_and_train(df_combined):
  train_list = []
  df_features = df_combined.copy()
  # === Select only lagged features
  lag_cols = [col for col in df_features.columns if 'lag_' in col]
  X = df_features[lag_cols]

  #now normalize the X names so the model is reusable later on without breaking previous functionality
  rename_mapping = {}

  X = X.rename(columns=rename_mapping)
  print(X.columns)

  y = df_features['target']

  print(X.shape)
  print(y.shape)

  # === Train-test split (time-aware, no shuffle)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

  for col in X_train.columns:
    train_list.append(col)

  # === Print data shapes
  print("X_train shape:", X_train.shape)  # Add this line
  print("y_train shape:", y_train.shape)  # Add this line

  # === Drop rows with NaN values in y_train and y_test ===
  X_train = X_train[~np.isnan(y_train)]
  y_train = y_train[~np.isnan(y_train)]
  X_test = X_test[~np.isnan(y_test)]
  y_test = y_test[~np.isnan(y_test)]


  # === Train the classifier
  clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
  clf.fit(X_train, y_train)

  # === Evaluate
  y_pred = clf.predict(X_test)
  print("Classification Report:\n", classification_report(y_test, y_pred))

  return clf, X_test, y_test, y_pred

clf, X_test, y_test, y_pred = rebalance_and_train(df_features)

from sklearn.metrics import roc_auc_score

# y_true: true labels (0 or 1)
# y_scores: predicted probabilities or decision function scores

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
