import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
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

table_map = {}
market_name_map = {}

load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')

conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
    user=f'{DB_USER}',
    password=f'{DB_PASS}',
    host = 'jsedocc7.scrc.nyu.edu',
    port=3306,
    encoding='utf-8',
    db=f'{DB_NAME}'
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
        pass

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
        pass

def get_table_names():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return table_names

def get_market_names(table_name):
    market_names = pd.read_sql(f'SELECT DISTINCT market_name FROM {table_name}', con=engine)['market_name'].tolist()
    return market_names

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

    prediction_label = convert_table_name_to_clean(P_prediction)
    market_label = convert_table_name_to_clean(P_market)
    fig = px.line(df_polymarket, x='timestamp', y=f'{choice}', title=f'Polymarket Graph for {prediction_label} and {market_label}')
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

def plot_kalshi_volatility(K_prediction, K_market, window=12):
    df_kalshi = get_kalshi_df(K_prediction, K_market)
    df_kalshi = df_kalshi.sort_values("timestamp")
    df_kalshi['volatility'] = df_kalshi['yes_price'].rolling(window=window).std()
    prediction_label = convert_table_name_to_clean(K_prediction)
    market_label = convert_table_name_to_clean(K_market)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_kalshi['timestamp'],
        y=df_kalshi['yes_price'],
        mode='lines',
        name='YES Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_kalshi['timestamp'],
        y=df_kalshi['volatility'],
        mode='lines',
        name=f'Volatility (Rolling {window})',
        line=dict(color='orange', dash='dot', width=2),
        opacity=0.5,
        yaxis='y2'
    ))
    fig.update_layout(
        title=f"Kalshi YES Price and Volatility — {prediction_label} / {market_label}",
        xaxis=dict(
            title='Timestamp',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=30, label="30s", step="second", stepmode="backward"), 
                    dict(count=1, label="1m", step="minute", stepmode="backward"), 
                    dict(count=1, label="1h", step="hour", stepmode="backward"), 
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="To Date"),
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
    df_polymarket = get_kalshi_df(P_prediction, P_market)
    df_polymarket = df_polymarket.sort_values("timestamp")
    df_polymarket['volatility'] = df_polymarket['yes_price'].rolling(window=window).std()
    prediction_label = convert_table_name_to_clean(P_prediction)
    market_label = convert_table_name_to_clean(P_market)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_polymarket['timestamp'],
        y=df_polymarket['yes_price'],
        mode='lines',
        name='YES Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_polymarket['timestamp'],
        y=df_polymarket['volatility'],
        mode='lines',
        name=f'Volatility (Rolling {window})',
        line=dict(color='orange', dash='dot', width=2),
        opacity=0.5,
        yaxis='y2'
    ))
    fig.update_layout(
        title=f"Kalshi YES Price and Volatility — {prediction_label} / {market_label}",
        xaxis=dict(
            title='Timestamp',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=30, label="30s", step="second", stepmode="backward"), 
                    dict(count=1, label="1m", step="minute", stepmode="backward"), 
                    dict(count=1, label="1h", step="hour", stepmode="backward"), 
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="To Date"),
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
            mapping[original_k_table] = p_table

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
    try:
        input_market_lower = input_market.lower()
        is_polymarket = input_table.startswith('P_')

        paired_table = table_map.get(input_table)
        if not paired_table:
            raise ValueError(f"Paired table not found for input table '{input_table}'.")

        submap = market_name_map.get(input_table)
        if not submap:
            raise ValueError(f"Market mapping not found for table '{input_table}'.")
        matches = get_close_matches(input_market_lower, list(submap.keys()), n=1, cutoff=0.6)
        if not matches:
            raise ValueError(f"Market '{input_market}' not found for table '{input_table}'.")
        base_market = submap[matches[0]]

        polymarket_table = paired_table if is_polymarket else input_table
        kalshi_table = input_table if is_polymarket else paired_table

        polymarket_market = base_market['polymarket']
        kalshi_market = base_market['kalshi']

        query_p = f"""
            SELECT yes_price, no_price, timestamp 
            FROM {polymarket_table}
            WHERE market_name LIKE %s 
            ORDER BY timestamp DESC 
            LIMIT 100
        """
        try:
            df_p = pd.read_sql(query_p, con=engine, params=(f'%{polymarket_market}%',))
        except Exception as e:
            raise e

        try:
            df_p['timestamp'] = pd.to_datetime(df_p['timestamp'])
            df_p = df_p.rename(columns={
                'yes_price': f'{input_market}_polymarket_yes_price',
                'no_price': f'{input_market}_polymarket_no_price'
            }).sort_values('timestamp')
            df_p = df_p.tail(30).reset_index(drop=True)
        except Exception as e:
            raise e

        query_k = f"""
            SELECT yes_price, no_price, timestamp 
            FROM {kalshi_table}
            WHERE market_name LIKE %s 
            ORDER BY timestamp DESC 
            LIMIT 100
        """
        try:
            df_k = pd.read_sql(query_k, con=engine, params=(f'%{kalshi_market}%',))
        except Exception as e:
            raise e

        try:
            df_k['timestamp'] = pd.to_datetime(df_k['timestamp'])
            df_k = df_k.rename(columns={
                'yes_price': f'{input_market}_kalshi_yes_price',
                'no_price': f'{input_market}_kalshi_no_price'
            }).sort_values('timestamp')
            df_k = df_k.tail(50).reset_index(drop=True)
        except Exception as e:
            raise e

        return df_p, df_k

    except Exception as e:
        raise e

def align_and_generate_features(df, lags=3, player=None):
    try:
        prefix = f"{player}_" if player else ""
        required_columns = [
            f'{prefix}kalshi_yes_price',
            f'{prefix}kalshi_no_price',
            f'{prefix}polymarket_yes_price',
            f'{prefix}polymarket_no_price'
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                pass

    
        df[f'{prefix}delta_log_kalshi_yes'] = (
            np.log(df[f'{prefix}kalshi_yes_price']) - np.log(df[f'{prefix}kalshi_yes_price'].shift(1))
        )
        df[f'{prefix}delta_log_kalshi_no'] = (
            np.log(df[f'{prefix}kalshi_no_price']) - np.log(df[f'{prefix}kalshi_no_price'].shift(1))
        )
        df[f'{prefix}delta_log_polymarket_yes'] = (
            np.log(df[f'{prefix}polymarket_yes_price']) - np.log(df[f'{prefix}polymarket_yes_price'].shift(1))
        )
        df[f'{prefix}delta_log_polymarket_no'] = (
            np.log(df[f'{prefix}polymarket_no_price']) - np.log(df[f'{prefix}polymarket_no_price'].shift(1))
        )

        df[f'{prefix}kalshi_spread'] = df[f'{prefix}kalshi_yes_price'] - df[f'{prefix}kalshi_no_price']
        df[f'{prefix}polymarket_spread'] = df[f'{prefix}polymarket_yes_price'] - df[f'{prefix}polymarket_no_price']

        df[f'{prefix}lag_momentum_5'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).sum().shift(1)
        df[f'{prefix}lag_momentum_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).sum().shift(1)

        df[f'{prefix}lag_volatility_5'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).std().shift(1)
        df[f'{prefix}lag_volatility_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std().shift(1)

        z = (df[f'{prefix}delta_log_polymarket_yes'] - df[f'{prefix}delta_log_polymarket_yes'].rolling(10).mean()) / df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std()
        df[f'{prefix}lag_zscore_10'] = z.shift(1)

        # Fill NaNs that result from rolling and shifting with 0 (or you can use another method if preferred)
        cols_to_fill = [
            f'{prefix}lag_momentum_5',
            f'{prefix}lag_momentum_10',
            f'{prefix}lag_volatility_5',
            f'{prefix}lag_volatility_10',
            f'{prefix}lag_zscore_10'
        ]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)

        try:
            lagged_features = []
            lagged_columns = []
            for i in range(1, lags + 1):
                for col_base in [
                    f'{prefix}delta_log_kalshi_yes',
                    f'{prefix}delta_log_kalshi_no',
                    f'{prefix}delta_log_polymarket_yes',
                    f'{prefix}delta_log_polymarket_no'
                ]:
                    lagged_features.append(df[col_base].shift(i - 1))
                    lagged_columns.append(f'{prefix}lag_{i}_' + col_base.split(prefix)[-1])
            lagged_df = pd.concat(lagged_features, axis=1)
            lagged_df.columns = lagged_columns
        except Exception as e:
            raise e

        try:
            final_df = pd.concat([df, lagged_df], axis=1)
        except Exception as e:
            raise e
        return final_df
    except Exception as overall_e:
        raise overall_e

def xgb_predict(final_df, key_players):
    df_features = final_df.copy()
    lag_cols = [col for col in df_features.columns if 'lag_' in col]

    X = df_features[lag_cols]
    rename_mapping = {}
    for idx, player in enumerate(key_players, start=1):
        for col in df_features.columns:
            if (col.startswith(player + '_')):
                new_col = col.replace(player + '_', f'Player{idx}_')
                rename_mapping[col] = new_col
    X = X.rename(columns=rename_mapping)

    model = xgb.XGBClassifier()
    model.load_model('./Models/two_player_xgb.json')
    try:
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)
        return y_pred, y_prob
    except Exception as e:
        print("DEBUG: Error during model prediction:", e)
        raise e

def get_random_dissimilar_market(input_table, input_market, threshold=0.6):
    input_market_lower = input_market.lower()
    all_markets = market_name_map.get(input_table, {}).keys()
    dissimilar_markets = [
        market for market in all_markets
        if SequenceMatcher(None, market, input_market_lower).ratio() < threshold
    ]
    if not dissimilar_markets:
        raise ValueError("No sufficiently dissimilar markets found.")
    chosen = random.choice(dissimilar_markets)
    return chosen

def xgb_algorithm(input_table, input_market):
    setup_maps()
    second_market = get_random_dissimilar_market(input_table, input_market)
    two_players = [input_market, second_market]
    mergedP, mergedK = get_team_market_data_for_xgb(input_table, input_market)
    merged2P, merged2K = get_team_market_data_for_xgb(input_table, two_players[1])
    for df in [mergedP, mergedK, merged2P, merged2K]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    start = max(df.index.min() for df in [mergedP, mergedK, merged2P, merged2K])
    end = min(df.index.max() for df in [mergedP, mergedK, merged2P, merged2K])
    common_index = pd.date_range(start=start, end=end, freq='20S')
    mergedP = mergedP.reindex(common_index).interpolate(method='time').ffill().bfill().fillna(0)
    mergedK = mergedK.reindex(common_index).interpolate(method='time').ffill().bfill().fillna(0)
    merged2P = merged2P.reindex(common_index).interpolate(method='time').ffill().bfill().fillna(0)
    merged2K = merged2K.reindex(common_index).interpolate(method='time').ffill().bfill().fillna(0)
    player1_df = pd.concat([mergedP, mergedK], axis=1)
    player1_df = player1_df.loc[~player1_df.index.duplicated(keep='first')]
    player2_df = pd.concat([merged2P, merged2K], axis=1)
    player2_df = player2_df.loc[~player2_df.index.duplicated(keep='first')]
    final_df = pd.DataFrame()
    features_player_1 = align_and_generate_features(player1_df, 3, input_market)
    if features_player_1 is not None:
        final_df = pd.concat([final_df, features_player_1], axis=1)
    features_player_2 = align_and_generate_features(player2_df, 3, two_players[1])
    if features_player_2 is not None:
        final_df = pd.concat([final_df, features_player_2], axis=1)
    final_df = final_df.fillna(0)
    try:
        y_pred, y_prob = xgb_predict(final_df, two_players)
        return y_pred, y_prob
    except Exception as e:
        print("DEBUG: Error in xgb_predict:", e)

def xgb_algorithm_plot(input_table, input_market, threshold):
    setup_maps()
    global market_name_map
    global table_map
    input_market_lower = input_market.lower()
    is_polymarket = input_table.startswith('P_')

    paired_table = table_map.get(input_table)
    if not paired_table:
        raise ValueError(f"Paired table not found for input table '{input_table}'.")

    submap = market_name_map.get(input_table)
    if not submap:
        raise ValueError(f"Market mapping not found for table '{input_table}'.")
    matches = get_close_matches(input_market_lower, list(submap.keys()), n=1, cutoff=0.6)
    if not matches:
        raise ValueError(f"Market '{input_market}' not found for table '{input_table}'.")
    base_market = submap[matches[0]]

    polymarket_table = paired_table if is_polymarket else input_table
    kalshi_table = input_table if is_polymarket else paired_table

    polymarket_market = base_market['polymarket']
    kalshi_market = base_market['kalshi']
    k_plot = plot_kalshi_data(kalshi_table, kalshi_market, 'yes')

    # Recreate polymarket data
    choice = 'yes'
    df_polymarket = get_polymarket_df(polymarket_table, polymarket_market)
    if choice == 'yes':
        choice = 'yes_price'
    else:
        choice = 'no_price'
    if df_polymarket is None or not isinstance(df_polymarket, pd.DataFrame) or df_polymarket.empty:
        raise ValueError("Polymarket DataFrame is invalid or empty.")
    
    if choice not in df_polymarket.columns:
        raise ValueError(f"Column '{choice}' not found in Polymarket DataFrame.")
    
    paper_bgcolor = 'white'  # Default background color

    # Run the XGBoost algorithm to get predictions and probabilities
    y_pred, y_prob = xgb_algorithm(kalshi_table, kalshi_market)

    # Calculate the absolute difference between probabilities
    prob_diff = abs(y_prob[0][0] - y_prob[0][1])

    # Convert threshold to float (if not already)
    threshold = float(threshold)

    # Adjust the color logic based on the threshold and probability difference
    if prob_diff < threshold:  # If the probability difference is small, it's more uncertain
        # Suggesting more risk by taking "uptrend" or "downtrend"
        if y_prob[0][0] > 0.5:  # If probability for uptrend is higher than 50%
            paper_bgcolor = 'lightgreen'  # Light green for uptrend
        elif y_prob[0][1] > 0.5:  # If probability for downtrend is higher than 50%
            paper_bgcolor = 'lightcoral'  # Light red for downtrend
        else:
            paper_bgcolor = 'grey'  # Neutral color (e.g., grey)
    else:  # If the probability difference is large, we're more confident about the prediction
        if y_pred == 1:  # If predicted as uptrend (1)
            paper_bgcolor = 'lightgreen'  # Light green for uptrend
        elif y_pred == 0:  # If predicted as downtrend (0)
            paper_bgcolor = 'lightcoral'  # Light red for downtrend
    
    prediction_label = convert_table_name_to_clean(polymarket_table)
    market_label = convert_table_name_to_clean(polymarket_market)
    fig = px.line(df_polymarket, x='timestamp', y=f'{choice}', title=f'Predict Poly for {market_label}')
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
    
    fig.update_layout(
        paper_bgcolor = paper_bgcolor,
    )
    p_plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    return p_plot, k_plot

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
