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
from xgboost import XGBClassifier
import xgboost as xgb
pymysql.install_as_MySQLdb()

model = xgb.XGBClassifier()
model = model.load_model('Models/two_player_xgb.json')

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

def merge_databases(P_prediction, K_prediction, P_market, K_market):

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

def align_and_generate_features(df, lags=3, market=None):
    """
    Enhances raw time-series features with engineered lag features, spread metrics, and momentum over longer time frames.
    Assumes:
    - df is indexed by timestamp
    - columns like 'kalshi_yes_price', 'polymarket_yes_price' already exist and are numeric
    """

    prefix = f"{market}_" if market else ""

    # Compute base delta logs (1-step returns)
    df[f'{prefix}delta_log_kalshi_yes'] = np.log(df[f'{prefix}kalshi_yes_price']) - np.log(df[f'{prefix}kalshi_yes_price'].shift(1))
    df[f'{prefix}delta_log_kalshi_no'] = np.log(df[f'{prefix}kalshi_no_price']) - np.log(df[f'{prefix}kalshi_no_price'].shift(1))
    df[f'{prefix}delta_log_polymarket_yes'] = np.log(df[f'{prefix}polymarket_yes_price']) - np.log(df[f'{prefix}polymarket_yes_price'].shift(1))
    df[f'{prefix}delta_log_polymarket_no'] = np.log(df[f'{prefix}polymarket_no_price']) - np.log(df[f'{prefix}polymarket_no_price'].shift(1))

    # Compute spreads
    df[f'{prefix}kalshi_spread'] = df[f'{prefix}kalshi_yes_price'] - df[f'{prefix}kalshi_no_price']
    df[f'{prefix}polymarket_spread'] = df[f'{prefix}polymarket_yes_price'] - df[f'{prefix}polymarket_no_price']

    # Add longer-term engineered features (momentum, volatility, z-score)
    df[f'{prefix}lag_momentum_polymarket_5'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).sum().shift(1)
    df[f'{prefix}lag_momentum_polymarket_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).sum().shift(1)

    df[f'{prefix}lag_volatility_polymarket_5'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(5).std().shift(1)
    df[f'{prefix}lag_volatility_polymarket_10'] = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std().shift(1)

     # Add longer-term engineered features (momentum, volatility, z-score)
    df[f'{prefix}lag_momentum_kalshi_5'] = df[f'{prefix}delta_log_kalshi_yes'].rolling(5).sum().shift(1)
    df[f'{prefix}lag_momentum_kalshi_10'] = df[f'{prefix}delta_log_kalshi_yes'].rolling(10).sum().shift(1)

    df[f'{prefix}lag_volatility_kalshi_5'] = df[f'{prefix}delta_log_kalshi_yes'].rolling(5).std().shift(1)
    df[f'{prefix}lag_volatility_kalshi_10'] = df[f'{prefix}delta_log_kalshi_yes'].rolling(10).std().shift(1)

    rolling_mean = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).mean().shift(1)
    rolling_std = df[f'{prefix}delta_log_polymarket_yes'].rolling(10).std().shift(1)

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
            lagged_col = df[col_base].shift(i)
            lagged_features.append(lagged_col)
            lagged_columns.append(f'{prefix}lag_{i}_' + col_base.split(prefix)[-1])

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

    # Align with lagged_df
    lagged_df = lagged_df.loc[df.index].dropna()

    # Merge everything into one DataFrame
    final_df = pd.concat([df, lagged_df], axis=1).dropna()

    return final_df

# Label using rolling volatility filter: 1 = Buy (UP), 0 = Sell (DOWN)
def label_with_volatility_filter(series, volatility_window=5, multiplier=0.5):
    """
    Labels only meaningful price movements using rolling volatility.
    Args:
    - series: the delta log price series
    - volatility_window: how many past steps to use to compute local volatility
    - multiplier: how 'strong' the move must be to count as a signal
    Returns:
    - target: Series with 1 = Buy, 0 = Sell, NaN = noise (optional to ffill)
    """
    rolling_std = series.rolling(volatility_window).std()

    def assign_label(x, threshold):
        if x > threshold:
            return 1
        elif x < -threshold:
            return 0
        else:
            return np.nan  # Weak signal (optional: fill later)

    thresholds = rolling_std * multiplier
    labels = series.combine(thresholds, assign_label)

    def assign_sell_label(x, threshold):
        if x < -threshold:
            return 1
        else:
            return 0
    def assign_buy_label(x, threshold):
        if x > threshold:
            return 1
        else:
            return 0

    thresholds = rolling_std * multiplier

    return labels

# Where the inputs from the frontend are going to be
def plot_merged_data():
    P_prediction = 'P_nba_western_conference_champion'
    P_market = 'denver_nuggets'
    K_prediction = 'K_nba_western_conference_championship'
    K_market = 'denver'

    df_market = merge_databases(P_prediction, K_prediction, P_market, K_market)
    df_market = df_market.rename(columns={col: f'{P_market}_{col}' for col in df_market.columns if col != 'timestamp'})
    df_market = align_and_generate_features(df_market, 5, P_market) # number of lags is second
    df_market['timestamp'] = df_market.index

    if df_market is not None:
        df_market.set_index('timestamp', inplace=True)

    # Drop rows with any missing values (optional)
    df_market = df_market.dropna()

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
'''