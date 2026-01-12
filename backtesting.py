import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import pickle
import warnings
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")

class Config:
    DATA_FILE = r"C:\Users\Kalpana\Downloads\Techfest-Algorithmic-Trading-main\Techfest-Algorithmic-Trading-main\all_stock_data.pkl"
    UNIVERSE_SIZE = 60
    MIN_PRICE = 10.0
    LIQUIDITY_LOOKBACK = 126
    TRAIN_END_DATE = '2024-11-28'
    BACKTEST_END_DATE = '2025-11-28'
    SHORTING_BAN_DATE = '2025-10-28'
    INITIAL_CAPITAL = 1000000
    MAX_POSITIONS = 30
    # Risk Management
    MAX_ALLOCATION_PER_STOCK = 0.25
    ENTRY_CONFIDENCE_LONG = 0.65
    ENTRY_CONFIDENCE_SHORT = 0.65
    EXIT_PROB_THRESHOLD = 0.75
    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.001

class FeatureConfig:
    ENTRY_HORIZON = 5
    ENTRY_MOVE = 0.015
    EXIT_HORIZON = 4
    EXIT_MOVE = 0.0125

def load_and_select_universe():
    print(f"--- Step 1: Loading Data ")
    try:
        with open(Config.DATA_FILE, "rb") as f:
            raw_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: File not found.")
        return None, None, None

    clean_data = {}
    liquidity_scores = []
    benchmark_df = None
    train_cutoff = pd.Timestamp(Config.TRAIN_END_DATE)

    possible_benchmarks = ['NSE:NIFTY 500', 'NSE:CNX500', 'NSE:NIFTY']
    for b in possible_benchmarks:
        if b in raw_data:
            benchmark_df = raw_data[b].copy()
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if c in benchmark_df.columns:
                    benchmark_df[c] = pd.to_numeric(benchmark_df[c], errors='coerce')
            benchmark_df = benchmark_df.dropna()
            print(f"Benchmark found: {b}")
            break

    for ticker, df in raw_data.items():
        if "NIFTY" in ticker and "ET" not in ticker:
            continue
        
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()

        # Slice to train period for liquidity check (Removes Lookahead)
        train_slice = df[df.index < train_cutoff]
        
        if len(train_slice) < Config.LIQUIDITY_LOOKBACK:
            continue
        if train_slice['Close'].iloc[-1] < Config.MIN_PRICE:
            continue
            
        train_slice['Dollar_Vol'] = train_slice['Close'] * train_slice['Volume']
        avg_liq = train_slice['Dollar_Vol'].tail(Config.LIQUIDITY_LOOKBACK).mean()
        
        liquidity_scores.append((ticker, avg_liq))
        clean_data[ticker] = df

    liquidity_scores.sort(key=lambda x: x[1], reverse=True)
    top_tickers = [x[0] for x in liquidity_scores[:Config.UNIVERSE_SIZE]]
    
    # Synthetic index created for relative strength
    all_closes = pd.concat([clean_data[t]['Close'] for t in top_tickers], axis=1).ffill()
    market_index = all_closes.mean(axis=1)
    
    final_universe = {t: clean_data[t] for t in top_tickers}
    print(f"Selected {len(final_universe)} stocks based on pre-training liquidity.")
    return final_universe, market_index, benchmark_df

def add_features(df, market_series):
    df = df.copy()
    common_index = df.index.intersection(market_series.index)
    df = df.loc[common_index]
    mkt = market_series.loc[common_index]
    
    # Technicals
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ROC_10'] = ta.momentum.roc(df['Close'], window=10)
    change = df['Close'].diff(10).abs()
    volatility = df['Close'].diff(1).abs().rolling(10).sum()
    df['Efficiency_Ratio'] = change / (volatility + 1e-9)
    
    # Relative Strength
    stock_ret = df['Close'].pct_change(20)
    mkt_ret = mkt.pct_change(20)
    df['Rel_Strength'] = stock_ret - mkt_ret
    
    # Volatility & Momentum
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['NATR'] = (df['ATR'] / df['Close']) * 100
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    ema20 = ta.trend.ema_indicator(df['Close'], window=20)
    df['Dist_EMA20'] = np.log(df['Close'] / ema20)
    
    future_ret_5d = df['Close'].shift(-FeatureConfig.ENTRY_HORIZON) / df['Close'] - 1
    future_ret_3d = df['Close'].shift(-FeatureConfig.EXIT_HORIZON) / df['Close'] - 1
    
    df['Target_Entry_Long'] = (future_ret_5d >= FeatureConfig.ENTRY_MOVE).astype(int)
    df['Target_Entry_Short'] = (future_ret_5d <= -FeatureConfig.ENTRY_MOVE).astype(int)
    df['Target_Exit_Long'] = (future_ret_3d <= -FeatureConfig.EXIT_MOVE).astype(int)
    df['Target_Exit_Short'] = (future_ret_3d >= FeatureConfig.EXIT_MOVE).astype(int)
    
    # Store Raw Future Returns to enable explicit exclusion later (Safety)
    df['future_ret_5d'] = future_ret_5d
    df['future_ret_3d'] = future_ret_3d

    return df.dropna()

def prepare_data(universe, market_index):
    processed = {}
    for t, df in universe.items():
        try:
            processed[t] = add_features(df, market_index)
        except:
            continue
    return processed

def train_lgbm_models(processed_data):
    print("--- Step 3: Training LGBM Binary Classifiers ---")
    all_dfs = []
    for t, df in processed_data.items():
        df['Symbol'] = t
        all_dfs.append(df)
    full_df = pd.concat(all_dfs).replace([np.inf, -np.inf], np.nan).dropna()
    
    split_date = pd.Timestamp(Config.TRAIN_END_DATE)
    train_df = full_df[full_df.index < split_date]
    
    exclude = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Dollar_Vol', 'Symbol', 'Date',
        'Target_Entry_Long', 'Target_Entry_Short', 'Target_Exit_Long', 'Target_Exit_Short',
        'future_ret_5d', 'future_ret_3d' 
    ]
    features = [c for c in full_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(full_df[c])]
    
    print(f"Training on {len(features)} features. Leakage columns excluded.")
    
    models = {}
    targets = ['Target_Entry_Long', 'Target_Entry_Short', 'Target_Exit_Long', 'Target_Exit_Short']
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_estimators': 600,
        'learning_rate': 0.02,
        'num_leaves': 20,
        'min_child_samples': 50
    }
    
    for target in targets:
        y = train_df[target]
        X = train_df[features]
        pos = y.sum()
        scale = (len(y) - pos) / pos if pos > 0 else 1
        model = lgb.LGBMClassifier(**params, scale_pos_weight=scale)
        model.fit(X, y)
        models[target] = model
        
    return models, features

class AdvancedStrategy:
    def __init__(self, initial_capital):
        self.cash = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trade_log = []
        self.pending_orders = [] 
        self.pending_exits = []

    def get_portfolio_value(self, current_prices):
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in current_prices:
                price = current_prices[sym]
                if pos['type'] == 'LONG':
                    equity += pos['qty'] * price
                else:
                    equity += (pos['entry_price'] - price) * pos['qty'] + (pos['qty'] * pos['entry_price'])
        return equity

    def run_day(self, date, data_snapshot, models, features):
        for order in self.pending_exits:
            sym = order['sym']
            if sym in data_snapshot:
                fill_price = data_snapshot[sym]['Open'] 
                self.close_position(sym, fill_price, date, order['reason'])
        self.pending_exits = [] 

        current_equity_est = self.get_portfolio_value({s: d['Open'] for s, d in data_snapshot.items()})
        
        for order in self.pending_orders:
            sym = order['sym']
            if sym in data_snapshot and sym not in self.positions:
                fill_price = data_snapshot[sym]['Open'] 
                self.open_position(order, fill_price, date, current_equity_est)
        self.pending_orders = [] 
        current_prices_close = {}
        active_symbols = list(self.positions.keys())
        
        for sym in active_symbols:
            if sym not in data_snapshot: continue
            row = data_snapshot[sym]
            pos = self.positions[sym]
            current_prices_close[sym] = row['Close']
            if pos['type'] == 'LONG':
                stop_price = pos['entry_price'] - (2 * row['ATR'])
                if row['Low'] <= stop_price:
                    self.close_position(sym, stop_price, date, "Hard Stop (Intraday)")
            else:
                stop_price = pos['entry_price'] + (2 * row['ATR'])
                if row['High'] >= stop_price:
                    self.close_position(sym, stop_price, date, "Hard Stop (Intraday)")
        for sym, pos in self.positions.items():
            if sym not in data_snapshot: continue
            row = data_snapshot[sym]
            X_curr = pd.DataFrame([row[features]], columns=features)
            
            should_exit = False
            reason = ""
            
            if pos['type'] == 'LONG':
                prob = models['Target_Exit_Long'].predict_proba(X_curr)[0][1]
                if prob > Config.EXIT_PROB_THRESHOLD:
                    should_exit = True
                    reason = "Prob Exit"
            else:
                prob = models['Target_Exit_Short'].predict_proba(X_curr)[0][1]
                if prob > Config.EXIT_PROB_THRESHOLD:
                    should_exit = True
                    reason = "Prob Exit"
            
            if should_exit:
                self.pending_exits.append({'sym': sym, 'reason': reason})
        if len(self.positions) + len(self.pending_orders) < Config.MAX_POSITIONS:
            candidates = []
            for sym, row in data_snapshot.items():
                if sym in self.positions: continue
                if any(x['sym'] == sym for x in self.pending_exits): continue
                
                X_curr = pd.DataFrame([row[features]], columns=features)
                
                p_long = models['Target_Entry_Long'].predict_proba(X_curr)[0][1]
                p_short = models['Target_Entry_Short'].predict_proba(X_curr)[0][1]
                
                if p_long > Config.ENTRY_CONFIDENCE_LONG:
                    candidates.append({'sym': sym, 'type': 'LONG', 'prob': p_long, 'atr': row['ATR']})
                elif p_short > Config.ENTRY_CONFIDENCE_SHORT:
                    if date < pd.Timestamp(Config.SHORTING_BAN_DATE):
                        candidates.append({'sym': sym, 'type': 'SHORT', 'prob': p_short, 'atr': row['ATR']})
            
            candidates.sort(key=lambda x: x['prob'], reverse=True)
            slots_available = Config.MAX_POSITIONS - len(self.positions) - len(self.pending_orders)
            for cand in candidates[:slots_available]:
                self.pending_orders.append(cand)
        self.equity_curve.append({'date': date, 'equity': self.get_portfolio_value(current_prices_close)})

    def open_position(self, cand, price, date, equity):
        risk_per_trade = equity * 0.005 
        stop_dist = 2 * cand['atr']
        if stop_dist == 0: return
        qty = int(risk_per_trade / stop_dist)
        if qty <= 0: return
        exec_price = price * (1 + Config.SLIPPAGE) if cand['type'] == 'LONG' else price * (1 - Config.SLIPPAGE)
        
        cost = qty * exec_price
        comm = cost * Config.TRANSACTION_COST
        
        if cand['type'] == 'LONG':
            if self.cash >= (cost + comm):
                self.cash -= (cost + comm)
                self.positions[cand['sym']] = {'type': 'LONG', 'qty': qty, 'entry_price': exec_price, 'entry_date': date}
        elif cand['type'] == 'SHORT':
            self.cash -= comm 
            self.positions[cand['sym']] = {'type': 'SHORT', 'qty': qty, 'entry_price': exec_price, 'entry_date': date}
        
    def close_position(self, sym, price, date, reason):
        pos = self.positions[sym]
        exec_price = price * (1 - Config.SLIPPAGE) if pos['type'] == 'LONG' else price * (1 + Config.SLIPPAGE)
        
        val = pos['qty'] * exec_price
        comm = val * Config.TRANSACTION_COST
        
        if pos['type'] == 'LONG':
            self.cash += (val - comm)
        else:
            pnl = (pos['entry_price'] - exec_price) * pos['qty']
            self.cash += (pos['qty'] * pos['entry_price']) + pnl - comm
            
        del self.positions[sym]

def run_advanced_system():
    data_res = load_and_select_universe()
    if not data_res[0]: return
    universe, mkt_idx, bench_df = data_res
    
    processed = prepare_data(universe, mkt_idx)
    models, features = train_lgbm_models(processed)
    
    all_dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = [d for d in all_dates if pd.Timestamp(Config.TRAIN_END_DATE) <= d <= pd.Timestamp(Config.BACKTEST_END_DATE)]
    
    strategy = AdvancedStrategy(Config.INITIAL_CAPITAL)
    
    print("--- Starting Backtest ---")
    for date in sim_dates:
        # Snapshot for the day
        snapshot = {s: df.loc[date] for s, df in processed.items() if date in df.index}
        if snapshot:
            strategy.run_day(date, snapshot, models, features)
            
    # Visuals
    equity_df = pd.DataFrame(strategy.equity_curve).set_index('date')
    if not equity_df.empty:
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns)>0 else 0
        print(f"\nBacktest Complete. Final Equity: {equity_df['equity'].iloc[-1]:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        equity_df['equity'].plot(title="Equity Curve")
        plt.show()

if __name__ == "__main__":
    run_advanced_system()