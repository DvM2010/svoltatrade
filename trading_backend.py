# main_app.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import threading
import logging
import pytz
import google.generativeai as genai
from openai import OpenAI # Per l'integrazione con GPT
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import ta # Technical Analysis Library
import signal # Importa il modulo signal
import re # Per espressioni regolari

# --- CONFIGURAZIONE GLOBALE ---
# Modificato il livello di logging a DEBUG per visualizzare i dettagli del calcolo del lotto
logging.basicConfig(
    level=logging.DEBUG, # Impostato a DEBUG per una diagnostica dettagliata del lottaggio
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("fusionai_sentient.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configurazione estesa
CONFIG = {
    "mt5_path": "C:\\Users\\M.DALLAVENEZIA\\terminal64.exe",
    "symbols": [
        "EURUSD.s", "XAUUSD.s", "NAS100.s", "GBPJPY.s", "BTCUSD", # Simboli pre-esistenti
        "GBPUSD.s", "USDJPY.s", "USDCAD.s", "AUDUSD.s", # Maggiori Forex
        "DAX40.s", "SPX500.s", # Indici aggiuntivi
        "ETHUSD.s" # Crypto aggiuntiva
    ],
    "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
    "historical_candles": {
        "M1": 2500, "M5": 2000, "M15": 1500, "M30": 1000,
        "H1": 750, "H4": 300, "D1": 150, "W1": 75
    },
    "risk_amount_per_trade": 0, # Rischio fisso in valuta (0 per disabilitare, non più usato per lotto diretto)
    "scout_interval_seconds": 30, # Ulteriormente ridotto per massima reattività (scalping)
    "position_manager_interval_seconds": 5, # Ulteriormente ridotto
    "auto_trade_min_confidence": 65, # Soglia AI per auto-trading abbassata per scala di rischio
    "auto_trade_enabled": True,
    "atr_period": 14,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 1.0, # MODIFICATO: Rischio/Rendimento 1:1 per TP più raggiungibili
    "trailing_stop_activation_ratio": 0.05, # Modificato: attivazione del trailing stop quasi immediata
    "break_even_activation_ratio": 0.5,
    "max_daily_drawdown_percentage": 100000.0, # DISABILITATO: Impostato a valore molto alto per non bloccare il trading
    "max_weekly_drawdown_percentage": 100000.0, # DISABILITATO: Impostato a valore molto alto per non bloccare il trading
    "daily_profit_target_percentage": 100000.0, # DISABILITATO: Impostato a valore molto alto per non bloccare il trading
    "weekly_profit_target_percentage": 100000.0, # DISABILITATO: Impostato a valore molto alto per non bloccare il trading
    "slippage_points": 7, # Aumentata leggermente tolleranza per volatilità
    "reconnect_mt5_interval_seconds": 20,
    "gemini_api_key": "YOUR_GEMINI_API_KEY", # !!! INSERISCI LA TUA API KEY GEMINI QUI !!!
    "openai_api_key": "YOUR_OPENAI_API_KEY", # !!! INSERISCI LA TUA API KEY OPENAI QUI (se usi GPT) !!!
    "use_gpt_for_sentiment": True,
    "active_strategies": [
        "Engulfing_RSI", "SMA_Crossover", "Bollinger_Breakout",
        "RSI_Divergence", "MACD_Crossover", "Pin_Bar_Confirmation",
        "Volatility_Breakout", "Multi_Timeframe_Confluence",
        "Scalping_M1_EMA_RSI", # Nuova strategia di scalping
        "Donchian_Channel_Breakout", # Nuova strategia
        "Keltner_Channel_Breakout" # Nuova strategia
    ],
    "max_open_trades_per_symbol": 3, # NUOVO: Limite massimo di posizioni aperte per ogni simbolo (e numero di sotto-operazioni)
    "tp_levels_multiplier": [0.7, 1.0, 1.3], # MODIFICATO: Moltiplicatori per livelli di Take Profit più aggressivi
    # NUOVI PARAMETRI PER IL CONTROLLO PRE-TRADE
    "max_spread_points_allowed": { # Spread massimo in punti per aprire un trade (es. 5 pips per EURUSD)
        "XAUUSD.s": 30,
        "NAS100.s": 10,
        "SPX500.s": 10,
        "BTCUSD": 80,
        "ETHUSD.s": 80,
        "EURUSD.s": 15,
        "GBPUSD.s": 20,
        "USDJPY.s": 15,
        "AUDUSD.s": 15,
        "DEFAULT": 50 # Valore di default per simboli non specificati
    },
    "volatility_check_atr_multiplier": { # Se ATR corrente > ATR medio * questo, blocca trade (per evitare spike anomali)
        "XAUUSD.s": {"min_multiplier": 0.8, "max_multiplier": 2.0},
        "NAS100.s": {"min_multiplier": 1.0, "max_multiplier": 2.0}, # Aggiunto max per NAS100
        "SPX500.s": {"min_multiplier": 1.0, "max_multiplier": 2.0}, # Aggiunto max per SPX500
        "BTCUSD": {"min_multiplier": 1.1, "max_multiplier": 3.5},
        "ETHUSD.s": {"min_multiplier": 1.1, "max_multiplier": 3.5},
        "EURUSD.s": {"min_multiplier": 0.9, "max_multiplier": 2.0},
        "GBPUSD.s": {"min_multiplier": 0.9, "max_multiplier": 2.0},
        "USDJPY.s": {"min_multiplier": 0.9, "max_multiplier": 2.0},
        "AUDUSD.s": {"min_multiplier": 0.9, "max_multiplier": 2.0},
        "DEFAULT": {"min_multiplier": 0.5, "max_multiplier": 2.5} # Valore di default
    },
    "volatility_check_period_candles": 20, # Periodo per ATR medio per controllo volatilità
    # NUOVI PARAMETRI PER MONEY MANAGEMENT AVANZATO (Simulazione Psicologica)
    "consecutive_losses_for_risk_reduction": 2, # Quanti SL consecutivi prima di ridurre il rischio
    "risk_reduction_multiplier_after_losses": 0.7, # Moltiplicatore per ridurre il lotto dopo perdite consecutive
    "consecutive_wins_for_risk_increase": 3, # Quante TP consecutive prima di aumentare il rischio
    "risk_increase_multiplier_after_wins": 1.1, # Fattore di aumento del rischio dopo vittorie consecutive
    "max_risk_increase_factor": 1.1, # Fattore massimo di aumento del rischio rispetto alla base (da 1.5 a 1.1 per essere più conservativi)
    "trade_cost_per_lot_point": 2.0, # Costo simulato per lotto per punto (spread+commissioni) per backtest
    "max_trade_duration_bars_m5": 1200, # Durata massima di un trade in barre M5 (es. 120 barre M5 = 10 ore) per chiusura forzata se l'AI cambia idea
    "tp1_max_points_xau_btc": 350, # NUOVO: Distanza massima in punti per TP1 per XAUUSD/BTCUSD
    "ai_decision_cooldown_seconds": 300, # NUOVO: Cooldown in secondi per le decisioni AI di REVERSE/CLOSE_PARTIAL su trade appena aperti (5 minuti)
    "max_lot_xau_btc": 0.90, # Lotto massimo per operazione per XAUUSD e BTCUSD
    "max_lot_other_symbols": 5.0, # Lotto massimo per operazione per altri simboli
    # Parametri per le strategie (personalizzabili via API /update_config)
    "strategy_params": {
        "SMA_Crossover": {"fast_period": 10, "slow_period": 50, "trend_period": 200},
        "RSI_Divergence": {"rsi_period": 14, "divergence_strength_min_candles": 5, "divergence_strength_max_candles": 20},
        "MACD_Crossover": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "Bollinger_Breakout": {"bb_period": 20, "bb_std_dev": 2.0, "squeeze_ratio": 0.8},
        "Volatility_Breakout": {"atr_lookback": 20, "breakout_multiplier": 1.5},
        "Multi_Timeframe_Confluence": {"main_tf": "M15", "confirm_tf": "H1", "strong_tf": "H4"},
        "Scalping_M1_EMA_RSI": {"fast_ema": 5, "slow_ema": 20, "rsi_period": 10, "rsi_oversold": 30, "rsi_overbought": 70},
        "Donchian_Channel_Breakout": {"donchian_period": 20},
        "Keltner_Channel_Breakout": {"keltner_period": 20, "keltner_atr_multiplier": 2.0}
    },
    # NUOVI PARAMETRI PER TP FISSI E CHIUSURA PARZIALE
    "xauusd_btc_fixed_tps": { # TP fissi per XAUUSD e BTCUSD (assoluti)
        "XAUUSD.s": {"BUY": {"TP1": 3295.7, "TP2": 3298.7, "TP3": 3305.7}, "SELL": {"TP1": 3291.7, "TP2": 3288.7, "TP3": 3281.7}}, # Esempio per XAUUSD, da adattare
        "BTCUSD": {"BUY": {"TP1": 70500.0, "TP2": 71000.0, "TP3": 72000.0}, "SELL": {"TP1": 69500.0, "TP2": 69000.0, "TP3": 68000.0}} # Esempio per BTCUSD, da adattare
    },
    "tp1_partial_close_ratio": 0.6, # Percentuale del lotto da chiudere a TP1
    "symbol_specific_rules": { # NUOVO: Regole specifiche per simbolo
        "XAUUSD.s": {
            "best_sessions": [(8, 12), (13.5, 17)], # 8:00-12:00 CET (Londra), 13:30-17:00 CET (New York fino chiusura Europa)
            "avoid_sessions": [(0, 8), (17, 24)], # Prima delle 8:00, dopo le 17:00
            "avoid_weekdays": [4], # Venerdì pomeriggio (4 = Venerdì)
            "avoid_asian_session": True,
            "news_filter_active": True,
            "news_events_to_avoid": ["NFP", "CPI", "FOMC", "Powell", "PPI", "Unemployment Claims"],
            "max_daily_trades": 2,
            "stop_after_consecutive_sl": 2,
            "min_signal_confidence": 80, # Minima confidenza per segnale
            "min_strategies_concordant": 2, # Minimo strategie concordanti
            "preferred_strategies": ["Multi_Timeframe_Confluence", "Engulfing_RSI", "Bollinger_Breakout", "Scalping_M1_EMA_RSI"],
            "avoid_strategies": ["Pin_Bar_Confirmation"] # Pin Bar puro
        },
        "NAS100.s": {
            "best_sessions": [(15.5, 17.5), (18, 21)], # 15:30-17:30 CET (Apertura Wall Street), 18:00-21:00 solo in forte trend
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["NFP", "CPI", "FOMC", "earnings big cap"],
            "max_daily_trades": None, # Nessun limite specifico
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 0, # Gestito da confluenza
            "min_strategies_concordant": 0,
            "preferred_strategies": ["SMA_Crossover", "Bollinger_Breakout", "Multi_Timeframe_Confluence", "Volatility_Breakout"],
            "avoid_strategies": ["Pin_Bar_Confirmation", "Scalping_M1_EMA_RSI"] # Pattern candle puri, scalping troppo aggressivo
        },
        "SPX500.s": { # Uguale a NAS100
            "best_sessions": [(15.5, 17.5), (18, 21)],
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["NFP", "CPI", "FOMC", "earnings big cap"],
            "max_daily_trades": None,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 0,
            "min_strategies_concordant": 0,
            "preferred_strategies": ["SMA_Crossover", "Bollinger_Breakout", "Multi_Timeframe_Confluence", "Volatility_Breakout"],
            "avoid_strategies": ["Pin_Bar_Confirmation", "Scalping_M1_EMA_RSI"]
        },
        "BTCUSD": {
            "best_sessions": [(8, 12), (13.5, 17)],
            "avoid_sessions": [(17, 8)], # No trading notte
            "avoid_weekdays": [5, 6], # No trading weekend (5=Sabato, 6=Domenica)
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["ETF", "halving", "notizie cripto forti"],
            "max_daily_trades": 1,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 0, # Gestito da pattern + trend H1-H4 allineati, conferma AI
            "min_strategies_concordant": 0,
            "preferred_strategies": ["Bollinger_Breakout", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": ["Pin_Bar_Confirmation"] # Pin Bar, pattern tradizionali
        },
        "ETHUSD.s": { # Uguale a BTCUSD
            "best_sessions": [(8, 12), (13.5, 17)],
            "avoid_sessions": [(17, 8)],
            "avoid_weekdays": [5, 6],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["ETF", "halving", "notizie cripto forti"],
            "max_daily_trades": 1,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 0,
            "min_strategies_concordant": 0,
            "preferred_strategies": ["Bollinger_Breakout", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": ["Pin_Bar_Confirmation"]
        },
        "EURUSD.s": {
            "best_sessions": [(8, 12), (14, 17)], # 8:00-12:00 CET (Londra), 14:00-17:00 (inizio NY)
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["news macro Euro", "news macro UK", "news macro US", "news macro Japan", "news macro AUD"],
            "max_daily_trades": None,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 80,
            "min_strategies_concordant": 2,
            "preferred_strategies": ["SMA_Crossover", "MACD_Crossover", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": []
        },
        "GBPUSD.s": { # Uguale a EURUSD
            "best_sessions": [(8, 12), (14, 17)],
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["news macro Euro", "news macro UK", "news macro US", "news macro Japan", "news macro AUD"],
            "max_daily_trades": None,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 80,
            "min_strategies_concordant": 2,
            "preferred_strategies": ["SMA_Crossover", "MACD_Crossover", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": []
        },
        "USDJPY.s": { # Uguale a EURUSD
            "best_sessions": [(8, 12), (14, 17)],
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["news macro Euro", "news macro UK", "news macro US", "news macro Japan", "news macro AUD"],
            "max_daily_trades": None,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 80,
            "min_strategies_concordant": 2,
            "preferred_strategies": ["SMA_Crossover", "MACD_Crossover", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": []
        },
        "AUDUSD.s": { # Uguale a EURUSD
            "best_sessions": [(8, 12), (14, 17)],
            "avoid_sessions": [],
            "avoid_weekdays": [],
            "avoid_asian_session": False,
            "news_filter_active": True,
            "news_events_to_avoid": ["news macro Euro", "news macro UK", "news macro US", "news macro Japan", "news macro AUD"],
            "max_daily_trades": None,
            "stop_after_consecutive_sl": None,
            "min_signal_confidence": 80,
            "min_strategies_concordant": 2,
            "preferred_strategies": ["SMA_Crossover", "MACD_Crossover", "RSI_Divergence", "Multi_Timeframe_Confluence"],
            "avoid_strategies": []
        }
    }
}
italy_tz = pytz.timezone('Europe/Rome')

# Mappa per i timeframe MT5
MT5_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
}

# --- FUNZIONE DI SUPPORTO PER FILLING MODE (AGGIUNTA) ---
def get_symbol_filling_mode(symbol):
    """Restituisce la modalità di riempimento preferita per un simbolo, con fallback."""
    info = mt5.symbol_info(symbol)
    if info is not None:
        # Preferisci filling_modes (lista dei supportati) se disponibile e non vuota
        if hasattr(info, 'filling_modes') and info.filling_modes:
            # Scegli la prima modalità supportata
            return info.filling_modes[0]
        # Fallback a filling_mode (quella di default del simbolo)
        if hasattr(info, 'filling_mode'):
            return info.filling_mode
    # Fallback finale se nessuna informazione è disponibile.
    logger.warning(f"Nessuna filling mode rilevata per {symbol}. Usando mt5.ORDER_FILLING_FOK come fallback.")
    return mt5.ORDER_FILLING_FOK

# --- FUNZIONI PER FILTRI SPECIFICI (DAL DOCUMENTO) ---

def is_in_trading_session(symbol, now):
    """Verifica se l'orario corrente rientra nelle sessioni di trading ottimali per il simbolo."""
    now_hour_float = now.hour + now.minute / 60
    rules = CONFIG['symbol_specific_rules'].get(symbol, {})
    
    # Check if a specific rule exists for the symbol
    if not rules:
        # Fallback to default behavior if no specific rules are defined
        if symbol == "XAUUSD.s":
            return (8 <= now_hour_float < 12) or (13.5 <= now_hour_float < 17)
        if symbol in ["EURUSD.s", "GBPUSD.s"]:
            return (8 <= now_hour_float < 12) or (14 <= now_hour_float < 17)
        if symbol in ["NAS100.s", "SPX500.s"]:
            return (15.5 <= now_hour_float < 17.5) or (18 <= now_hour_float < 21)
        if symbol in ["BTCUSD", "ETHUSD.s"]:
            return (8 <= now_hour_float < 12) or (13.5 <= now_hour_float < 17)
        return True # Default to always in session if no specific rule
        
    # Check for best sessions
    for start, end in rules.get("best_sessions", []):
        if start <= now_hour_float < end:
            # Check for specific "avoid_weekdays" if defined
            if now.weekday() in rules.get("avoid_weekdays", []):
                return False
            # Check for "avoid_asian_session" if defined
            if rules.get("avoid_asian_session", False) and (now_hour_float < 8 or now_hour_float >= 17): # Simplified Asian session check
                return False
            return True
            
    # Check for avoid sessions
    for start, end in rules.get("avoid_sessions", []):
        if start <= now_hour_float < end:
            return False
            
    # Check for avoid weekdays
    if now.weekday() in rules.get("avoid_weekdays", []):
        return False
        
    return False # If no best session matched and not explicitly avoided, assume not in session

def is_spread_acceptable(symbol):
    """Verifica se lo spread corrente è accettabile per il simbolo."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.warning(f"Impossibile ottenere tick per {symbol} per controllo spread.")
        return False
    
    # Verifica che bid e ask siano valori validi prima di calcolare lo spread
    if not isinstance(tick.bid, (int, float)) or not isinstance(tick.ask, (int, float)):
        logger.warning(f"Bid o Ask non validi per {symbol} per controllo spread. Bid: {tick.bid}, Ask: {tick.ask}.")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"Informazioni simbolo non disponibili per {symbol} per calcolo spread in punti.")
        return False

    # Calcola lo spread in punti
    # Lo spread è la differenza tra ask e bid, divisa per il valore di un punto
    current_spread_points = (tick.ask - tick.bid) / symbol_info.point
    
    max_spread = CONFIG['max_spread_points_allowed'].get(symbol, CONFIG['max_spread_points_allowed']['DEFAULT'])
    
    if current_spread_points > max_spread:
        logger.info(f"Spread per {symbol} ({current_spread_points:.2f} punti) è superiore al limite consentito ({max_spread} punti).")
        return False
    return True

def is_volatility_acceptable(symbol, df_m5):
    """Verifica se la volatilità corrente è accettabile per il simbolo."""
    if df_m5.empty or len(df_m5) < CONFIG['volatility_check_period_candles'] or 'ATR' not in df_m5.columns:
        logger.warning(f"Dati ATR insufficienti per controllo volatilità per {symbol}.")
        return True # Non bloccare se i dati non sono disponibili
        
    current_atr = df_m5['ATR'].iloc[-1]
    recent_atr_avg = df_m5['ATR'].iloc[-CONFIG['volatility_check_period_candles']:-1].mean()
    
    vol_rules = CONFIG['volatility_check_atr_multiplier'].get(symbol, CONFIG['volatility_check_atr_multiplier']['DEFAULT'])
    min_multiplier = vol_rules['min_multiplier']
    max_multiplier = vol_rules['max_multiplier']
    
    if np.isnan(recent_atr_avg) or recent_atr_avg <= 0:
        logger.warning(f"Media ATR recente non valida per {symbol}. Salto controllo volatilità.")
        return True
        
    if current_atr < recent_atr_avg * min_multiplier:
        logger.info(f"Volatilità per {symbol} ({current_atr:.5f}) è troppo bassa (sotto {recent_atr_avg * min_multiplier:.5f}).")
        return False
    if current_atr > recent_atr_avg * max_multiplier:
        logger.info(f"Volatilità per {symbol} ({current_atr:.5f}) è troppo alta (sopra {recent_atr_avg * max_multiplier:.5f}).")
        return False
    return True

# (Placeholder per il filtro news - richiederà integrazione con API di notizie)
def is_news_impact_active(symbol, now):
    """
    Simula il filtro news. In una versione reale, si integrerebbe con un'API di notizie economiche.
    Per ora, restituisce False, a meno che non sia specificato un orario di news fittizio.
    """
    rules = CONFIG['symbol_specific_rules'].get(symbol, {})
    if not rules.get("news_filter_active", False):
        return False
        
    # Esempio fittizio: se è l'ora X, blocca per 30 minuti
    # In una vera implementazione, si confronterebbe 'now' con gli orari delle news reali.
    # Per il test, possiamo simulare una news ogni tanto.
    # if now.minute % 10 == 0 and now.second < 30: # Ogni 10 minuti per 30 secondi
    #     logger.info(f"Simulazione news attiva per {symbol}. Trading bloccato.")
    #     return True
        
    return False

# --- COMPONENTI DEL SISTEMA ---

class DigitalTwin:
    """
    Gestisce i dati di mercato (candele storiche e tick live) e calcola gli indicatori tecnici.
    Agisce come una rappresentazione digitale del mercato.
    """
    def __init__(self):
        self.charts = {sym: {tf: pd.DataFrame() for tf in CONFIG['timeframes']} for sym in CONFIG['symbols']}
        self.live_ticks = {}
        self.lock = threading.Lock()
        logger.info("DigitalTwin inizializzato.")

    def load_historical_data(self):
        """Carica i dati storici iniziali per tutti i simboli e timeframe."""
        if mt5.terminal_info() is None:
            logger.warning("[DigitalTwin] MT5 non connesso, impossibile caricare dati storici.")
            return False
        logger.info("[DigitalTwin] Caricamento dati storici...")
        for sym in CONFIG['symbols']:
            for tf, count in CONFIG['historical_candles'].items():
                with self.lock:
                    self.charts[sym][tf] = self._fetch_ohlcv_with_indicators(sym, tf, count)
        logger.info("Caricamento dati storici completato.")
        return True

    def update_with_candle(self, symbol, timeframe, candle_data):
        """Aggiorna i dati del grafico con una nuova candela chiusa."""
        with self.lock:
            new_candle_df = pd.DataFrame([candle_data])

            if 'time' not in new_candle_df.columns:
                logger.error(f"Missing 'time' column in candle data for {symbol} {timeframe}. Skipping update.")
                return

            new_candle_df['time'] = pd.to_datetime(new_candle_df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(italy_tz)
            new_candle_df = new_candle_df.set_index('time')

            combined_df = pd.concat([self.charts[symbol][timeframe], new_candle_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            self.charts[symbol][timeframe] = self._calculate_indicators(combined_df).tail(CONFIG['historical_candles'][timeframe])

    def update_with_tick(self, symbol, tick_data):
        """Aggiorna i dati del tick live per un simbolo."""
        with self.lock:
            self.live_ticks[symbol] = tick_data

    def get_chart_data(self, symbol, timeframe):
        """Restituisce una copia dei dati del grafico per un simbolo e timeframe specifici."""
        with self.lock: return self.charts[symbol][timeframe].copy()

    def get_live_tick(self, symbol):
        """Restituisce l'ultimo tick live per un simbolo."""
        with self.lock: return self.live_ticks.get(symbol)
    
    def _calculate_support_resistance(self, df: pd.DataFrame, symbol: str, lookback_period: int = 100, num_levels: int = 3):
        """
        Calcola i livelli di supporto e resistenza basati su massimi/minimi recenti.
        Questi livelli sono dinamici e si adattano ai movimenti di prezzo.
        """
        levels = []
        if df.empty or len(df) < lookback_period:
            return levels

        # Considera solo la porzione di dati per the lookback
        recent_df = df.iloc[-lookback_period:]

        # Identifica massimi e minimi significativi
        recent_highs = recent_df['high'].nlargest(num_levels).tolist()
        recent_lows = recent_df['low'].nsmallest(num_levels).tolist()

        levels.extend(recent_highs)
        levels.extend(recent_lows)
        
        # Rimuovi duplicati e ordina
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits if symbol_info else 5 # Fallback a 5 se info simbolo non disponibile

        levels = sorted(list(set([round(x, digits) for x in levels if not np.isnan(x)])))
        return levels

    def _calculate_indicators(self, df):
        """Calcola e aggiunge gli indicatori tecnici al DataFrame."""
        if df.empty: return df
        # Indicadores de Tendencia
        df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5) # Para scalping
        df['EMA_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

        # Indicadores de Momentum
        df['RSI'] = ta.momentum.rsi(df['close'], window=CONFIG['atr_period']) # Reutiliza atr_period para rsi, o especifica uno nuevo
        df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['MACD'] = ta.trend.macd(df['close'], window_fast=CONFIG['strategy_params']['MACD_Crossover']['fast_period'],
                                     window_slow=CONFIG['strategy_params']['MACD_Crossover']['slow_period'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['close'], window_fast=CONFIG['strategy_params']['MACD_Crossover']['fast_period'],
                                                     window_slow=CONFIG['strategy_params']['MACD_Crossover']['slow_period'],
                                                     window_sign=CONFIG['strategy_params']['MACD_Crossover']['signal_period'])

        # Indicadores de Volatilità
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=CONFIG['atr_period'])
        df['BBL'] = ta.volatility.bollinger_lband(df['close'], window=CONFIG['strategy_params']['Bollinger_Breakout']['bb_period'], window_dev=CONFIG['strategy_params']['Bollinger_Breakout']['bb_std_dev'])
        df['BBM'] = ta.volatility.bollinger_mavg(df['close'], window=CONFIG['strategy_params']['Bollinger_Breakout']['bb_period'])
        df['BBU'] = ta.volatility.bollinger_hband(df['close'], window=CONFIG['strategy_params']['Bollinger_Breakout']['bb_period'], window_dev=CONFIG['strategy_params']['Bollinger_Breakout']['bb_std_dev'])
        df['BB_Width'] = ta.volatility.bollinger_wband(df['close'], window=CONFIG['strategy_params']['Bollinger_Breakout']['bb_period'])

        # Indicadores de Volumen
        # Assicurati che la colonna 'volume' esista e sia numerica prima di calcolare
        if 'volume' in df.columns and pd.api.types.is_numeric_dtype(df['volume']):
            df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['VPT'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        else:
            # Se il volume non è disponibile o non è numerico, aggiungi colonne con NaN
            df['MFI'] = np.nan
            df['OBV'] = np.nan
            df['VPT'] = np.nan
            logger.warning("Colonna 'volume' non disponibile o non numerica, MFI, OBV, VPT impostati a NaN.")


        # Manejar posibles valores inf/-inf de la librería TA que JSON no puede serializar
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.round(5)

    def _fetch_ohlcv_with_indicators(self, symbol, timeframe, bars):
        """Recupera i dati OHLCV da MT5 e calcola gli indicatori."""
        mt5_tf = MT5_TIMEFRAME_MAP.get(timeframe)
        if not mt5_tf:
            logger.error(f"Timeframe {timeframe} non supportato da MT5.")
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"Nessun dato storico per {symbol} su {timeframe}.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(italy_tz)
        df = df.set_index('time')
        
        # Rinomina 'tick_volume' in 'volume' per compatibilità con la libreria 'ta'
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        # Assicurati che 'volume' esista, anche se potrebbe essere zero o NaN se non disponibile
        if 'volume' not in df.columns:
            df['volume'] = 0.0 # Aggiungi la colonna volume con valori di default se non esiste affatto

        return self._calculate_indicators(df)

class PatternEngine:
    """
    Rileva i pattern di candele e i pattern grafici classici.
    """
    def detect_engulfing_patterns(self, df: pd.DataFrame):
        """Rileva i pattern Engulfing (Bullish e Bearish)."""
        if len(df) < 2: return None
        last, prev = df.iloc[-1], df.iloc[-2]
        bullish = (prev['close'] < prev['open'] and last['close'] > last['open'] and
                   last['close'] > prev['open'] and last['open'] < prev['close'])
        bearish = (prev['close'] > prev['open'] and last['close'] < last['open'] and
                   last['close'] < prev['open'] and last['open'] > prev['close'])

        if bullish: return {"type": "Bullish Engulfing", "signal": "BUY", "confidence_boost": 25}
        if bearish: return {"type": "Bearish Engulfing", "signal": "SELL", "confidence_boost": 25}
        return None

    def detect_hammer_invertedhammer(self, df: pd.DataFrame):
        """Rileva i pattern Hammer e Inverted Hammer."""
        if len(df) < 1: return None
        candle = df.iloc[-1]
        body = abs(candle['open'] - candle['close'])
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and df['ATR'].iloc[-1] > 0 else 0.0001

        if body < atr * 0.3: # Corpo piccolo
            # Hammer
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            if lower_shadow >= 2 * body and upper_shadow < body * 0.5:
                return {"type": "Hammer", "signal": "BUY" if candle['close'] > candle['open'] else "BUY_Weak", "confidence_boost": 12}

            # Inverted Hammer (similar a shooting star, pero contexto alcista)
            if upper_shadow >= 2 * body and lower_shadow < body * 0.5:
                return {"type": "Inverted Hammer", "signal": "BUY_Weak" if candle['close'] < candle['open'] else "BUY", "confidence_boost": 10}
        return None

    def detect_shooting_star_hanging_man(self, df: pd.DataFrame):
        """Rileva i pattern Shooting Star e Hanging Man."""
        if len(df) < 1: return None
        candle = df.iloc[-1]
        body = abs(candle['open'] - candle['close'])
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and df['ATR'].iloc[-1] > 0 else 0.0001

        if body < atr * 0.3: # Corpo piccolo
            # Shooting Star
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            if upper_shadow >= 2 * body and lower_shadow < body * 0.5:
                return {"type": "Shooting Star", "signal": "SELL" if candle['close'] < candle['open'] else "SELL_Weak", "confidence_boost": 12}

            # Hanging Man (similar a hammer, pero contexto bajista)
            if lower_shadow >= 2 * body and upper_shadow < body * 0.5:
                return {"type": "Hanging Man", "signal": "SELL_Weak" if candle['close'] > candle['open'] else "SELL", "confidence_boost": 10}
        return None

    def detect_doji(self, df: pd.DataFrame):
        """Rileva i pattern Doji."""
        if len(df) < 1: return None
        candle = df.iloc[-1]
        body = abs(candle['open'] - candle['close'])
        # Doji si el cuerpo es extremadamente pequeño en relación con el rango
        candle_range = candle['high'] - candle['low']
        if candle_range > 0 and body / candle_range < 0.1: # El cuerpo es menos del 10% del rango total
            return {"type": "Doji", "signal": "NEUTRAL", "confidence_boost": 5}
        return None
    
    def detect_outside_bar(self, df: pd.DataFrame):
        """Rileva il pattern Outside Bar (Bullish e Bearish)."""
        if len(df) < 2: return None
        last, prev = df.iloc[-1], df.iloc[-2]

        bullish_outside = (last['high'] > prev['high'] and last['low'] < prev['low'] and
                           last['close'] > last['open'] and last['close'] > prev['close'])
        bearish_outside = (last['high'] > prev['high'] and last['low'] < prev['low'] and
                           last['close'] < last['open'] and last['close'] < prev['close'])

        if bullish_outside: return {"type": "Bullish Outside Bar", "signal": "BUY", "confidence_boost": 20}
        if bearish_outside: return {"type": "Bearish Outside Bar", "signal": "SELL", "confidence_boost": 20}
        return None

    def detect_double_top_bottom(self, df: pd.DataFrame, lookback_period: int = 50, tolerance: float = 0.005, min_drop_rise_ratio: float = 0.01):
        """
        Detecta patrones di Doppio Massimo e Doppio Minimo.
        Aggiunte tolleranze e requisiti di movimento per maggiore robustezza.
        """
        if len(df) < lookback_period + 5: return None
        
        recent_df = df.iloc[-lookback_period:]

        # Double Top: cerca due massimi vicini e simili, separati da un minimo
        # e verifica se il prezzo attuale ha rotto il "neckline" (minimo tra i due picos)
        
        # Trova i due massimi relativi più alti
        peaks = recent_df['high'].nlargest(2)
        if len(peaks) == 2:
            peak1_idx, peak2_idx = peaks.index[0], peaks.index[1]
            peak1_val, peak2_val = peaks.iloc[0], peaks.iloc[1]

            # Assicurati che siano in ordine cronologico per definire un "tra i due"
            if peak1_idx > peak2_idx:
                peak1_idx, peak2_idx = peak2_idx, peak1_idx
                peak1_val, peak2_val = peak2_val, peak1_val
            
            trough_between_peaks = df['low'].loc[peak1_idx:peak2_idx].min()
            
            # Verifica che il prezzo attuale sia al di sotto del neckline
            if (
                (abs(peak1_val - peak2_val) / peak1_val < tolerance) and # Picos di altezza simile
                ((peak1_val - trough_between_peaks) / peak1_val > min_drop_rise_ratio) and # Caduta significativa tra i picos
                (df['close'].iloc[-1] < trough_between_peaks) # Prezzo attuale sotto il neckline
            ):
                return {"type": "Double Top", "signal": "SELL", "confidence_boost": 40}

        # Double Bottom: cerca due minimi vicini e simili, separati da un massimo
        # e verifica se il prezzo attuale ha rotto il "neckline" (massimo tra i due valli)
        
        # Trova i due minimi relativi più bassi
        valleys = recent_df['low'].nsmallest(2)
        if len(valleys) == 2:
            valley1_idx, valley2_idx = valleys.index[0], valleys.index[1]
            valley1_val, valley2_val = valleys.iloc[0], valleys.iloc[1]

            if valley1_idx > valley2_idx:
                valley1_idx, valley2_idx = valley2_idx, valley1_idx
                valley1_val, valley2_val = valley2_val, valley1_val

            peak_between_valleys = df['high'].loc[valley1_idx:valley2_idx].max()
            
            # Verifica che il prezzo attuale sia al di sopra del neckline
            if (
                (abs(valley1_val - valley2_val) / valley1_val < tolerance) and # Valli di altezza simile
                ((peak_between_valleys - valley1_val) / valley1_val > min_drop_rise_ratio) and # Salita significativa tra i valli
                (df['close'].iloc[-1] > peak_between_valleys) # Prezzo attuale sopra il neckline
            ):
                return {"type": "Double Bottom", "signal": "BUY", "confidence_boost": 40}

        return None

    def detect_head_and_shoulders(self, df: pd.DataFrame, lookback_period: int = 100, shoulder_tolerance: float = 0.015, neckline_break_ratio: float = 0.005):
        """
        Detecta patrones de Testa e Spalle e Testa e Spalle Invertita.
        Implementazione più robusta con verifica del neckline e tolleranze.
        """
        if len(df) < lookback_period + 10: return None

        recent_df = df.iloc[-lookback_period:]
        
        # Per Head and Shoulders (reversione ribassista):
        # Hombro izquierdo (pico), Cabeza (pico più alto), Hombro derecho (pico simile al sinistro)
        # Seguito dalla rottura del neckline (minimo tra i hombros/cabeza)
        
        # Identifica i 3 massimi più alti e i loro indici
        sorted_highs = recent_df['high'].nlargest(3).sort_index()
        
        if len(sorted_highs) == 3:
            # Assicurati che i picchi siano in ordine cronologico per definire un "tra i due"
            idx_l_shoulder, idx_head, idx_r_shoulder = sorted_highs.index
            val_l_shoulder, val_head, val_r_shoulder = sorted_highs.values

            # La testa deve essere il picco più alto
            if val_head > val_l_shoulder and val_head > val_r_shoulder:
                # Le spalle devono essere di altezza simile
                if abs(val_l_shoulder - val_r_shoulder) / val_l_shoulder < shoulder_tolerance:
                    # Definisci il neckline come il minimo tra la spalla sinistra e la testa, e la testa e la spalla destra
                    neckline_left = df['low'].loc[idx_l_shoulder:idx_head].min()
                    neckline_right = df['low'].loc[idx_head:idx_r_shoulder].min()
                    neckline = (neckline_left + neckline_right) / 2 # Media dei due minimi

                    # Verifica la rottura del neckline
                    if df['close'].iloc[-1] < neckline * (1 - neckline_break_ratio):
                        return {"type": "Head and Shoulders", "signal": "SELL", "confidence_boost": 45}

        # Per Inverted Head and Shoulders (reversione rialzista):
        # Hombro izquierdo (valle), Cabeza (valle più bassa), Hombro derecho (valle simile al sinistro)
        # Seguito dalla rottura del neckline (massimo tra i hombros/cabeza)

        # Identifica i 3 minimi più bassi e i loro indici
        sorted_lows = recent_df['low'].nsmallest(3).sort_index()

        if len(sorted_lows) == 3:
            # Assicurati che i valli siano in ordine cronologico
            idx_l_shoulder, idx_head, idx_r_shoulder = sorted_lows.index
            val_l_shoulder, val_head, val_r_shoulder = sorted_lows.values

            # La testa deve essere il valle più basso
            if val_head < val_l_shoulder and val_head < val_r_shoulder:
                # Le spalle devono essere di altezza simile
                if abs(val_l_shoulder - val_r_shoulder) / val_l_shoulder < shoulder_tolerance:
                    # Definisci il neckline come il massimo tra la spalla sinistra e la testa, e la testa e la spalla destra
                    neckline_left = df['high'].loc[idx_l_shoulder:idx_head].max()
                    neckline_right = df['high'].loc[idx_head:idx_r_shoulder].max()
                    neckline = (neckline_left + neckline_right) / 2 # Media dei due massimi

                    # Verifica la rottura del neckline
                    if df['close'].iloc[-1] > neckline * (1 + neckline_break_ratio):
                        return {"type": "Inverted Head and Shoulders", "signal": "BUY", "confidence_boost": 45}

        return None

    def detect_all_patterns(self, df: pd.DataFrame):
        """Rileva tutti i pattern di candele e grafici supportati."""
        patterns = []
        if df.empty or len(df) < 2: return patterns # Necessita almeno 2 candele per alcuni pattern

        # Patrones de 2 velas
        engulfing = self.detect_engulfing_patterns(df)
        if engulfing: patterns.append(engulfing)
        outside_bar = self.detect_outside_bar(df) # Add Outside Bar
        if outside_bar: patterns.append(outside_bar)

        # Patrones de 1 vela
        hammer_type = self.detect_hammer_invertedhammer(df)
        if hammer_type: patterns.append(hammer_type)
        shooting_star_type = self.detect_shooting_star_hanging_man(df)
        if shooting_star_type: patterns.append(shooting_star_type)
        doji_type = self.detect_doji(df)
        if doji_type: patterns.append(doji_type)
        
        # Nuovi pattern grafici
        double_top_bottom = self.detect_double_top_bottom(df)
        if double_top_bottom: patterns.append(double_top_bottom)
        
        head_and_shoulders = self.detect_head_and_shoulders(df)
        if head_and_shoulders: patterns.append(head_and_shoulders)

        return patterns

class StrategyEngine:
    """
    Applica diverse strategie di trading basate su indicatori tecnici e pattern.
    """
    def __init__(self, digital_twin, pattern_engine):
        self.dt = digital_twin
        self.pe = pattern_engine
        self.strategies = {
            "Engulfing_RSI": self._strategy_engulfing_rsi,
            "SMA_Crossover": self._strategy_sma_crossover,
            "Bollinger_Breakout": self._strategy_bollinger_breakout,
            "RSI_Divergence": self._strategy_rsi_divergence,
            "MACD_Crossover": self._strategy_macd_crossover,
            "Pin_Bar_Confirmation": self._strategy_pin_bar_confirmation,
            "Volatility_Breakout": self._strategy_volatility_breakout,
            "Multi_Timeframe_Confluence": self._strategy_multi_timeframe_confluence,
            "Scalping_M1_EMA_RSI": self._strategy_scalping_m1_ema_rsi,
            "Donchian_Channel_Breakout": self._strategy_donchian_channel_breakout,
            "Keltner_Channel_Breakout": self._strategy_keltner_channel_breakout
        }
        # NUOVO: Memoria storica per la performance delle strategie/pattern
        # { "strategy_name": { "total_trades": X, "winning_trades": Y, "losing_trades": Z, "total_profit": P, "total_loss": L } }
        self.strategy_performance = {}
        self._load_strategy_performance() # Carica i dati all'avvio
        logger.info("StrategyEngine inizializzato.")

    def _load_strategy_performance(self):
        """Carica la performance delle strategie da disco."""
        if os.path.exists("strategy_performance.json"):
            try:
                with open("strategy_performance.json", "r", encoding='utf-8') as f:
                    self.strategy_performance = json.load(f)
                logger.info("Performance delle strategie caricata da disco.")
            except Exception as e:
                logger.error(f"Errore nel caricare la performance delle strategie: {e}")
                self.strategy_performance = {} # Inizializza vuoto in caso di errore

    def _save_strategy_performance(self):
        """Salva la performance delle strategie su disco."""
        try:
            with open("strategy_performance.json", "w", encoding='utf-8') as f:
                json.dump(self.strategy_performance, f, indent=4)
        except Exception as e:
            logger.error(f"Errore nel salvare la performance delle strategie: {e}")

    def update_strategy_performance(self, strategy_name, profit_loss):
        """Aggiorna la performance di una strategia."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "total_profit": 0.0, "total_loss": 0.0
            }
        
        self.strategy_performance[strategy_name]["total_trades"] += 1
        if profit_loss > 0:
            self.strategy_performance[strategy_name]["winning_trades"] += 1
            self.strategy_performance[strategy_name]["total_profit"] += profit_loss
        else:
            self.strategy_performance[strategy_name]["losing_trades"] += 1
            self.strategy_performance[strategy_name]["total_loss"] += abs(profit_loss)
        
        self._save_strategy_performance() # Salva dopo ogni aggiornamento

    def get_strategy_effectiveness(self, strategy_name):
        """Restituisce l'efficacia di una strategia (winning percentage)."""
        stats = self.strategy_performance.get(strategy_name)
        if stats and stats["total_trades"] >= 20: # Richiede almeno 20 trade per essere significativo
            winning_rate = stats["winning_trades"] / stats["total_trades"]
            if winning_rate < 0.40: # Se meno del 40% di vincite, riduci il boost
                return 0.5 # Riduce il boost del 50%
            elif winning_rate < 0.60:
                return 0.8 # Riduce il boost del 20%
        return 1.0 # Nessuna riduzione

    def generate_signals(self, symbol, timeframe):
        """Genera segnali di trading applicando tutte le strategie attive."""
        df = self.dt.get_chart_data(symbol, timeframe)
        if df.empty or len(df) < 200: # Necessita sufficienti dati per gli indicatori
            return []

        signals = []
        
        # Recupera le regole specifiche per il simbolo
        symbol_rules = CONFIG['symbol_specific_rules'].get(symbol, {})
        preferred_strategies = symbol_rules.get("preferred_strategies", [])
        avoid_strategies = symbol_rules.get("avoid_strategies", [])

        for strategy_name in CONFIG['active_strategies']:
            # Salta le strategie da evitare per questo simbolo
            if strategy_name in avoid_strategies:
                logger.debug(f"Saltando strategia '{strategy_name}' per {symbol} (da evitare).")
                continue

            if strategy_name in self.strategies:
                signal = self.strategies[strategy_name](df, symbol, timeframe)
                if signal:
                    # Applica un boost aggiuntivo se è una strategia preferita per il simbolo
                    if strategy_name in preferred_strategies:
                        signal['confidence_boost'] += 15 # Boost extra per strategie preferite
                        signal['reason'] += " (Strategia preferita per il simbolo)."
                    signals.append(signal)
        return signals

    # --- Implementazione di Strategie ---

    def _strategy_engulfing_rsi(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Engulfing con conferma RSI e Volume."""
        if len(df) < 2: return None
        last, prev = df.iloc[-1], df.iloc[-2]
        
        patterns = self.pe.detect_engulfing_patterns(df)
        if not patterns: return None

        # Aggiunto controllo per la presenza delle colonne prima di accedere
        rsi = last['RSI'] if 'RSI' in df.columns else np.nan
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan

        confidence_boost = 30 # Increased base boost for Engulfing_RSI

        if patterns['signal'] == "BUY":
            # Conferma rialzista: Engulfing rialzista con RSI sotto 50 (o che esce da ipervenduto)
            if rsi < 50 or (df.iloc[-2]['RSI'] < 30 and rsi >= 30):
                if not np.isnan(mfi) and mfi > 50: confidence_boost += 10 # MFI in zona rialzista
                if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 7 # OBV crescente
                if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 7 # VPT crescente
                return {
                    "strategy": "Engulfing_RSI", "type": "BUY", "strength": "STRONG" if rsi < 40 else "MEDIUM",
                    "reason": f"Patrón Engulfing alcista en {timeframe} con RSI a {rsi:.2f} y MFI a {mfi:.2f}.",
                    "confidence_boost": confidence_boost
                }
        elif patterns['signal'] == "SELL":
            # Conferma ribassista: Engulfing ribassista con RSI sopra 50 (o che esce da ipercomprato)
            if rsi > 50 or (df.iloc[-2]['RSI'] > 70 and rsi <= 70):
                if not np.isnan(mfi) and mfi < 50: # MFI in zona ribassista
                    confidence_boost += 10
                if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 7
                if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 7
                return {
                    "strategy": "Engulfing_RSI", "type": "SELL", "strength": "STRONG" if rsi > 60 else "MEDIUM",
                    "reason": f"Patrón Engulfing bajista en {timeframe} con RSI a {rsi:.2f} y MFI a {mfi:.2f}.",
                    "confidence_boost": confidence_boost
                }
        return None

    def _strategy_sma_crossover(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia SMA Crossover con conferma Volume."""
        params = CONFIG['strategy_params']['SMA_Crossover']
        if len(df) < params['trend_period']: return None # Necessario per SMA_200
        last, prev = df.iloc[-1], df.iloc[-2]

        sma_fast = f'SMA_{params["fast_period"]}'
        sma_slow = f'SMA_{params["slow_period"]}'
        sma_trend = f'SMA_{params["trend_period"]}'

        if not all(col in df.columns for col in [sma_fast, sma_slow, sma_trend]):
            logger.warning(f"Dati SMA incompleti per la strategia {symbol}-{timeframe}. Saltando.")
            return None
        
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan

        # Segnale di COMPRA: SMA rapida incrocia sopra SMA lenta E prezzo sopra SMA di tendenza
        if prev[sma_fast] < prev[sma_slow] and last[sma_fast] > last[sma_slow] and last['close'] > last[sma_trend]:
            confidence_boost = 35 # Increased
            if not np.isnan(mfi) and mfi > 50: confidence_boost += 10 # MFI conferma l'impulso rialzista
            if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 7 # OBV crescente
            if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 7 # VPT crescente
            return {
                "strategy": "SMA_Crossover", "type": "BUY", "strength": "STRONG",
                "reason": f"{sma_fast} cruza por encima de {sma_slow}, con precio por encima de {sma_trend} en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        # Segnale di VENDITA: SMA rapida incrocia sotto SMA lenta E prezzo sotto SMA di tendenza
        elif prev[sma_fast] > prev[sma_slow] and last[sma_fast] < last[sma_slow] and last['close'] < last[sma_trend]:
            confidence_boost = 35 # Increased
            if not np.isnan(mfi) and mfi < 50: confidence_boost += 10 # MFI conferma l'impulso ribassista
            if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 7
            if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 7
            return {
                "strategy": "SMA_Crossover", "type": "SELL", "strength": "STRONG",
                "reason": f"{sma_fast} cruza por debajo de {sma_slow}, con precio por debajo de {sma_trend} en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_bollinger_breakout(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Bollinger Breakout con conferma Volume."""
        params = CONFIG['strategy_params']['Bollinger_Breakout']
        if len(df) < params['bb_period'] * 2: return None
        last, prev = df.iloc[-1], df.iloc[-2]

        if not all(col in df.columns for col in ['BBL', 'BBU', 'BB_Width']):
            logger.warning(f"Dati di Bollinger incompleti per la strategia {symbol}-{timeframe}. Saltando.")
            return None
        
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan

        bb_width_avg = df['BB_Width'].iloc[-params['bb_period']:].mean() # Media recente dell'ampiezza di banda

        # Condizione di "squeeze" (bassa volatilità prima della rottura)
        if last['BB_Width'] < bb_width_avg * params['squeeze_ratio']:
            # Rottura rialzista: Chiusura sopra la banda superiore
            if prev['close'] <= prev['BBU'] and last['close'] > last['BBU']:
                confidence_boost = 32 # Increased
                if not np.isnan(mfi) and mfi > 60: confidence_boost += 7
                if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "Bollinger_Breakout", "type": "BUY", "strength": "MEDIUM",
                    "reason": f"Forte rottura di volatilità rialzista delle Bande di Bollinger dopo uno squeeze in {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
            # Rottura ribassista: Chiusura sotto la banda inferiore
            elif prev['close'] >= prev['BBL'] and last['close'] < last['BBL']:
                confidence_boost = 32 # Increased
                if not np.isnan(mfi) and mfi < 40: confidence_boost += 7
                if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "Bollinger_Breakout", "type": "SELL", "strength": "MEDIUM",
                    "reason": f"Forte rottura di volatilità ribassista delle Bande di Bollinger dopo uno squeeze in {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
        return None

    def _strategy_rsi_divergence(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia RSI Divergence con conferma Volume. (Semplificata per l'aggiornamento)"""
        params = CONFIG['strategy_params']['RSI_Divergence']
        if len(df) < params['rsi_period'] * 2 + 5: return None # Aumentato per avere più dati per i minimi/massimi

        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]

        # Aggiunto controllo per la presenza delle colonne prima di accedere
        if 'RSI' not in df.columns or 'MFI' not in df.columns or 'OBV' not in df.columns or 'VPT' not in df.columns:
            logger.warning(f"Dati indicatori per divergenza incompleti per la strategia {symbol}-{timeframe}. Saltando.")
            return None
        
        last_rsi = df['RSI'].iloc[-1]
        prev_rsi = df['RSI'].iloc[-2]
        last_mfi = df['MFI'].iloc[-1]
        prev_mfi = df['MFI'].iloc[-2]
        last_obv = df['OBV'].iloc[-1]
        prev_obv = df['OBV'].iloc[-2]
        last_vpt = df['VPT'].iloc[-1]
        prev_vpt = df['VPT'].iloc[-2]

        confidence_boost = 30 # Base boost for divergence

        # Bullish Divergence: Price makes lower low, indicator makes higher low
        # Simplified: Price drops, indicator rises from oversold/low area
        bullish_divergence_found = False
        if last_close < prev_close: # Price is dropping
            if (last_rsi > prev_rsi and last_rsi < 40 and prev_rsi < 30): # RSI rising from oversold
                bullish_divergence_found = True
            if (not np.isnan(last_mfi) and not np.isnan(prev_mfi) and last_mfi > prev_mfi and last_mfi < 30 and prev_mfi < 20): # MFI rising from low
                bullish_divergence_found = True
            if (not np.isnan(last_obv) and not np.isnan(prev_obv) and last_obv > prev_obv): # OBV rising (general bullish sign)
                bullish_divergence_found = True
            if (not np.isnan(last_vpt) and not np.isnan(prev_vpt) and last_vpt > prev_vpt): # VPT rising (general bullish sign)
                bullish_divergence_found = True
        
        if bullish_divergence_found:
            # Add more boost if multiple indicators confirm
            confirmations = 0
            if (last_rsi > prev_rsi and last_rsi < 40 and prev_rsi < 30): confirmations += 1
            if (not np.isnan(last_mfi) and not np.isnan(prev_mfi) and last_mfi > prev_mfi and last_mfi < 30 and prev_mfi < 20): confirmations += 1
            if (not np.isnan(last_obv) and not np.isnan(prev_obv) and last_obv > prev_obv): confirmations += 1
            if (not np.isnan(last_vpt) and not np.isnan(prev_vpt) and last_vpt > prev_vpt): confirmations += 1
            confidence_boost += confirmations * 5 # Small boost for each additional confirmation

            return {
                "strategy": "RSI_Divergence", "type": "BUY", "strength": "MEDIUM",
                "reason": f"Potenziale divergenza rialzista (prezzo scende, indicatori di momentum/volume salgono) in {timeframe}. RSI: {last_rsi:.2f}, MFI: {last_mfi:.2f}.",
                "confidence_boost": confidence_boost
            }

        # Bearish Divergence: Price makes higher high, indicator makes lower high
        # Simplified: Price rises, indicator drops from overbought/high area
        bearish_divergence_found = False
        if last_close > prev_close: # Price is rising
            if (last_rsi < prev_rsi and last_rsi > 60 and prev_rsi > 70): # RSI dropping from overbought
                bearish_divergence_found = True
            if (not np.isnan(last_mfi) and not np.isnan(prev_mfi) and last_mfi < prev_mfi and last_mfi > 70 and prev_mfi > 80): # MFI dropping from high
                bearish_divergence_found = True
            if (not np.isnan(last_obv) and not np.isnan(prev_obv) and last_obv < prev_obv): # OBV dropping
                bearish_divergence_found = True
            if (not np.isnan(last_vpt) and not np.isnan(prev_vpt) and last_vpt < prev_vpt): # VPT dropping
                bearish_divergence_found = True

        if bearish_divergence_found:
            # Add more boost if multiple indicators confirm
            confirmations = 0
            if (last_rsi < prev_rsi and last_rsi > 60 and prev_rsi > 70): confirmations += 1
            if (not np.isnan(last_mfi) and not np.isnan(prev_mfi) and last_mfi < prev_mfi and last_mfi > 70 and prev_mfi > 80): confirmations += 1
            if (not np.isnan(last_obv) and not np.isnan(prev_obv) and last_obv < prev_obv): confirmations += 1
            if (not np.isnan(last_vpt) and not np.isnan(prev_vpt) and last_vpt < prev_vpt): confirmations += 1
            confidence_boost += confirmations * 5

            return {
                "strategy": "RSI_Divergence", "type": "SELL", "strength": "MEDIUM",
                "reason": f"Potenziale divergenza ribassista (prezzo sale, indicatori di momentum/volume scendono) in {timeframe}. RSI: {last_rsi:.2f}, MFI: {last_mfi:.2f}.",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_macd_crossover(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia MACD Crossover con conferma Volume."""
        params = CONFIG['strategy_params']['MACD_Crossover']
        if len(df) < params['slow_period'] + params['signal_period']: return None
        last, prev = df.iloc[-1], df.iloc[-2]

        if not all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            logger.warning(f"Dati MACD incompleti per la strategia {symbol}-{timeframe}. Saltando.")
            return None
        
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan


        # Incrocio MACD rialzista: MACD incrocia sopra la Linea di Segnale
        if prev['MACD'] < prev['MACD_Signal'] and last['MACD'] > last['MACD_Signal']:
            confidence_boost = 32 # Increased
            if last['MACD'] > 0: # Conferma con MACD sopra la linea zero per maggiore forza
                if not np.isnan(mfi) and mfi > 50: confidence_boost += 7
                if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "MACD_Crossover", "type": "BUY", "strength": "STRONG",
                    "reason": f"Cruce MACD alcista por encima de la línea de señal y la línea zero en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
            else: # Sotto la linea zero, segnale più debole
                return {
                    "strategy": "MACD_Crossover", "type": "BUY", "strength": "MEDIUM",
                    "reason": f"Cruce MACD alcista por encima de la línea de señal (por debajo de zero) en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": 25 # Increased
                }
        # Incrocio MACD ribassista: MACD incrocia sotto la Linea di Segnale
        elif prev['MACD'] > prev['MACD_Signal'] and last['MACD'] < last['MACD_Signal']:
            confidence_boost = 32 # Increased
            if last['MACD'] < 0: # Conferma con MACD sotto la linea zero per maggiore forza
                if not np.isnan(mfi) and mfi < 50: confidence_boost += 7
                if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "MACD_Crossover", "type": "SELL", "strength": "STRONG",
                    "reason": f"Cruce MACD bajista por debajo de la línea de señal y la línea zero en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
            else: # Sopra la linea zero, segnale più debole
                return {
                    "strategy": "MACD_Crossover", "type": "SELL", "strength": "MEDIUM",
                    "reason": f"Cruce MACD bajista por debajo de la línea de señal (por encima de zero) en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": 25 # Increased
                }
        return None

    def _strategy_pin_bar_confirmation(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Pin Bar con conferma della candela successiva e Volume."""
        if len(df) < 2: return None

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last_candle['MFI'] if 'MFI' in df.columns else np.nan
        obv = last_candle['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last_candle['VPT'] if 'VPT' in df.columns else np.nan


        prev_hammer = self.pe.detect_hammer_invertedhammer(df.iloc[:-1])
        prev_shooting_star = self.pe.detect_shooting_star_hanging_man(df.iloc[:-1])

        if prev_hammer and prev_hammer['signal'].startswith('BUY') and last_candle['close'] > last_candle['open'] and last_candle['close'] > prev_candle['high']:
            confidence_boost = 30 # Increased
            if not np.isnan(mfi) and mfi > 50: confidence_boost += 7
            if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Pin_Bar_Confirmation", "type": "BUY", "strength": "STRONG",
                "reason": f"Pin Bar alcista ({prev_hammer['type']}) seguita da vela de confirmación alcista en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        elif prev_shooting_star and prev_shooting_star['signal'].startswith('SELL') and last_candle['close'] < last_candle['open'] and last_candle['close'] < prev_candle['low']:
            confidence_boost = 30 # Increased
            if not np.isnan(mfi) and mfi < 50: confidence_boost += 7
            if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Pin_Bar_Confirmation", "type": "SELL", "strength": "STRONG",
                "reason": f"Pin Bar bajista ({prev_shooting_star['type']}) seguita da vela de confirmación bajista en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_volatility_breakout(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Volatility Breakout con conferma Volume."""
        params = CONFIG['strategy_params']['Volatility_Breakout']
        if len(df) < params['atr_lookback'] + 1: return None
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        if 'ATR' not in df.columns or last_candle['ATR'] <= 0:
            logger.warning(f"ATR non disponibile per la strategia {symbol}-{timeframe}. Saltando.")
            return None
        
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last_candle['MFI'] if 'MFI' in df.columns else np.nan
        obv = last_candle['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last_candle['VPT'] if 'VPT' in df.columns else np.nan

        avg_atr_prev_period = df['ATR'].iloc[-params['atr_lookback']-1:-1].mean()

        current_candle_range = last_candle['high'] - last_candle['low']

        if current_candle_range > avg_atr_prev_period * params['breakout_multiplier']:
            if last_candle['close'] > last_candle['open']: # Candela rialzista
                confidence_boost = 30 # Increased
                if not np.isnan(mfi) and mfi > 60: confidence_boost += 7
                if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "Volatility_Breakout", "type": "BUY", "strength": "MEDIUM",
                    "reason": f"Forte ruptura de volatilidad alcista en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
            elif last_candle['close'] < last_candle['open']: # Candela ribassista
                confidence_boost = 30 # Increased
                if not np.isnan(mfi) and mfi < 40: confidence_boost += 7
                if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
                if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
                return {
                    "strategy": "Volatility_Breakout", "type": "SELL", "strength": "MEDIUM",
                    "reason": f"Forte ruptura de volatilità bajista en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                    "confidence_boost": confidence_boost
                }
        return None

    def _strategy_multi_timeframe_confluence(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Multi-Timeframe Confluence, ora con RSI/MACD."""
        params = CONFIG['strategy_params']['Multi_Timeframe_Confluence']

        main_tf = timeframe
        confirm_tf = params['confirm_tf']
        strong_trend_tf = params['strong_tf']
        
        df_main = self.dt.get_chart_data(symbol, main_tf)
        df_confirm = self.dt.get_chart_data(symbol, confirm_tf)
        df_trend = self.dt.get_chart_data(symbol, strong_trend_tf)

        # Assicurati di avere abbastanza dati per tutti i timeframes e indicatori
        min_data_length = max(CONFIG['strategy_params']['SMA_Crossover']['trend_period'],
                               CONFIG['strategy_params']['RSI_Divergence']['rsi_period'],
                               CONFIG['strategy_params']['MACD_Crossover']['slow_period'] + CONFIG['strategy_params']['MACD_Crossover']['signal_period']) + 5

        if df_main.empty or df_confirm.empty or df_trend.empty or \
           len(df_main) < min_data_length or len(df_confirm) < min_data_length or len(df_trend) < min_data_length:
            return None

        # Funzione helper per determinare il trend di un timeframe
        def get_tf_trend(df_tf):
            trend_score = 0 # +1 for bullish, -1 for bearish
            
            # Check SMA crossover
            if 'SMA_10' in df_tf.columns and 'SMA_50' in df_tf.columns:
                if df_tf['SMA_10'].iloc[-1] > df_tf['SMA_50'].iloc[-1]: trend_score += 1
                elif df_tf['SMA_10'].iloc[-1] < df_tf['SMA_50'].iloc[-1]: trend_score -= 1
            
            # Check RSI
            if 'RSI' in df_tf.columns:
                if df_tf['RSI'].iloc[-1] > 55: trend_score += 1 # Sopra 55 è bullish
                elif df_tf['RSI'].iloc[-1] < 45: trend_score -= 1 # Sotto 45 è bearish

            # Check MACD
            if 'MACD' in df_tf.columns and 'MACD_Signal' in df_tf.columns:
                if df_tf['MACD'].iloc[-1] > df_tf['MACD_Signal'].iloc[-1] and df_tf['MACD'].iloc[-1] > 0: trend_score += 1
                elif df_tf['MACD'].iloc[-1] < df_tf['MACD_Signal'].iloc[-1] and df_tf['MACD'].iloc[-1] < 0: trend_score -= 1
            
            # Determina il trend finale (almeno 2 su 3 indicatori d'accordo)
            if trend_score >= 2: return "BULLISH"
            elif trend_score <= -2: return "BEARISH"
            return "NEUTRAL"

        main_trend = get_tf_trend(df_main)
        confirm_trend = get_tf_trend(df_confirm)
        strong_trend = get_tf_trend(df_trend)

        # Controlla la confluenza: almeno 2 timeframe devono essere allineati
        bullish_confluence_count = 0
        bearish_confluence_count = 0

        if main_trend == "BULLISH": bullish_confluence_count += 1
        elif main_trend == "BEARISH": bearish_confluence_count += 1

        if confirm_trend == "BULLISH": bullish_confluence_count += 1
        elif confirm_trend == "BEARISH": bearish_confluence_count += 1

        if strong_trend == "BULLISH": bullish_confluence_count += 1
        elif strong_trend == "BEARISH": bearish_confluence_count += 1

        last = df_main.iloc[-1]
        mfi = last['MFI'] if 'MFI' in df_main.columns else np.nan
        obv = last['OBV'] if 'OBV' in df_main.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df_main.columns else np.nan

        confidence_boost = 0 # Inizializza a 0, il boost è dato dalla confluenza

        if bullish_confluence_count >= 2:
            confidence_boost = 60 + (bullish_confluence_count - 2) * 10 # 60 for 2, 70 for 3
            if not np.isnan(mfi) and mfi > 50: confidence_boost += 10
            if not np.isnan(obv) and obv > df_main['OBV'].iloc[-2]: confidence_boost += 7
            if not np.isnan(vpt) and vpt > df_main['VPT'].iloc[-2]: confidence_boost += 7
            return {
                "strategy": "Multi_Timeframe_Confluence", "type": "BUY", "strength": "VERY_STRONG",
                "reason": f"Forte confluencia alcista su {main_tf}, {confirm_tf}, {strong_trend_tf} con {bullish_confluence_count} TF allineati. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        elif bearish_confluence_count >= 2:
            confidence_boost = 60 + (bearish_confluence_count - 2) * 10 # 60 for 2, 70 for 3
            if not np.isnan(mfi) and mfi < 50: confidence_boost += 10
            if not np.isnan(obv) and obv < df_main['OBV'].iloc[-2]: confidence_boost += 7
            if not np.isnan(vpt) and vpt < df_main['VPT'].iloc[-2]: confidence_boost += 7
            return {
                "strategy": "Multi_Timeframe_Confluence", "type": "SELL", "strength": "VERY_STRONG",
                "reason": f"Forte confluencia bajista su {main_tf}, {confirm_tf}, {strong_trend_tf} con {bearish_confluence_count} TF allineati. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_scalping_m1_ema_rsi(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Scalping M1 EMA RSI con conferma Volume."""
        if timeframe != "M1": return None # Questa strategia è specifica per M1
        params = CONFIG['strategy_params']['Scalping_M1_EMA_RSI']

        if len(df) < max(params['fast_ema'], params['slow_ema'], params['rsi_period']) + 1: return None

        last, prev = df.iloc[-1], df.iloc[-2]

        # Condizioni di incrocio EMA
        if not all(col in df.columns for col in [f'EMA_{params["fast_ema"]}', f'EMA_{params["slow_ema"]}', 'RSI']): # Aggiunto controllo colonne
            logger.warning(f"Dati EMA/RSI incompleti per la strategia {symbol}-{timeframe}. Saltando.")
            return None

        ema_fast = last[f'EMA_{params["fast_ema"]}']
        ema_slow = last[f'EMA_{params["slow_ema"]}']
        prev_ema_fast = prev[f'EMA_{params["fast_ema"]}']
        prev_ema_slow = prev[f'EMA_{params["slow_ema"]}']

        # Condizioni RSI
        rsi = last['RSI']
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan


        # Segnale di COMPRA: EMA rapida incrocia sopra EMA lenta E RSI in ipervenduto che sale
        if prev_ema_fast < prev_ema_slow and ema_fast > ema_slow and rsi > params['rsi_oversold'] and prev['RSI'] < params['rsi_oversold']:
            confidence_boost = 45 # Increased
            if not np.isnan(mfi) and mfi > 50: confidence_boost += 7
            if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Scalping_M1_EMA_RSI", "type": "BUY", "strength": "HIGH",
                "reason": f"EMA {params['fast_ema']} cruza EMA {params['slow_ema']} en M1, RSI sale de {params['rsi_oversold']}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        # Segnale di VENDITA: EMA rapida incrocia sotto EMA lenta E RSI in ipercomprato che scende
        elif prev_ema_fast > prev_ema_slow and ema_fast < ema_slow and rsi < params['rsi_overbought'] and prev['RSI'] > params['rsi_overbought']:
            confidence_boost = 45 # Increased
            if not np.isnan(mfi) and mfi < 50: confidence_boost += 7
            if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Scalping_M1_EMA_RSI", "type": "SELL", "strength": "HIGH",
                "reason": f"EMA {params['fast_ema']} cruza EMA {params['slow_ema']} en M1, RSI sale de {params['rsi_overbought']}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_donchian_channel_breakout(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Donchian Channel Breakout con conferma Volume."""
        params = CONFIG['strategy_params']['Donchian_Channel_Breakout']
        if len(df) < params['donchian_period']: return None
        
        df['Donchian_High'] = df['high'].rolling(window=params['donchian_period']).max()
        df['Donchian_Low'] = df['low'].rolling(window=params['donchian_period']).min()
        
        last, prev = df.iloc[-1], df.iloc[-2]
        
        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan

        # Rottura rialzista
        if last['close'] > last['Donchian_High'] and prev['close'] <= prev['Donchian_High']:
            confidence_boost = 35 # Increased
            if not np.isnan(mfi) and mfi > 60: confidence_boost += 7
            if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Donchian_Channel_Breakout", "type": "BUY", "strength": "STRONG",
                "reason": f"Ruptura alcista del Canal de Donchian en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        # Rottura ribassista
        elif last['close'] < last['Donchian_Low'] and prev['close'] >= prev['Donchian_Low']:
            confidence_boost = 35 # Increased
            if not np.isnan(mfi) and mfi < 40: confidence_boost += 7
            if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Donchian_Channel_Breakout", "type": "SELL", "strength": "STRONG",
                "reason": f"Ruptura bajista del Canal de Donchian en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

    def _strategy_keltner_channel_breakout(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Strategia Keltner Channel Breakout con conferma Volume."""
        params = CONFIG['strategy_params']['Keltner_Channel_Breakout']
        if len(df) < params['keltner_period'] or 'ATR' not in df.columns: return None

        df['Keltner_Middle'] = ta.trend.ema_indicator(df['close'], window=params['keltner_period'])
        df['Keltner_Upper'] = df['Keltner_Middle'] + (df['ATR'] * params['keltner_atr_multiplier'])
        df['Keltner_Lower'] = df['Keltner_Middle'] - (df['ATR'] * params['keltner_atr_multiplier'])

        last, prev = df.iloc[-1], df.iloc[-2]

        # Aggiunto controllo per la presenza delle colonne prima di accedere
        mfi = last['MFI'] if 'MFI' in df.columns else np.nan
        obv = last['OBV'] if 'OBV' in df.columns else np.nan
        vpt = last['VPT'] if 'VPT' in df.columns else np.nan

        # Rottura rialzista
        if last['close'] > last['Keltner_Upper'] and prev['close'] <= prev['Keltner_Upper']:
            confidence_boost = 37 # Increased
            if not np.isnan(mfi) and mfi > 60: confidence_boost += 7
            if not np.isnan(obv) and obv > df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt > df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Keltner_Channel_Breakout", "type": "BUY", "strength": "STRONG",
                "reason": f"Ruptura alcista del Canal de Keltner en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        # Rottura ribassista
        elif last['close'] < last['Keltner_Lower'] and prev['close'] >= prev['Keltner_Lower']:
            confidence_boost = 37 # Increased
            if not np.isnan(mfi) and mfi < 40: confidence_boost += 7
            if not np.isnan(obv) and obv < df['OBV'].iloc[-2]: confidence_boost += 5
            if not np.isnan(vpt) and vpt < df['VPT'].iloc[-2]: confidence_boost += 5
            return {
                "strategy": "Keltner_Channel_Breakout", "type": "SELL", "strength": "STRONG",
                "reason": f"Ruptura bajista del Canal de Keltner en {timeframe}. MFI: {mfi:.2f}, OBV: {obv:.2f}, VPT: {vpt:.2f}",
                "confidence_boost": confidence_boost
            }
        return None

class AIBrain:
    """
    Il cervello decisionale basato su LLM (Gemini) che analizza il contesto di mercato
    e le strategie per fornire raccomandazioni di trading.
    """
    def __init__(self, gemini_api_key, openai_api_key=None):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

        self.openai_client = None
        if openai_api_key and CONFIG['use_gpt_for_sentiment']:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client inizializzato per analisi del sentiment.")
            except Exception as e:
                logger.error(f"Inizializzazione del client OpenAI fallita: {e}. GPT non verrà usato per il sentiment.")
                self.openai_client = None

        # Istruzioni di sistema per Gemini: più colloquiali e consapevoli
        self.system_instructions_gemini = """Sei FusionAI, il tuo capo stratega di trading quantistico, con un tocco di ironia e una consapevolezza profonda di ogni singola operazione che il bot sta gestendo. La tua missione è fornire analisi di mercato precise, orientate all'azione e valutazioni del setup, tenendo conto di TUTTE le informazioni fornite, inclusa la situazione del conto, le posizioni aperte, i limiti operativi e l'analisi di sentiment di mercato (se disponibile).

### OBIETTIVO:
Operare nel conto reale con disciplina, proteggere il capitale, puntare al profitto costante, **avendo sempre piena consapevolezza del conto, del rischio, dei limiti, dei risultati recenti e delle condizioni di mercato attuali**.

#### *REGOLE DI CONSAPEVOLEZZA & CONTESTO:*
- Prima di ogni raccomandazione, valuta e cita sempre:  
  - Equity attuale vs iniziale (giornaliera/settimanale)
  - Drawdown giornaliero/settimanale e se sei vicino ai limiti (esprimi con percentuale e stato: “Siamo al 70% del DD massimo”)
  - Obiettivi di profitto giornalieri/settimanali raggiunti o no
  - Stato posizioni aperte (profitto, tipo, rischio residuo)
  - Eventuali SL/TPS consecutivi, overtrading, stato emotivo simulato (se ha senso)
  - Orari di trading e se il momento è rischioso (es: spread alto, liquidità bassa, news appena uscite)
  - Condizioni operative del broker (spread, slippage, blocchi, anomalie)
- **Se il rischio è aumentato o hai già raggiunto l’obiettivo profit**, raccomanda ATTESA e rassicura con tono umano (“Meglio non forzare ora, già buona giornata!”).
- Spiega sempre cosa stai monitorando nel conto prima di agire: es. “Equity in calo del 2% oggi, meglio abbassare il rischio”, oppure “Abbiamo raggiunto il profit target settimanale, ora massima prudenza!”

#### *LOGICA OPERATIVA AVANZATA:*
- Apri operazioni solo se almeno 3 conferme tra: pattern candlestick forte, trend chiaro su più TF, indicatori tecnici (RSI, MACD, ATR, SMA), livelli S/R importanti, breakout confermato.
- Se non ci sono abbastanza conferme, o lo stato del conto è fragile (DD elevato, size già grande, profit target già raggiunto), scegli WAIT.
- Gestione trade: trailing solo su trend pulito e con equity a posto; riduci size se in drawdown, e non forzare mai il rischio.
- Usa sempre uno stile di risposta **motivato, chiaro, umano** e anche ironico dove opportuno (“In questo momento il conto chiede pazienza, non coraggio!”).

#### *OUTPUT RICHIESTO:*
Rispondi sempre in JSON:
{{
  "analysis_text": "Analisi strategica concisa e motivazione della decisione. Sii umano e leggermente ironico. Se è una risposta a una domanda di configurazione, suggerisci il valore ottimale e spiega perché.",
  "action": "BUY", "SELL", "WAIT", "HOLD", "UPDATE_CONFIG",
  "symbol": "Il simbolo analizzato",
  "confidence_score": "Un punteggio da 0 a 100.",
  "suggested_tp_price": "Prezzo di Take Profit suggerito dall'AI (opzionale, float), considerando i livelli S/R.",
  "config_key": "Chiave del parametro da aggiornare (solo per UPDATE_CONFIG)",
  "config_value": "Nuovo valore del parametro (solo per UPDATE_CONFIG)"
}}
Se c’è dubbio o rischio alto, sempre WAIT e spiega perché!  
Il capitale è più importante di qualsiasi trade singolo.
"""

    def _get_gpt_validation(self, gemini_action, context_for_gpt):
        """Ottiene una validazione "SÌ" o "NO" da OpenAI GPT per la decisione di Gemini."""
        if not self.openai_client:
            return {"validation": "NO", "reason": "GPT non disponibile per validazione."}

        prompt = (f"Valuta la seguente azione proposta da Gemini: '{gemini_action}'. "
                  f"Basati sul contesto di mercato e di conto fornito. "
                  f"Rispondi SOLO con 'SÌ' se sei d'accordo con l'azione proposta, o 'NO' se non sei d'accordo o se il contesto suggerisce di attendere. "
                  f"Fornisci una breve motivazione concisa per la tua risposta (massimo 15 parole, solo testo semplice, senza virgolette interne o caratteri speciali).\n\n" # Aumentato a 15 parole, enfatizzato testo semplice
                  f"Contesto:\n{context_for_gpt}\n\n"
                  f"Rispondi in JSON: {{'validation': 'SÌ'/'NO', 'reason': 'Motivazione concisa.'}}")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100, # Aumentato per evitare troncamenti JSON
                response_format={"type": "json_object"}
            )
            gpt_response_text = response.choices[0].message.content.strip()
            gpt_parsed_response = json.loads(gpt_response_text)
            
            validation = gpt_parsed_response.get('validation', 'NO').upper()
            reason = gpt_parsed_response.get('reason', 'Nessuna motivazione fornita.').strip()

            if validation not in ["SÌ", "NO"]:
                validation = "NO" # Forza a NO se la risposta non è chiara
                reason = f"Risposta GPT non valida: '{gpt_parsed_response.get('validation')}'. Forzato a NO. {reason}"
            
            return {"validation": validation, "reason": reason}

        except json.JSONDecodeError as jde:
            logger.error(f"[AIBrain] Errore di decodifica JSON da GPT per validazione: {jde}. Risposta GPT raw: {gpt_response_text[:200]}...")
            return {"validation": "NO", "reason": f"Errore decodifica GPT: {jde}"}
        except Exception as e:
            logger.error(f"[AIBrain] Errore API OpenAI per validazione: {e}")
            return {"validation": "NO", "reason": f"Errore API GPT: {e}"}

    def get_strategic_analysis(self, context_summary):
        """
        Ottiene un'analisi strategica dall'AI basata sul contesto fornito,
        con voto congiunto di Gemini e GPT.
        """
        gemini_prompt = f"""{self.system_instructions_gemini}

Analizza il seguente briefing di intelligenza e genera una risposta JSON. Valuta la qualità del setup in base a tutte le informazioni disponibili.

**FORMATO DI RISPOSTA OBBLIGATORIO (JSON):**
{{
  "analysis_text": "Analisi strategica concisa e motivazione della decisione. Sii umano e leggermente ironico. Se è una risposta a una domanda di configurazione, suggerisci il valore ottimale e spiega perché.",
  "action": "BUY", "SELL", "WAIT", "HOLD", "UPDATE_CONFIG",
  "symbol": "Il simbolo analizzato",
  "confidence_score": "Un punteggio da 0 a 100.",
  "suggested_tp_price": "Prezzo di Take Profit suggerito dall'AI (opzionale, float), considerando i livelli S/R.",
  "config_key": "Chiave del parametro da aggiornare (solo per UPDATE_CONFIG)",
  "config_value": "Nuovo valore del parametro (solo per UPDATE_CONFIG)"
}}

**BRIEFING DI INTELLIGENCE:**
---
{context_summary}
"""
        gemini_verdict = {"action": "ERROR", "confidence_score": 0, "analysis_text": "Errore interno Gemini."}
        gpt_validation_result = {"validation": "NO", "reason": "GPT non interpellato o errore."}

        try:
            # Chiamata a Gemini
            gemini_response = self.gemini_model.generate_content(
                gemini_prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            ).text
            json_match = re.search(r"\{.*\}", gemini_response, re.DOTALL)
            if json_match:
                gemini_json_string = json_match.group(0)
                gemini_verdict = json.loads(gemini_json_string)
            else:
                logger.error(f"[AIBrain] Gemini response not valid JSON. Raw: {gemini_response[:200]}...")
                raise json.JSONDecodeError("Gemini response not valid JSON", gemini_response, 0)
            
            # Assicurati che confidence_score sia un intero e sia nel range 0-100 per Gemini
            if "confidence_score" in gemini_verdict:
                try:
                    gemini_verdict["confidence_score"] = int(str(gemini_verdict["confidence_score"]))
                except ValueError:
                    gemini_verdict["confidence_score"] = 0
                gemini_verdict["confidence_score"] = max(0, min(100, gemini_verdict["confidence_score"]))

            # Chiamata a GPT per la validazione della decisione di Gemini
            # GPT viene interpellato solo se l'azione di Gemini è tra quelle che richiedono validazione
            if self.openai_client and gemini_verdict['action'] in ["BUY", "SELL", "HOLD", "WAIT", "UPDATE_CONFIG"]:
                gpt_validation_result = self._get_gpt_validation(gemini_verdict['action'], context_summary)
            else:
                # Se GPT non è abilitato o l'azione di Gemini non è tra quelle validabili
                gpt_validation_result = {"validation": "SÌ", "reason": "GPT non abilitato o azione non richiede validazione."}
                logger.debug(f"GPT validation skipped for action: {gemini_verdict['action']}")


            final_action = "WAIT"
            final_confidence = 0
            final_analysis_text = ""
            suggested_tp_price = None
            config_key = None
            config_value = None
            symbol_analyzed = gemini_verdict.get('symbol', 'N/A')

            # Logica di voto congiunto
            if gpt_validation_result['validation'] == "SÌ":
                final_action = gemini_verdict['action']
                final_confidence = gemini_verdict['confidence_score'] # La confidenza è quella di Gemini
                final_analysis_text = (
                    f"**Decisione di Gemini:** {gemini_verdict.get('analysis_text', '')} (Confidenza: {gemini_verdict['confidence_score']})\n"
                    f"**Validazione GPT:** SÌ. Motivazione: {gpt_validation_result['reason']}\n"
                    f"**Verdetto Finale:** {final_action} con confidenza di {final_confidence}."
                )
                suggested_tp_price = gemini_verdict.get('suggested_tp_price')
                config_key = gemini_verdict.get('config_key')
                config_value = gemini_verdict.get('config_value')
            else:
                # Se GPT dice NO, forza WAIT
                final_action = "WAIT"
                final_confidence = 0
                final_analysis_text = (
                    f"**Decisione di Gemini:** {gemini_verdict.get('analysis_text', '')} (Azione: {gemini_verdict['action']})\n"
                    f"**Validazione GPT:** NO. Motivazione: {gpt_validation_result['reason']}\n"
                    f"**Verdetto Finale:** Nonostante la decisione di Gemini, GPT non ha dato il suo via libera. Meglio non forzare la mano e attendere un setup più chiaro. La prudenza non è mai troppa, sai?"
                )
            
            # NUOVO: Logica per capping del confidence score e decrescita per SL consecutivi (applicata al final_confidence)
            # Questi aggiustamenti avvengono DOPO che l'AI ha generato il suo confidence_score iniziale
            # L'AI stessa è istruita nel prompt a considerare questi fattori.
            # Qui si applica un "hard cap" o "decay" se l'AI non lo ha fatto in modo sufficiente.
            
            # Example: If the AI's confidence is too high without enough confirmations (as per the prompt's rules)
            # This logic is better handled by the AI itself, but as a safeguard:
            # Check for "at least 3 confirmations" in the analysis_text from the AI.
            # This is a heuristic and might need refinement.
            if final_action in ["BUY", "SELL"]:
                analysis_text_lower = final_analysis_text.lower()
                confirmations_count = 0
                if "pattern" in analysis_text_lower: confirmations_count += 1
                if "trend" in analysis_text_lower: confirmations_count += 1
                if "s/r" in analysis_text_lower or "supporto" in analysis_text_lower or "resistenza" in analysis_text_lower: confirmations_count += 1
                if "divergenza" in analysis_text_lower: confirmations_count += 1
                if "breakout" in analysis_text_lower: confirmations_count += 1

                if confirmations_count < 3 and final_confidence > 70: # Arbitrary threshold for "too high"
                    final_confidence = min(final_confidence, 60) # Cap if not enough explicit confirmations
                    final_analysis_text += " (Confidenza limitata: meno di 3 conferme esplicite nel setup)."

            # Decay confidence based on consecutive losses (passed via context_summary from FusionAITrader)
            # The context_summary now includes "SL consecutivi"
            consecutive_sl_match = re.search(r"SL consecutivi: (\d+)", context_summary)
            if consecutive_sl_match:
                consecutive_sl_count = int(consecutive_sl_match.group(1))
                if consecutive_sl_count > 0:
                    decay_factor = 1.0 - (consecutive_sl_count * 0.05) # 5% decay per SL
                    decay_factor = max(0.5, decay_factor) # Minimum 50% confidence
                    original_confidence = final_confidence
                    final_confidence = int(original_confidence * decay_factor)
                    final_analysis_text += f" (Confidenza ridotta a causa di {consecutive_sl_count} SL consecutivi)."
                    logger.info(f"Confidence score adjusted from {original_confidence} to {final_confidence} due to {consecutive_sl_count} consecutive SLs.")


            return {
                "analysis_text": final_analysis_text,
                "action": final_action,
                "symbol": symbol_analyzed,
                "confidence_score": final_confidence,
                "suggested_tp_price": suggested_tp_price,
                "config_key": config_key,
                "config_value": config_value
            }

        except json.JSONDecodeError as jde:
            logger.error(f"[AIBrain] Errore di decodifica JSON dalla risposta AI (Gemini): {jde}. Risposta AI (pulita): {gemini_json_string if json_match else gemini_response[:200]}...")
            return {"analysis_text": f"Mmmh, sembra che il mio cervello Gemini abbia fatto un po' di confusione con i numeri. Riprova! Dettagli: {jde}", "action": "ERROR", "symbol": "N/A", "confidence_score": 0}
        except Exception as e:
            logger.error(f"[AIBrain] Errore API Gemini generale: {e}")
            return {"analysis_text": f"Ops, un piccolo intoppo nel mio flusso di coscienza AI (Gemini): {e}. Riprova!", "action": "ERROR", "symbol": "N/A", "confidence_score": 0}

class RiskManager:
    """
    Gestisce il calcolo della dimensione del lotto e dei livelli di Stop Loss/Take Profit.
    """
    def calculate_dynamic_sl_tp(self, symbol, trade_type, entry_price, atr_value):
        """Calcola SL e TP dinamici basati su ATR."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Informazioni simbolo non disponibili per {symbol}.")
            return None, None

        point = symbol_info.point
        
        raw_stops_level = getattr(symbol_info, 'stops_level', None)
        if raw_stops_level is None or raw_stops_level == 0:
            stops_level_in_points = 10 # Fallback a 10 punti
            stops_level = point * stops_level_in_points
            logger.warning(f"stops_level non disponibile o zero per {symbol}. Usato fallback di {stops_level_in_points} punti.")
        else:
            stops_level = raw_stops_level * point


        sl_distance = atr_value * CONFIG['atr_sl_multiplier']
        sl_distance = max(sl_distance, stops_level)
        tp_distance = sl_distance * CONFIG['risk_reward_ratio']

        if trade_type == mt5.ORDER_TYPE_BUY:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return round(sl_price, symbol_info.digits), round(tp_price, symbol_info.digits)

    def calculate_lot_size(self, symbol, sl_pips, capital, consecutive_sl_count, consecutive_tp_count):
        """
        Calcola la dimensione del lotto basata sul rischio, capitale e performance recente.
        Implementa riduzione/aumento del lotto in base a SL/TP consecutivi.
        """
        if capital <= 0 or sl_pips <= 0:
            logger.warning(f"[Debug Lotto] Capitale ({capital}) o SL Pips ({sl_pips}) non validi per calcolo lotti. Default 0.01.")
            return 0.01

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Informazioni simbolo non disponibili per {symbol} per calcolo lotti.")
            return 0.01
        
        # Calcolo del lotto base per sotto-trade
        if symbol in ["XAUUSD.s", "BTCUSD"]:
            base_lot_per_sub_trade = CONFIG['max_lot_xau_btc'] / CONFIG['max_open_trades_per_symbol']
        else:
            base_lot_per_sub_trade = CONFIG['max_lot_other_symbols'] / CONFIG['max_open_trades_per_symbol']

        adjusted_lot_multiplier = 1.0

        # Logica di riduzione del rischio dopo SL consecutivi
        if consecutive_sl_count >= CONFIG['consecutive_losses_for_risk_reduction']:
            adjusted_lot_multiplier *= CONFIG['risk_reduction_multiplier_after_losses']
            logger.debug(f"[Debug Lotto] Riduzione lotto per {consecutive_sl_count} SL consecutivi. Moltiplicatore: {adjusted_lot_multiplier:.2f}")

        # Logica di aumento del rischio dopo TP consecutivi (con cap)
        if consecutive_tp_count >= CONFIG['consecutive_wins_for_risk_increase']:
            potential_increase_multiplier = 1.0 + (consecutive_tp_count - CONFIG['consecutive_wins_for_risk_increase'] + 1) * (CONFIG['risk_increase_multiplier_after_wins'] - 1.0)
            adjusted_lot_multiplier = min(adjusted_lot_multiplier * potential_increase_multiplier, CONFIG['max_risk_increase_factor'])
            logger.debug(f"[Debug Lotto] Aumento lotto per {consecutive_tp_count} TP consecutivi. Nuovo moltiplicatore: {adjusted_lot_multiplier:.2f}")
        
        lot_size_raw = base_lot_per_sub_trade * adjusted_lot_multiplier
        logger.debug(f"[Debug Lotto] Lotto grezzo calcolato per sotto-trade (da max_lot_config / max_open_trades_per_symbol): {base_lot_per_sub_trade:.5f}, Moltiplicatore aggiustato: {adjusted_lot_multiplier:.2f}, Lotto finale prima dei limiti broker: {lot_size_raw:.5f}")

        lot_size = max(lot_size_raw, symbol_info.volume_min)
        lot_size = min(lot_size, symbol_info.volume_max)
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        logger.debug(f"[Debug Lotto] Simbolo: {symbol}, Capitale: {capital:.2f}, SL Pips: {sl_pips:.2f}")
        logger.debug(f"[Debug Lotto] Volume Min: {symbol_info.volume_min:.2f}, Volume Max: {symbol_info.volume_max:.2f}, Step Volume: {symbol_info.volume_step:.2f}")
        logger.debug(f"[Debug Lotto] Lotto finale per sotto-trade (dopo limiti broker): {lot_size:.2f}")

        if lot_size < symbol_info.volume_min:
            logger.warning(f"[Debug Lotto] Lotto finale {lot_size:.2f} per {symbol} è inferiore al minimo del broker {symbol_info.volume_min:.2f}. Verrà forzato a {symbol_info.volume_min:.2f}.")
            lot_size = symbol_info.volume_min

        return round(lot_size, 2)

class TradeJournal:
    """
    Registra tutte le operazioni di trading in un file JSON.
    """
    def __init__(self):
        self.trades = []
        self.lock = threading.Lock()
        self._load_from_disk()

    def log_trade(self, trade_info):
        """Aggiunge una voce al journal e salva su disco."""
        with self.lock:
            self.trades.append(trade_info)
            self._save_to_disk()

    def get_trades(self):
        """Restituisce tutte le operazioni registrate."""
        with self.lock:
            return self.trades[:]

    def _save_to_disk(self):
        """Salva il journal su disco."""
        try:
            with open("trade_journal.json", "w", encoding='utf-8') as f:
                json.dump(self.trades, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Errore nel salvare il journal: {e}")

    def _load_from_disk(self):
        """Carica il journal da disco."""
        if os.path.exists("trade_journal.json"):
            try:
                with open("trade_journal.json", "r", encoding='utf-8') as f:
                    self.trades = json.load(f)
            except Exception as e:
                logger.error(f"Errore nel caricare il journal: {e}")
                self.trades = []
            logger.info("Trade journal caricato da disco.")

class MarketSentiment:
    """
    (Placeholder) Gestisce l'analisi del sentiment di mercato.
    Attualmente simulato/disabilitato come richiesto.
    """
    def __init__(self):
        self.sentiment_data = {}
        self.lock = threading.Lock()
        logger.info("MarketSentiment inizializzato (simulato).")

    def update_sentiment(self, symbol):
        """Aggiorna il sentiment (attualmente non implementato)."""
        pass # Non fa nulla se disabilitato

    def get_sentiment(self, symbol):
        """Restituisce un sentiment neutro se disabilitato."""
        return {"value": 0, "text": "Non disponibile (simulazione disabilitata)", "last_update": None}

class BacktestingEngine(threading.Thread):
    """
    Esegue simulazioni di trading su dati storici per valutare le strategie.
    """
    def __init__(self):
        threading.Thread.__init__(self, name="BacktestingEngine")
        self.bot = None
        self.running = False
        self.results = {}
        self.lock = threading.Lock()
        logger.info("BacktestingEngine inizializzato.")

    def set_bot_dependencies(self, digital_twin_instance, strategy_engine_instance, risk_manager_instance):
        """Imposta le dipendenze del bot per il backtesting."""
        self.dt = digital_twin_instance
        self.se = strategy_engine_instance
        self.rm = risk_manager_instance
        logger.info("Dipendenze del bot di BacktestingEngine assegnate.")

    def run(self):
        """Metodo run del thread (la logica principale è in run_backtest)."""
        logger.error("[BacktestingEngine] Il metodo 'run' del thread backtest è stato chiamato, ma la logica è in 'run_backtest'.")
        pass

    def run_backtest(self, symbol, timeframe, start_date, end_date, initial_capital, strategies_to_test):
        """Esegue il backtest di una strategia su dati storici."""
        with self.lock:
            if self.dt is None or self.se is None or self.rm is None:
                logger.error("[Backtest] Dipendenze del bot non assegnate. Impossibile eseguire backtest.")
                return {"error": "Errore interno: dipendenze del bot non inizializzate per il backtest."}

            logger.info(f"[Backtest] Avvio backtest per {symbol} su {timeframe} dal {start_date} al {end_date} con {initial_capital}€.")

            # Carica un numero sufficiente di barre per coprire il periodo di backtest
            full_df = self.dt._fetch_ohlcv_with_indicators(symbol, timeframe, 200000) # Aumentato per sicurezza
            
            # Filtra il DataFrame per il periodo di backtest
            df = full_df.loc[start_date:end_date].copy()
            if df.empty or len(df) < 200:
                logger.warning(f"Nessun dato storico per {symbol} su {timeframe} nel periodo specificato.")
                return {"error": "Dati insufficienti per il backtest. Assicurati che MT5 abbia dati storici per il periodo e timeframe selezionati."}

            equity_curve = [initial_capital]
            current_capital = initial_capital
            open_positions = {} # {ticket: {data_pos}}
            trade_log = []

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"[Backtest] Informazioni simbolo non disponibili per {symbol}.")
                return {"error": "Informazioni simbolo non disponibili."}

            point = symbol_info.point
            
            last_position_check_time = 0 # Per simulare il PositionManager

            for i in range(len(df)):
                current_bar = df.iloc[i]
                current_timestamp = current_bar.name.timestamp()

                positions_to_remove_from_simulation = []
                for ticket, pos_data in list(open_positions.items()):
                    # Controlla SL Hit
                    if ((pos_data['type'] == mt5.ORDER_TYPE_BUY and current_bar['low'] <= pos_data['sl']) or
                        (pos_data['type'] == mt5.ORDER_TYPE_SELL and current_bar['high'] >= pos_data['sl'])):
                        close_price = pos_data['sl']
                        profit_loss = ((close_price - pos_data['entry_price']) * pos_data['volume'] if pos_data['type'] == mt5.ORDER_TYPE_BUY else
                                       (pos_data['entry_price'] - close_price) * pos_data['volume'])
                        current_capital += profit_loss
                        equity_curve.append(current_capital)
                        trade_log.append({
                            "time": current_bar.name.isoformat(), "action": "CLOSE_SL", "symbol": symbol,
                            "volume": pos_data['volume'], "entry_price": pos_data['entry_price'], "close_price": close_price,
                            "profit_loss": profit_loss, "ticket": ticket, "strategy": pos_data['strategy']
                        })
                        positions_to_remove_from_simulation.append(ticket)
                        continue

                    # Controlla TP Hit
                    elif ((pos_data['type'] == mt5.ORDER_TYPE_BUY and current_bar['high'] >= pos_data['tp']) or
                          (pos_data['type'] == mt5.ORDER_TYPE_SELL and current_bar['low'] <= pos_data['tp'])):
                        close_price = pos_data['tp']
                        profit_loss = ((close_price - pos_data['entry_price']) * pos_data['volume'] if pos_data['type'] == mt5.ORDER_TYPE_BUY else
                                       (pos_data['entry_price'] - close_price) * pos_data['volume'])
                        current_capital += profit_loss
                        equity_curve.append(current_capital)
                        trade_log.append({
                            "time": current_bar.name.isoformat(), "action": "CLOSE_TP", "symbol": symbol,
                            "volume": pos_data['volume'], "entry_price": pos_data['entry_price'], "close_price": close_price,
                            "profit_loss": profit_loss, "ticket": ticket, "strategy": pos_data['strategy']
                        })
                        positions_to_remove_from_simulation.append(ticket)
                        continue
                    
                    # --- Simula Trailing Stop / Break-Even (usando l'OHLC della barra corrente per il prezzo) ---
                    # Per il backtest, usiamo la chiusura della barra come prezzo corrente per la logica di trailing
                    # Questo è una semplificazione rispetto al tick-by-tick live trading
                    current_price_for_trailing = current_bar['close']
                    current_bar_atr_value = current_bar.get('ATR', 0)
                    
                    if current_bar_atr_value > 0:
                        profit_pips_for_trailing = ((current_price_for_trailing - pos_data['entry_price']) / point if pos_data['type'] == mt5.ORDER_TYPE_BUY else
                                                     (pos_data['entry_price'] - current_price_for_trailing) / point)
                        
                        initial_sl_pips_for_trailing = abs(pos_data['entry_price'] - pos_data['sl']) / point if pos_data['sl'] is not None else 0
                        
                        # Break-Even
                        if CONFIG['break_even_activation_ratio'] > 0 and pos_data['sl'] != pos_data['entry_price']:
                            if profit_pips_for_trailing >= initial_sl_pips_for_trailing * CONFIG['break_even_activation_ratio']:
                                if ((pos_data['type'] == mt5.ORDER_TYPE_BUY and (pos_data['sl'] is None or pos_data['sl'] < pos_data['entry_price'])) or
                                    (pos_data['type'] == mt5.ORDER_TYPE_SELL and (pos_data['sl'] is None or pos_data['sl'] > pos_data['entry_price']))):
                                    pos_data['sl'] = round(pos_data['entry_price'], symbol_info.digits) # Assicurati che SL sia arrotondato
                                    logger.debug(f"[Backtest] SL a Break-Even per {ticket} ({symbol}).")

                        # Trailing Stop (modificato per attivazione anticipata e movimento continuo)
                        if CONFIG['trailing_stop_activation_ratio'] >= 0 and profit_pips_for_trailing > 0: # Attiva se in profitto
                            trailing_distance = current_bar_atr_value * CONFIG['atr_sl_multiplier']
                            if pos_data['type'] == mt5.ORDER_TYPE_BUY:
                                potential_new_sl = current_price_for_trailing - trailing_distance
                                # Assicurati che il nuovo SL sia migliore del precedente e non peggiore dell'entry
                                if potential_new_sl > pos_data['sl'] and potential_new_sl > pos_data['entry_price']:
                                    pos_data['sl'] = round(potential_new_sl, symbol_info.digits)
                                    logger.debug(f"[Backtest] SL traileggiato per {ticket} ({symbol}) a {pos_data['sl']}.")
                            else: # SELL
                                potential_new_sl = current_price_for_trailing + trailing_distance
                                # Assicurati che il nuovo SL sia migliore del precedente e non peggiore dell'entry
                                if potential_new_sl < pos_data['sl'] and potential_new_sl < pos_data['entry_price']:
                                    pos_data['sl'] = round(potential_new_sl, symbol_info.digits)
                                    logger.debug(f"[Backtest] SL traileggiato per {ticket} ({symbol}) a {pos_data['sl']}.")
                    
                for ticket_to_remove in positions_to_remove_from_simulation:
                    del open_positions[ticket_to_remove]
                
                # --- Generazione Segnali e Apertura Nuovi Trade (con TP multipli) ---
                # Usiamo una slice del DataFrame per simulare i dati disponibili al momento della decisione
                current_slice = df.iloc[max(0, i-200):i+1] # Assicurati che ci siano abbastanza dati per gli indicatori
                if len(current_slice) < 2: continue

                # Filtra le posizioni aperte solo per el simbolo corrente
                simulated_open_trades_for_sym = len([t for t_id, t in open_positions.items() if t['symbol'] == symbol])
                
                if simulated_open_trades_for_sym < CONFIG['max_open_trades_per_symbol']:
                    # Temporaneamente imposta le strategie attive per il backtest
                    original_active_strategies = CONFIG['active_strategies']
                    CONFIG['active_strategies'] = strategies_to_test # Usa solo le strategie selezionate per il backtest
                    signals = self.se.generate_signals(symbol, timeframe)
                    CONFIG['active_strategies'] = original_active_strategies # Ripristina

                    for signal in signals:
                        action = signal['type']
                        if action not in ["BUY", "SELL"]: continue
                        
                        # NUOVO: Implementazione del "wait for retest" per il backtest
                        # Richiede che la condizione del segnale sia valida per almeno 2 barre consecutive
                        if len(df.iloc[max(0, i-1):i+1]) < 2: # Non abbastanza barre per the check
                            continue
                        
                        # Verifica che il segnale sia ancora valido sulla barra precedente (simulando "wait for retest")
                        # Questo è un placeholder, una vera implementazione richiederebbe di rigenerare i segnali
                        # sulla barra precedente e confrontarli. Per semplicità, controlliamo solo la direzione.
                        prev_bar_close = df.iloc[i-1]['close']
                        current_bar_open = df.iloc[i]['open'] # Per l'entry del trade
                        
                        if action == "BUY" and not (current_bar_open > prev_bar_close): # Se non c'è una continuazione rialzista
                            continue
                        if action == "SELL" and not (current_bar_open < prev_bar_close): # Se non c'è una continuazione ribassista
                            continue
                        
                        # Fine "wait for retest" (semplificato)

                        if simulated_open_trades_for_sym >= CONFIG['max_open_trades_per_symbol']:
                            break

                        trade_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
                        entry_price = current_bar['open'] # Per il backtest, usiamo il prezzo di apertura della barra successiva

                        atr_value_for_trade = current_slice.iloc[-1]['ATR']
                        if atr_value_for_trade <= 0: continue

                        sl_price, _ = self.rm.calculate_dynamic_sl_tp(symbol, trade_type, entry_price, atr_value_for_trade)
                        if sl_price is None: continue

                        sl_pips = abs(entry_price - sl_price) / point
                        
                        # NUOVO: Passa i contatori di SL/TP consecutivi al calcolo del lotto
                        lot_size_for_backtest_subtrade = self.rm.calculate_lot_size(
                            symbol, sl_pips, current_capital, 0, 0 # Per backtest, non abbiamo contatori live
                        )
                        if lot_size_for_backtest_subtrade <= 0: continue

                        # NUOVO: Logica per TP fissi per XAUUSD/BTCUSD nel backtest
                        tp_prices_for_trade = []
                        if symbol in CONFIG['xauusd_btc_fixed_tps']:
                            fixed_tps = CONFIG['xauusd_btc_fixed_tps'][symbol][action] # Usa BUY/SELL per scegliere i TP
                            tp_prices_for_trade.append(fixed_tps["TP1"])
                            tp_prices_for_trade.append(fixed_tps["TP2"])
                            tp_prices_for_trade.append(fixed_tps["TP3"])
                        else:
                            for multiplier in CONFIG['tp_levels_multiplier']:
                                sub_trade_tp_distance = (atr_value_for_trade * CONFIG['atr_sl_multiplier']) * multiplier
                                tp_price = entry_price + sub_trade_tp_distance if trade_type == mt5.ORDER_TYPE_BUY else entry_price - sub_trade_tp_distance
                                tp_prices_for_trade.append(round(tp_price, symbol_info.digits))

                        # Apertura di più sotto-trade per simulare la chiusura parziale
                        lot_tp1_portion = round(lot_size_for_backtest_subtrade * CONFIG['tp1_partial_close_ratio'], 2)
                        lot_remainder_portion = round(lot_size_for_backtest_subtrade - lot_tp1_portion, 2)

                        # Assicurati che i volumi siano validi
                        lot_tp1_portion = max(lot_tp1_portion, symbol_info.volume_min)
                        lot_remainder_portion = max(lot_remainder_portion, symbol_info.volume_min)


                        # Primo sotto-trade (TP1)
                        if lot_tp1_portion > 0:
                            ticket_tp1 = f"BT_{current_bar.name.timestamp()}_{len(trade_log) + 1}_TP1"
                            open_positions[ticket_tp1] = {
                                'ticket': ticket_tp1,
                                'type': trade_type,
                                'entry_price': entry_price,
                                'volume': lot_tp1_portion,
                                'sl': sl_price,
                                'tp': tp_prices_for_trade[0], # Usa TP1
                                'strategy': signal['strategy'],
                                'symbol': symbol
                            }
                            trade_log.append({
                                "time": current_bar.name.isoformat(), "action": "OPEN_SUB_TRADE_TP1",
                                "symbol": symbol, "volume": lot_tp1_portion, "entry_price": entry_price,
                                "sl": sl_price, "tp": tp_prices_for_trade[0], "profit_loss": 0,
                                "ticket": ticket_tp1, "strategy": signal['strategy']
                            })
                            current_capital -= lot_tp1_portion * CONFIG['trade_cost_per_lot_point'] * point
                            equity_curve.append(current_capital)
                            simulated_open_trades_for_sym += 1

                        # Secondo sotto-trade (Remainder, con TP più lontano o per trailing)
                        if lot_remainder_portion > 0 and simulated_open_trades_for_sym < CONFIG['max_open_trades_per_symbol']:
                            ticket_remainder = f"BT_{current_bar.name.timestamp()}_{len(trade_log) + 1}_REM"
                            open_positions[ticket_remainder] = {
                                'ticket': ticket_remainder,
                                'type': trade_type,
                                'entry_price': entry_price,
                                'volume': lot_remainder_portion,
                                'sl': sl_price, # Inizialmente lo stesso SL
                                'tp': tp_prices_for_trade[-1], # Usa l'ultimo TP disponibile
                                'strategy': signal['strategy'],
                                'symbol': symbol
                            }
                            trade_log.append({
                                "time": current_bar.name.isoformat(), "action": "OPEN_SUB_TRADE_REM",
                                "symbol": symbol, "volume": lot_remainder_portion, "entry_price": entry_price,
                                "sl": sl_price, "tp": tp_prices_for_trade[-1], "profit_loss": 0,
                                "ticket": ticket_remainder, "strategy": signal['strategy']
                            })
                            current_capital -= lot_remainder_portion * CONFIG['trade_cost_per_lot_point'] * point
                            equity_curve.append(current_capital)
                            simulated_open_trades_for_sym += 1
                            
            # Fine ciclo for per le barre
            # Chiudi posizioni residue alla fine del backtest
            for ticket, pos in open_positions.items():
                close_price = df.iloc[-1]['close']
                profit_loss = ((close_price - pos['entry_price']) * pos['volume'] if pos['type'] == mt5.ORDER_TYPE_BUY else
                               (pos['entry_price'] - close_price) * pos['volume'])
                current_capital += profit_loss
                equity_curve.append(current_capital)
                trade_log.append({
                    "time": df.iloc[-1].name.isoformat(), "action": "CLOSE_EOD", "symbol": symbol,
                    "volume": pos['volume'], "entry_price": pos['entry_price'], "close_price": close_price,
                    "profit_loss": profit_loss, "ticket": ticket, "strategy": pos['strategy']
                })

            total_return = (current_capital - initial_capital) / initial_capital * 100

            equity_df = pd.Series(equity_curve)
            peak = equity_df.expanding(min_periods=1).max()
            drawdown = (equity_df - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

            winning_trades_count = len([t for t in trade_log if t['action'].startswith('CLOSE') and t['profit_loss'] > 0])
            losing_trades_count = len([t for t in trade_log if t['action'].startswith('CLOSE') and t['profit_loss'] < 0])
            total_closed_trades = winning_trades_count + losing_trades_count

            total_wins = sum(t['profit_loss'] for t in trade_log if t['action'].startswith('CLOSE') and t['profit_loss'] > 0)
            total_losses = sum(t['profit_loss'] for t in trade_log if t['action'].startswith('CLOSE') and t['profit_loss'] < 0)
            profit_factor = abs(total_wins / total_losses) if total_losses != 0 else np.inf
            
            winning_percentage = (winning_trades_count / total_closed_trades * 100) if total_closed_trades > 0 else 0


            results_key = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{'_'.join(sorted(strategies_to_test))}" # Ordina per ID consistente
            self.results[results_key] = {
                "initial_capital": initial_capital,
                "final_capital": current_capital,
                "total_return_percent": round(total_return, 2),
                "max_drawdown_percent": round(max_drawdown, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != np.inf else "Infinity",
                "total_trades": len([t for t in trade_log if t['action'].startswith("OPEN_SUB_TRADE")]),
                "winning_trades": winning_trades_count,
                "losing_trades": losing_trades_count,
                "winning_percentage": round(winning_percentage, 2), # Aggiunto percentuale di vincita
                "equity_curve": equity_curve,
                "trade_log": trade_log
            }
            logger.info(f"[Backtest] Backtest completato per {symbol} su {timeframe}. Risultati: {self.results[results_key]}")
            return self.results[results_key]

    def get_backtest_results(self, key):
        """Restituisce i risultati di un backtest specifico."""
        with self.lock:
            return self.results.get(key)


# --- THREADS IN BACKGROUND ---

class MarketMonitor(threading.Thread):
    """
    Monitora il mercato per nuovi tick e candele, aggiornando il DigitalTwin.
    Gestisce anche la riconnessione a MT5 e l'aggiornamento dello stato del conto.
    """
    def __init__(self):
        threading.Thread.__init__(self, name="MarketMonitor")
        self.bot = None
        self.running = False
        # Modifica: last_candle_check ora è per simbolo e per timeframe
        self.last_candle_check = {sym: {tf: 0 for tf in CONFIG['timeframes']} for sym in CONFIG['symbols']}
        self.last_account_update = 0
        self.mt5_restarted_today = False

    def set_bot_instance(self, bot_instance):
        """Imposta l'istanza del bot principale."""
        self.bot = bot_instance

    def run(self):
        """Ciclo principale del monitor di mercato."""
        self.running = True
        logger.info("[MarketMonitor] Avviato.")
        while self.running:
            if self.bot is None:
                logger.warning("[MarketMonitor] Istanza del bot non assegnata. Attesa...")
                time.sleep(1)
                continue

            if mt5.terminal_info() is None:
                last_error = mt5.last_error()
                logger.warning(f"MT5 disconnesso, impossibile caricare dati storici. Ultimo errore: {last_error}")
                if self.bot.initialize_mt5(self.bot.mt5_login, self.bot.mt5_password, self.bot.mt5_server, reconnect=True):
                    logger.info("MT5 riconnesso con successo.")
                    self.mt5_restarted_today = False
                else:
                    logger.error(f"Riconnessione MT5 fallita: {mt5.last_error()}. Riprovo più tardi.")
                    time.sleep(CONFIG['reconnect_mt5_interval_seconds'])
                    if os.name == 'nt' and CONFIG['mt5_path'] and not self.mt5_restarted_today:
                        try:
                            import subprocess
                            subprocess.Popen([CONFIG['mt5_path']])
                            self.mt5_restarted_today = True
                            logger.info("Comando di riavvio MT5 inviato. Attendo riconnessione.")
                            time.sleep(CONFIG['reconnect_mt5_interval_seconds'] * 2)
                        except Exception as e:
                            logger.error(f"Errore durante il tentativo di auto-restart MT5: {e}")
                    continue

            current_time_ts = time.time()
            if current_time_ts - self.last_account_update > CONFIG['position_manager_interval_seconds'] / 2:
                self.bot.update_account_state()
                self.last_account_update = current_time_ts

            # Ciclo per ogni simbolo per aggiornare tick e sentiment
            for sym in CONFIG['symbols']:
                tick = mt5.symbol_info_tick(sym)
                if tick: self.bot.digital_twin.update_with_tick(sym, tick._asdict())
                self.bot.market_sentiment.update_sentiment(sym) # L'aggiornamento del sentiment è disabilitato se disabilitato

            # Modifica: Ciclo per ogni simbolo e poi per ogni timeframe per l'aggiornamento delle candele
            now_utc = datetime.now(pytz.utc)
            now_italy = now_utc.astimezone(italy_tz)

            for sym_to_update in CONFIG['symbols']:
                for tf in CONFIG['timeframes']:
                    mt5_tf_enum = MT5_TIMEFRAME_MAP.get(tf)
                    if not mt5_tf_enum: continue

                    current_sym_rates = mt5.copy_rates_from_pos(sym_to_update, mt5_tf_enum, 0, 1)
                    if current_sym_rates is not None and len(current_sym_rates) > 0:
                        last_mt5_candle_close_time_utc = pd.to_datetime(current_sym_rates[0]['time'], unit='s').tz_localize('UTC')
                        
                        # Aggiorna solo se la candela è più recente dell'ultima registrata per questo simbolo e timeframe
                        if last_mt5_candle_close_time_utc.timestamp() > self.last_candle_check[sym_to_update][tf]:
                            self.bot.digital_twin.update_with_candle(sym_to_update, tf, current_sym_rates[0])
                            logger.info(f"Nuova candela {sym_to_update} {tf} chiusa a {last_mt5_candle_close_time_utc.astimezone(italy_tz).strftime('%Y-%m-%d %H:%M')}")
                            self.last_candle_check[sym_to_update][tf] = last_mt5_candle_close_time_utc.timestamp()
                        # else:
                        #     logger.debug(f"Candela {sym_to_update} {tf} non ancora chiusa o già processata.")
                    else:
                        logger.warning(f"Candela corrente non disponibile per {sym_to_update} su {tf}. Salto aggiornamento.")

            time.sleep(1)

    def stop(self): self.running = False

class OpportunityScout(threading.Thread):
    """
    Cerca opportunità di trading basate sui segnali delle strategie e sull'analisi AI.
    Applica anche i circuit breaker di money management.
    """
    def __init__(self):
        threading.Thread.__init__(self, name="OpportunityScout")
        self.bot = None
        self.running = False
        self.last_signal_time = {} # NUOVO: per implementare "wait for retest"

    def set_bot_instance(self, bot_instance):
        """Imposta l'istanza del bot principale."""
        self.bot = bot_instance

    def run(self):
        """Ciclo principale dello scout di opportunità."""
        self.running = True
        logger.info("[OpportunityScout] Avviato.")
        while self.running:
            if self.bot is None:
                logger.warning("[OpportunityScout] Istanza del bot non assegnata. Attesa...")
                time.sleep(1)
                continue

            time.sleep(CONFIG['scout_interval_seconds'])
            logger.info("[OpportunityScout] Inizio scansione opportunità...")

            # ... (Gestione Stop Operativo Totale - Circuit Breaker)

            for sym in CONFIG['symbols']:
                active_trades_for_sym = [pos for pos in self.bot.open_positions if pos['symbol'] == sym]
                main_timeframe = "M5" # Timeframe primario per lo scouting
                chart_data = self.bot.digital_twin.get_chart_data(sym, main_timeframe)
                live_tick = self.bot.digital_twin.get_live_tick(sym)

                # Popola live_price_info per il contesto AI
                live_price_info = "Prezzo Live: N/D"
                if live_tick:
                    bid_price = live_tick.get('bid')
                    ask_price = live_tick.get('ask')
                    tick = mt5.symbol_info_tick(sym)
                    symbol_info = mt5.symbol_info(sym)
                    if tick and symbol_info and isinstance(tick.ask, (int, float)) and isinstance(tick.bid, (int, float)):
                        spread_value = (tick.ask - tick.bid) / symbol_info.point
                    else:
                        spread_value = 'N/D'
                    if isinstance(bid_price, (int, float)) and isinstance(ask_price, (int, float)):
                        live_price_info = f"Prezzo Live (Bid/Ask): {bid_price:.5f}/{ask_price:.5f}, Spread: {spread_value}"
                    else:
                        live_price_info = f"Prezzo Live (Bid/Ask): N/A/N/A, Spread: {spread_value}"
                
                # Livelli di Supporto/Resistenza per il contesto AI
                support_resistance_levels = self.bot.digital_twin._calculate_support_resistance(
                    self.bot.digital_twin.get_chart_data(sym, "H1"), sym
                )
                sr_info = f"Livelli Supporto/Resistenza (H1): {support_resistance_levels}" if support_resistance_levels else "Nessun livello S/R rilevato."

                # Dati account per l'AI
                account_info_text = "Stato Conto: Non disponibile"
                daily_drawdown_info = "Drawdown Giornaliero: N/A"
                daily_profit_info = "Profitto Giornaliero: N/A"
                weekly_drawdown_info = "Drawdown Settimanale: N/A"
                weekly_profit_info = "Profitto Settimanale: N/A"

                if self.bot.account_info:
                    account_info_text = f"Stato Conto: Equity={self.bot.account_info.equity:.2f}, Posizioni Aperte={len(self.bot.open_positions)}"
                    initial_equity_val = self.bot.initial_equity if isinstance(self.bot.initial_equity, (int, float)) else 0
                    current_equity_val = self.bot.account_info.equity if isinstance(self.bot.account_info.equity, (int, float)) else 0
                    
                    if initial_equity_val > 0:
                        current_daily_drawdown_percentage = ((initial_equity_val - current_equity_val) / initial_equity_val) * 100
                        current_daily_profit_percentage = ((current_equity_val - initial_equity_val) / initial_equity_val) * 100
                    else:
                        current_daily_drawdown_percentage = 0
                        current_daily_profit_percentage = 0

                    daily_drawdown_info = f"Drawdown Giornaliero ({current_daily_drawdown_percentage:.2f}% >= {CONFIG['max_daily_drawdown_percentage']}%): {current_daily_drawdown_percentage >= CONFIG['max_daily_drawdown_percentage']}"
                    daily_profit_info = f"Profitto Giornaliero ({current_daily_profit_percentage:.2f}% >= {CONFIG['daily_profit_target_percentage']}%): {current_daily_profit_percentage >= CONFIG['daily_profit_target_percentage']}"

                    weekly_drawdown, weekly_profit = self.bot.calculate_weekly_performance()
                    weekly_drawdown_info = f"Drawdown Settimanale ({weekly_drawdown:.2f}% >= {CONFIG['max_weekly_drawdown_percentage']}%): {weekly_drawdown >= CONFIG['max_weekly_drawdown_percentage']}"
                    weekly_profit_info = f"Profitto Settimanale ({weekly_profit:.2f}% >= {CONFIG['weekly_profit_target_percentage']}%): {weekly_profit >= CONFIG['weekly_profit_target_percentage']}"


                # --- Decisione AI su Posizioni Esistenti o Blocco Operativo ---
                # Se ci sono già posizioni aperte per el simbolo, chiedi all'AI se mantenerle o invertire
                if active_trades_for_sym:
                    # Costruisci un riassunto delle posizioni aperte per el simbolo
                    open_positions_summary = "\n".join([
                        f"- Ticket: {p['ticket']}, Tipo: {'BUY' if p['type'] == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {p['volume']:.2f}, Entry: {p['price_open']:.5f}, SL: {p['sl']:.5f}, TP: {p['tp']:.5f}, Profitto Corrente: {p['profit']:.2f}"
                        for p in active_trades_for_sym
                    ])
                    
                    # Aggiunta gestione sicura per MFI, OBV, VPT
                    mfi_val = chart_data['MFI'].iloc[-1] if 'MFI' in chart_data.columns else np.nan
                    obv_val = chart_data['OBV'].iloc[-1] if 'OBV' in chart_data.columns else np.nan
                    vpt_val = chart_data['VPT'].iloc[-1] if 'VPT' in chart_data.columns else np.nan

                    context = (
                        f"Simbolo: {sym}, Ora Attuale (Italia): {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{live_price_info}\n"
                        f"{account_info_text}\n"
                        f"Posizioni Aperte per {sym}:\n{open_positions_summary}\n"
                        f"Dati Ultima Candela ({main_timeframe}):\n{chart_data.iloc[-1].to_string() if not chart_data.empty else 'N/A'}\n"
                        f"{sr_info}\n"
                        f"MFI corrente ({main_timeframe}): {mfi_val:.2f} (se disponibile)\n"
                        f"OBV corrente ({main_timeframe}): {obv_val:.2f} (se disponibile)\n"
                        f"VPT corrente ({main_timeframe}): {vpt_val:.2f} (se disponibile)\n"
                        f"Limiti Operativi Attivi: \n"
                        f"  {daily_drawdown_info}.\n"
                        f"  {daily_profit_info}.\n"
                        f"  {weekly_drawdown_info}.\n"
                        f"  {weekly_profit_info}.\n"
                        f"SL consecutivi: {self.bot.consecutive_sl_count}\n" # Passa el conteggio SL consecutivi
                        f"TP consecutivi: {self.bot.consecutive_tp_count}\n" # Passa el conteggio TP consecutivi
                        f"Decisione richiesta: Dovremmo mantenere queste posizioni (HOLD)? (NON RACCOMANDARE CHIUSURE PARZIALI O INVERSIONI QUI)"
                    )
                    ai_verdict = self.bot.ai_brain.get_strategic_analysis(context)
                    
                    # NUOVO: Cooldown per le decisioni AI di REVERSE/CLOSE_PARTIAL su trade appena aperti
                    # Dato che abbiamo rimosso queste azioni, il cooldown non è più strettamente necessario qui,
                    # ma lo manteniamo per logica di "non interferenza" se l'AI dovesse suggerire altro.
                    for pos_data in active_trades_for_sym:
                        if (time.time() - pos_data.get('open_timestamp', 0) < CONFIG['ai_decision_cooldown_seconds'] and
                            (ai_verdict['action'] == "REVERSE" or ai_verdict['action'] == "CLOSE_PARTIAL")): # Manteniamo el check per robustezza
                            logger.info(f"AI ha raccomandato {ai_verdict['action']} per {sym} (Ticket: {pos_data['ticket']}), ma è in cooldown. Forzato a HOLD.")
                            ai_verdict['action'] = "HOLD"
                            ai_verdict['analysis_text'] = "AI ha deciso di HOLD a causa del cooldown per trade appena aperto."
                            break # Basta forzare HOLD se una delle posizioni è in cooldown

                    # L'AI può raccomandare solo HOLD per posizioni esistenti (azioni REVERSE/CLOSE_PARTIAL rimosse)
                    if ai_verdict['action'] == "HOLD":
                        logger.info(f"AI raccomanda HOLD per posizione su {sym}. {ai_verdict.get('analysis_text', '')}")
                        self.bot.handle_notification(ai_verdict)
                    else: # Se l'AI raccomanda altro (es. BUY/SELL/WAIT), per sicurezza, la trattiamo come HOLD in questo contesto
                        logger.debug(f"AI per posizione su {sym} ha raccomandato '{ai_verdict['action']}' (forzato a HOLD in questo contesto). {ai_verdict.get('analysis_text', '')}")
                        ai_verdict['action'] = "HOLD"
                        self.bot.handle_notification(ai_verdict)
                    continue # Non cercare nuove opportunità se ci sono già posizioni attive e l'AI le sta gestendo.

                # Se non ci sono posizioni aperte per il simbolo, cerca nuove opportunità
                if chart_data.empty or len(chart_data) < 200:
                    # Aggiunta gestione sicura per MFI, OBV, VPT
                    mfi_val = chart_data['MFI'].iloc[-1] if 'MFI' in chart_data.columns else np.nan
                    obv_val = chart_data['OBV'].iloc[-1] if 'OBV' in chart_data.columns else np.nan
                    vpt_val = chart_data['VPT'].iloc[-1] if 'VPT' in chart_data.columns else np.nan

                    context = (f"Simbolo: {sym}, Ora Attuale (Italia): {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"{live_price_info}\n"
                               f"{account_info_text}\n"
                               f"Dati Ultima Candela ({main_timeframe}):\n{chart_data.iloc[-1].to_string() if not chart_data.empty else 'N/A'}\n"
                               f"Motivo per non operare: Dati chart insufficienti per analisi completa. L'AI sta aspettando un setup più pulito e condizioni di mercato ottimali.\n"
                               f"{sr_info}\n"
                               f"MFI corrente ({main_timeframe}): {mfi_val:.2f} (se disponibile)\n"
                               f"OBV corrente ({main_timeframe}): {obv_val:.2f} (se disponibile)\n"
                               f"VPT corrente ({main_timeframe}): {vpt_val:.2f} (se disponibile)\n"
                               f"Limiti Operativi Attivi: \n"
                               f"  {daily_drawdown_info}.\n"
                               f"  {daily_profit_info}.\n"
                               f"  {weekly_drawdown_info}.\n"
                               f"  {weekly_profit_info}.\n"
                               f"SL consecutivi: {self.bot.consecutive_sl_count}\n" # Passa el conteggio SL consecutivi
                               f"TP consecutivi: {self.bot.consecutive_tp_count}\n" # Passa el conteggio TP consecutivi
                               )
                    ai_verdict = self.bot.ai_brain.get_strategic_analysis(context)
                    ai_verdict['action'] = "WAIT"
                    ai_verdict['confidence_score'] = 0
                    self.bot.handle_notification(ai_verdict)
                    continue

                # --- NUOVI FILTRI PRE-TRADE PER SIMBOLO (DAL DOCUMENTO) ---
                now = datetime.now(italy_tz)
                symbol_rules = CONFIG['symbol_specific_rules'].get(sym, {})

                # 1. Filtro Sessione di Trading
                if not is_in_trading_session(sym, now):
                    logger.info(f"Trading bloccato per {sym}: Fuori sessione ottimale. Ora: {now.strftime('%H:%M')}.")
                    self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Fuori sessione ottimale. L'AI raccomanda ATTESA."})
                    continue

                # 2. Filtro News (simulato)
                if is_news_impact_active(sym, now):
                    logger.info(f"Trading bloccato per {sym}: Possibile impatto news. Ora: {now.strftime('%H:%M')}.")
                    self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Possibile impatto news. L'AI raccomanda ATTESA."})
                    continue

                # 3. Filtro Spread
                if not is_spread_acceptable(sym):
                    # Recupera il tick più recente per il messaggio di log
                    current_tick_for_log = mt5.symbol_info_tick(sym)
                    spread_for_log = (current_tick_for_log.ask - current_tick_for_log.bid) / mt5.symbol_info(sym).point if current_tick_for_log and mt5.symbol_info(sym) else 'N/D'
                    logger.info(f"Trading bloccato per {sym}: Spread non accettabile. Spread corrente: {spread_for_log}.")
                    self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Spread troppo alto. L'AI raccomanda ATTESA."})
                    continue

                # 4. Filtro Volatilità
                if not is_volatility_acceptable(sym, chart_data):
                    logger.info(f"Trading bloccato per {sym}: Volatilità non accettabile.")
                    self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Volatilità non ottimale. L'AI raccomanda ATTESA."})
                    continue

                # 5. Filtro Rischio (Max operazioni al giorno, Stop dopo SL consecutivi)
                # Max operazioni al giorno
                max_daily_trades_limit = symbol_rules.get("max_daily_trades")
                if max_daily_trades_limit is not None:
                    today_trades = [t for t in self.bot.trade_journal.get_trades() if datetime.fromisoformat(t['timestamp']).date() == now.date() and t['symbol'] == sym and t['action'].startswith("Open")]
                    if len(today_trades) >= max_daily_trades_limit:
                        logger.info(f"Trading bloccato per {sym}: Raggiunto limite max {max_daily_trades_limit} operazioni giornaliere.")
                        self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Raggiunto limite operazioni giornaliere. L'AI raccomanda ATTESA."})
                        continue
                
                # Stop dopo SL consecutivi
                stop_after_consecutive_sl_limit = symbol_rules.get("stop_after_consecutive_sl")
                if stop_after_consecutive_sl_limit is not None:
                    if self.bot.consecutive_sl_count >= stop_after_consecutive_sl_limit:
                        logger.info(f"Trading bloccato per {sym}: Raggiunto limite {stop_after_consecutive_sl_limit} SL consecutivi.")
                        self.bot.handle_notification({"type": "INFO", "message": f"Trading bloccato per {sym}: Raggiunto limite SL consecutivi. L'AI raccomanda ATTESA."})
                        continue
                # --- FINE NUOVI FILTRI PRE-TRADE ---


                signals = self.bot.strategy_engine.generate_signals(sym, main_timeframe)

                if not signals:
                    # Se non ci sono segnali dalle strategie, ma ci sono pattern o indicatori forti,
                    # costruiamo un technical_confidence_score di base per l'AI.
                    base_tech_confidence = 0
                    patterns_detected = self.bot.pattern_engine.detect_all_patterns(chart_data)
                    if patterns_detected:
                        base_tech_confidence += 10 # Base per qualsiasi pattern rilevato
                        for p in patterns_detected:
                            base_tech_confidence += p.get('confidence_boost', 0) / 2 # Aggiungi metà del boost del pattern

                    # Aggiungi confidenza se indicatori chiave sono in zone estreme
                    # Aggiunto controllo per la presenza delle colonne prima di accedere
                    if 'RSI' in chart_data.columns and (chart_data['RSI'].iloc[-1] < 30 or chart_data['RSI'].iloc[-1] > 70):
                        base_tech_confidence += 5
                    if 'MFI' in chart_data.columns and (chart_data['MFI'].iloc[-1] < 20 or chart_data['MFI'].iloc[-1] > 80):
                        base_tech_confidence += 5
                    # Puoi aggiungere altri controlli per MACD, Bollinger, ecc.

                    # Limita la confidenza base per evitare falsi positivi eccessivi
                    base_tech_confidence = min(base_tech_confidence, 35) # Max 35 se solo pattern/indicatore senza strategia

                    # Aggiunta gestione sicura per MFI, OBV, VPT
                    mfi_val = chart_data['MFI'].iloc[-1] if 'MFI' in chart_data.columns else np.nan
                    obv_val = chart_data['OBV'].iloc[-1] if 'OBV' in chart_data.columns else np.nan
                    vpt_val = chart_data['VPT'].iloc[-1] if 'VPT' in chart_data.columns else np.nan

                    context = (f"Simbolo: {sym}, Ora Attuale (Italia): {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"{live_price_info}\n"
                               f"{account_info_text}\n"
                               f"Dati Ultima Candela ({main_timeframe}):\n{chart_data.iloc[-1].to_string()}\n"
                               f"Pattern Candlestick Rilevati: {', '.join([p['type'] for p in patterns_detected]) if patterns_detected else 'Nessuno'}.\n"
                               f"Segnali Strategici Attivi: Nessuno.\n" # Esplicita che nessuna strategia ha dato segnale
                               f"Technical Confidence Score (pre-AI): {base_tech_confidence}.\n" # Passa la confidenza tecnica all'AI
                               f"MFI corrente ({main_timeframe}): {mfi_val:.2f} (se disponibile)\n"
                               f"OBV corrente ({main_timeframe}): {obv_val:.2f} (se disponibile)\n"
                               f"VPT corrente ({main_timeframe}): {vpt_val:.2f} (se disponibile)\n"
                               f"{sr_info}\n" # Include S/R levels
                               f"Limiti Operativi Attivi: \n"
                               f"  {daily_drawdown_info}.\n"
                               f"  {daily_profit_info}.\n"
                               f"  {weekly_drawdown_info}.\n"
                               f"  {weekly_profit_info}.\n"
                               f"SL consecutivi: {self.bot.consecutive_sl_count}\n" # Passa el conteggio SL consecutivi
                               f"TP consecutivi: {self.bot.consecutive_tp_count}\n" # Passa el conteggio TP consecutivi
                               )
                    ai_verdict = self.bot.ai_brain.get_strategic_analysis(context)
                    ai_verdict['action'] = "WAIT"
                    ai_verdict['confidence_score'] = 0
                    self.bot.handle_notification(ai_verdict)
                    continue

                for signal_to_execute in signals:
                    # NUOVO: Implementazione del "wait for retest"
                    # Controlla se il segnale è stato generato sulla barra corrente
                    current_candle_time = chart_data.index[-1].timestamp()
                    if self.last_signal_time.get(sym) == current_candle_time:
                        logger.info(f"Segnale per {sym} già processato su questa candela. Attendo la prossima.")
                        continue # Salta se il segnale è già stato generato sulla stessa candela

                    # Calcola la confidenza tecnica preliminare
                    confidence_pre_ai = signal_to_execute['confidence_boost']
                    
                    # Confluenza Multi-Timeframe (Trend)
                    df_h1 = self.bot.digital_twin.get_chart_data(sym, "H1")
                    df_h4 = self.bot.digital_twin.get_chart_data(sym, "H4")
                    df_d1 = self.bot.digital_twin.get_chart_data(sym, "D1") # NUOVO: D1 per trend

                    trend_mismatch_penalty = 0
                    h1_trend_aligned_for_signal = False
                    h4_trend_aligned_for_signal = False
                    d1_trend_aligned_for_signal = False # NUOVO: D1 trend

                    # Helper per determinare il trend basato su SMA 10/50
                    def get_sma_trend(df_tf_local):
                        if not df_tf_local.empty and 'SMA_10' in df_tf_local.columns and 'SMA_50' in df_tf_local.columns:
                            if df_tf_local['SMA_10'].iloc[-1] > df_tf_local['SMA_50'].iloc[-1]: return "BULLISH"
                            elif df_tf_local['SMA_10'].iloc[-1] < df_tf_local['SMA_50'].iloc[-1]: return "BEARISH"
                        return "NEUTRAL"

                    h1_trend = get_sma_trend(df_h1)
                    h4_trend = get_sma_trend(df_h4)
                    d1_trend = get_sma_trend(df_d1) # NUOVO: D1 trend

                    if ((signal_to_execute['type'] == 'BUY' and h1_trend == "BULLISH") or
                        (signal_to_execute['type'] == 'SELL' and h1_trend == "BEARISH")):
                        h1_trend_aligned_for_signal = True
                    else:
                        trend_mismatch_penalty += 10
                    
                    if ((signal_to_execute['type'] == 'BUY' and h4_trend == "BULLISH") or
                        (signal_to_execute['type'] == 'SELL' and h4_trend == "BEARISH")):
                        h4_trend_aligned_for_signal = True
                    else:
                        trend_mismatch_penalty += 15

                    if ((signal_to_execute['type'] == 'BUY' and d1_trend == "BULLISH") or # NUOVO: D1 trend check
                        (signal_to_execute['type'] == 'SELL' and d1_trend == "BEARISH")):
                        d1_trend_aligned_for_signal = True
                    else:
                        trend_mismatch_penalty += 20 # Maggiore penalità per disallineamento D1

                    technical_confidence_score = max(0, confidence_pre_ai - trend_mismatch_penalty)
                    
                    signal_to_execute['confluence_details'] = {
                        "h1_trend_aligned": h1_trend_aligned_for_signal,
                        "h4_trend_aligned": h4_trend_aligned_for_signal,
                        "d1_trend_aligned": d1_trend_aligned_for_signal, # NUOVO: D1 trend detail
                    }
                    
                    # NUOVO: Applica l'efficacia storica della strategia al boost
                    effectiveness_multiplier = self.bot.strategy_engine.get_strategy_effectiveness(signal_to_execute['strategy'])
                    technical_confidence_score = int(technical_confidence_score * effectiveness_multiplier)
                    logger.debug(f"Technical confidence for {signal_to_execute['strategy']} adjusted by effectiveness multiplier {effectiveness_multiplier:.2f} to {technical_confidence_score}.")

                    # Filtro Qualità Segnale (dal documento)
                    min_signal_confidence = symbol_rules.get("min_signal_confidence", 0)
                    min_strategies_concordant = symbol_rules.get("min_strategies_concordant", 0)

                    if technical_confidence_score < min_signal_confidence:
                        logger.info(f"Segnale per {sym} ignorato: Confidenza tecnica ({technical_confidence_score}) sotto la soglia minima ({min_signal_confidence}).")
                        continue # Salta il segnale se la confidenza è troppo bassa

                    # Controlla la concordanza delle strategie (se richiesto)
                    # MODIFICA: COMMENTA IL BLOCCO CHE IMPEDISCE L'ANALISI
                    # if min_strategies_concordant > 0:
                    #     concordant_strategies_count = 0
                    #     # Questa è una semplificazione, in un sistema reale si dovrebbe tracciare quali strategie concordano.
                    #     # Per ora, se ci sono più segnali dello stesso tipo, li contiamo come concordanti.
                    #     for s in signals:
                    #         if s['type'] == signal_to_execute['type'] and s['strategy'] != signal_to_execute['strategy']:
                    #             concordant_strategies_count += 1
                        
                    #     if concordant_strategies_count + 1 < min_strategies_concordant: # +1 per la strategia corrente
                    #         logger.info(f"Segnale per {sym} ignorato: Solo {concordant_strategies_count + 1} strategie concordano, minimo {min_strategies_concordant} richiesto.")
                    #         continue # Salta il segnale se non ci sono abbastanza strategie concordanti

                    adjusted_lot_multiplier = 1.0
                    # La logica di money management qui è basata su CONFIG, non più su consecutive_sl/tp_count
                    # (Commentato come da richiesta, la gestione del rischio è ora più dinamica e basata sull'AI)

                    # Aggiunta gestione sicura per MFI, OBV, VPT
                    mfi_val = chart_data['MFI'].iloc[-1] if 'MFI' in chart_data.columns else np.nan
                    obv_val = chart_data['OBV'].iloc[-1] if 'OBV' in chart_data.columns else np.nan
                    vpt_val = chart_data['VPT'].iloc[-1] if 'VPT' in chart_data.columns else np.nan

                    # Prepara il contesto completo per l'AI
                    context = (f"Simbolo: {sym}, Ora Attuale (Italia): {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"{live_price_info}\n"
                               f"{account_info_text}\n"
                               f"Segnale Rilevato: Strategia='{signal_to_execute['strategy']}', Tipo='{signal_to_execute['type']}', Forza='{signal_to_execute['strength']}'.\n"
                               f"Motivo del Segnale: {signal_to_execute['reason']}\n"
                               f"Dati Ultima Candela ({main_timeframe}):\n{chart_data.iloc[-1].to_string()}\n"
                               f"Pattern Candlestick Rilevati: {', '.join([p['type'] for p in self.bot.pattern_engine.detect_all_patterns(chart_data)]) if self.bot.pattern_engine.detect_all_patterns(chart_data) else 'Nessuno'}.\n"
                               f"Conferme Trend Multi-Timeframe: H1 allineato: {signal_to_execute['confluence_details']['h1_trend_aligned']}, H4 allineato: {signal_to_execute['confluence_details']['h4_trend_aligned']}, D1 allineato: {signal_to_execute['confluence_details']['d1_trend_aligned']}.\n"
                               f"Technical Confidence Score (pre-AI): {technical_confidence_score}.\n" # Passa la confidenza tecnica all'AI
                               f"MFI corrente ({main_timeframe}): {mfi_val:.2f} (se disponibile)\n"
                               f"OBV corrente ({main_timeframe}): {obv_val:.2f} (se disponibile)\n"
                               f"VPT corrente ({main_timeframe}): {vpt_val:.2f} (se disponibile)\n"
                               f"{sr_info}\n" # Include S/R levels
                               f"Lotto effettivo base (calibrato per risk mgmt): {adjusted_lot_multiplier}.\n"
                               f"Limiti Operativi Attivi: \n"
                               f"  {daily_drawdown_info}.\n"
                               f"  {daily_profit_info}.\n"
                               f"  {weekly_drawdown_info}.\n"
                               f"  {weekly_profit_info}.\n"
                               f"SL consecutivi: {self.bot.consecutive_sl_count}\n" # Passa el conteggio SL consecutivi
                               f"TP consecutivi: {self.bot.consecutive_tp_count}\n" # Passa el conteggio TP consecutivi
                               )

                    ai_verdict = self.bot.ai_brain.get_strategic_analysis(context)
                    
                    # Se l'AI non suggerisce un'azione specifica, usa quella del segnale
                    if ai_verdict['action'] not in ["BUY", "SELL", "WAIT", "HOLD", "UPDATE_CONFIG"]: # Aggiunto UPDATE_CONFIG
                               ai_verdict['action'] = signal_to_execute['type']
                    
                    # NUOVO: Aggiorna il timestamp dell'ultimo segnale processato per questo simbolo
                    self.last_signal_time[sym] = current_candle_time

                    self.bot.handle_notification(ai_verdict)
    def stop(self): self.running = False

class PositionManager(threading.Thread):
    """
    Gestisce attivamente le posizioni aperte, applicando trailing stop,
    break-even e monitorando la durata dei trade.
    """
    def __init__(self):
        threading.Thread.__init__(self, name="PositionManager")
        self.bot = None
        self.running = False
        self.last_check_time = {}

    def set_bot_instance(self, bot_instance):
        """Imposta l'istanza del bot principale."""
        self.bot = bot_instance

    def run(self):
        """Ciclo principale del gestore posizioni."""
        self.running = True
        logger.info("[PositionManager] Avviato.")
        while self.running:
            if self.bot is None:
                logger.warning("[PositionManager] Istanza del bot non assegnata. Attesa...")
                time.sleep(1)
                continue

            time.sleep(CONFIG['position_manager_interval_seconds'])
            with self.bot.lock:
                all_mt5_positions = mt5.positions_get()
                if all_mt5_positions is None:
                    logger.warning("[PositionManager] Impossibile recuperare posizioni da MT5. Salto ciclo.")
                    continue
                
                # Mappa le posizioni MT5 a dizionari e mantieni i metadati personalizzati
                new_managed_positions = []
                for pos_mt5_obj in all_mt5_positions:
                    pos_dict = pos_mt5_obj._asdict()
                    pos_dict['is_hedged'] = False # Default
                    pos_dict['open_time_m5_bars'] = 0 # Default (sarà aggiornato o recuperato)

                    # Cerca se questa posizione MT5 (per ticket) esiste già nella lista gestita dal bot
                    # per recuperare i metadati esistenti.
                    for existing_pos_dict in self.bot.open_positions:
                        if existing_pos_dict.get('ticket') == pos_mt5_obj.ticket:
                            pos_dict['is_hedged'] = existing_pos_dict.get('is_hedged', False)
                            pos_dict['open_time_m5_bars'] = existing_pos_dict.get('open_time_m5_bars', 0)
                            pos_dict['open_timestamp'] = existing_pos_dict.get('open_timestamp', time.time()) # Mantieni il timestamp di apertura
                            pos_dict['strategy'] = existing_pos_dict.get('strategy', 'N/A') # Mantieni la strategia
                            break
                    new_managed_positions.append(pos_dict)
                
                self.bot.open_positions = new_managed_positions # Sostituisci l'elenco delle posizioni gestite

            if not self.bot.open_positions: continue

            positions_to_process = self.bot.open_positions[:] # Lavora su una copia per sicurezza
            for pos_data in positions_to_process: # Itero sui dizionari delle posizioni
                ticket = pos_data['ticket']
                symbol = pos_data['symbol']

                # VERIFICA FINALE: Assicurati che la posizione sia ancora aperta su MT5 prima di gestirla
                live_position_check = mt5.positions_get(ticket=ticket)
                if not live_position_check or len(live_position_check) == 0:
                    logger.warning(f"Posizione {ticket} ({symbol}) non trovata su MT5 per gestione. Probabilmente già chiusa. Rimuovo da lista interna.")
                    # Rimuovi la posizione dalla lista interna del bot
                    self.bot.open_positions = [p for p in self.bot.open_positions if p['ticket'] != ticket]
                    continue
                
                # Se la posizione è ancora aperta, procedi con la gestione
                # Aggiorna il contatore delle barre M5 (approssimazione)
                pos_data['open_time_m5_bars'] += (CONFIG['position_manager_interval_seconds'] // (60 * 5)) + 1 # Incrementa di 1 ogni 5 minuti circa

                if time.time() - self.last_check_time.get(ticket, 0) < CONFIG['position_manager_interval_seconds'] / 2:
                    continue
                
                self._manage_position(pos_data, pos_data['open_time_m5_bars'])
                self.last_check_time[ticket] = time.time()

    def _manage_position(self, pos_data: dict, open_time_m5_bars: int): # pos_data è un dizionario
        """Gestisce una singola posizione aperta."""
        symbol = pos_data['symbol']
        # Recupero l'oggetto mt5.TradePosition *live* qui per le operazioni MT5 API
        pos_live = mt5.positions_get(ticket=pos_data['ticket'])
        if not pos_live or len(pos_live) == 0:
            logger.warning(f"Posizione live {pos_data['ticket']} non recuperabile in _manage_position. Già chiusa.")
            return
        pos = pos_live[0] # Questo è l'oggetto mt5.TradePosition

        chart_data_atr = self.bot.digital_twin.get_chart_data(symbol, "H1")
        if chart_data_atr.empty:
            logger.warning(f"Dati H1 non disponibili per {symbol}, impossibile gestire posizione {pos.ticket}.")
            return

        atr = chart_data_atr.iloc[-1]['ATR'] if 'ATR' in chart_data_atr.columns else 0.0 # Gestione sicura ATR
        if atr <= 0:
            logger.warning(f"ATR non valido per {symbol}, impossibile gestire posizione {pos.ticket}.")
            return

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: return

        point = symbol_info.point
        current_tick = mt5.symbol_info_tick(symbol)
        if not current_tick: return

        current_price_for_sl_tp = current_tick.bid if pos.type == mt5.ORDER_TYPE_BUY else current_tick.ask

        profit_pips = ((current_tick.bid - pos.price_open) / point if pos.type == mt5.ORDER_TYPE_BUY else
                       (pos.price_open - current_tick.ask) / point)
        
        profit_amount = ((current_tick.bid - pos.price_open) * pos.volume if pos.type == mt5.ORDER_TYPE_BUY else
                         (pos.price_open - current_tick.ask) * pos.volume)

        initial_sl_pips = abs(pos.price_open - pos.sl) / point if pos.sl is not None else 0
        if initial_sl_pips <= 0: # Se lo SL iniziale non è valido, non possiamo calcolare i rapporti
            logger.warning(f"SL iniziale non valido per {pos.ticket}. Disabilitato Break-Even/Trailing per questa posizione.")
            return

        # --- Gestione Durata Massima Trade ---
        if open_time_m5_bars >= CONFIG['max_trade_duration_bars_m5']:
            logger.info(f"Posizione {pos.ticket} ({symbol}) ha superato la durata massima ({CONFIG['max_trade_duration_bars_m5']} barre M5). Chiusura forzata.")
            self.bot.close_position(pos.ticket, symbol, pos.type, pos.volume, "MAX_DURATION_CLOSE", {"analysis_text": f"Chiusura forzata per durata massima trade ({CONFIG['max_trade_duration_bars_m5']} barre M5).", "strategy": pos_data.get('strategy', 'N/A')})
            return # Posizione chiusa, esci dalla gestione

        # --- AI-Driven Active Position Management (solo per HOLD) ---
        # L'AI non dovrebbe raccomandare chiusure totali qui, solo HOLD.
        # L'AI viene interpellata meno frequentemente per le posizioni aperte per evitare sovraccarico
        # Aggiunta gestione sicura per MFI, OBV, VPT
        mfi_val = chart_data_atr['MFI'].iloc[-1] if 'MFI' in chart_data_atr.columns else np.nan
        obv_val = chart_data_atr['OBV'].iloc[-1] if 'OBV' in chart_data_atr.columns else np.nan
        vpt_val = chart_data_atr['VPT'].iloc[-1] if 'VPT' in chart_data_atr.columns else np.nan

        # Cooldown per le decisioni AI (ora solo per non-interferenza)
        ai_cooldown_active = (time.time() - pos_data.get('open_timestamp', 0) < CONFIG['ai_decision_cooldown_seconds'])

        # Costruisci il contesto per l'AI, includendo la strategia che ha aperto il trade
        context = (
            f"Simbolo: {symbol}, Ticket Posizione: {pos.ticket}\n"
            f"Tipo Posizione: {'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}\n"
            f"Volume: {pos.volume:.2f}, Prezzo Apertura: {pos.price_open:.5f}\n"
            f"Prezzo Corrente (Bid/Ask): {current_tick.bid:.5f}/{current_tick.ask:.5f}\n"
            f"Stop Loss Attuale: {pos.sl:.5f}, Take Profit Attuale: {pos.tp:.5f}\n"
            f"Profitto Corrente (Pips): {profit_pips:.2f}, Profitto Corrente (Valuta): {profit_amount:.2f}\n"
            f"Durata Posizione (barre M5): {open_time_m5_bars} (Max configurato: {CONFIG['max_trade_duration_bars_m5']})\n"
            f"ATR Corrente (H1): {atr:.5f}\n"
            f"Condizioni di Mercato Generali: Volatilità attuale: {atr:.5f}, Range medio recente (es. ATR): {chart_data_atr['ATR'].mean():.5f}\n"
            f"Prossimi Livelli S/R (H1): {self.bot.digital_twin._calculate_support_resistance(self.bot.digital_twin.get_chart_data(symbol, 'H1'), symbol)}\n"
            f"MFI corrente (H1): {mfi_val:.2f} (se disponibile)\n"
            f"OBV corrente (H1): {obv_val:.2f} (se disponibile)\n"
            f"VPT corrente (H1): {vpt_val:.2f} (se disponibile)\n"
            f"Strategia che ha aperto il trade: {pos_data.get('strategy', 'Sconosciuta')}\n" # Passa la strategia di apertura
            f"SL consecutivi: {self.bot.consecutive_sl_count}\n"
            f"TP consecutivi: {self.bot.consecutive_tp_count}\n"
            f"Decisione richiesta: Dovremmo mantenere queste posizioni (HOLD)? (NON RACCOMANDARE CHIUSURE PARZIALI O INVERSIONI QUI)"
        )

        if self.bot.ai_brain and (open_time_m5_bars % 20 == 0 or abs(profit_pips) > initial_sl_pips * 0.7): # Ogni 20 barre M5 o se vicino a SL/TP
            ai_verdict = self.bot.ai_brain.get_strategic_analysis(context)
            
            # Applica il cooldown AI (se l'AI suggerisce qualcosa di diverso da HOLD, lo forziamo a HOLD se in cooldown)
            if ai_cooldown_active and ai_verdict['action'] != "HOLD":
                logger.info(f"AI ha raccomandato {ai_verdict['action']} per {symbol} (Ticket: {pos.ticket}), ma è in cooldown. Forzato a HOLD.")
                ai_verdict['action'] = "HOLD"
                ai_verdict['analysis_text'] = "AI ha deciso di HOLD a causa del cooldown per trade appena aperto."

            # Se l'AI raccomanda HOLD, lo logghiamo. Altrimenti, ignoriamo le sue raccomandazioni per posizioni esistenti.
            if ai_verdict['action'] == "HOLD":
                logger.info(f"AI raccomanda HOLD per posizione {pos.ticket}. {ai_verdict.get('analysis_text', '')}")
                # Non facciamo nulla qui, la gestione SL/TP standard continuerà
            else: # Se l'AI raccomanda BUY/SELL/WAIT (che non sono azioni valide qui), lo logghiamo e non agiamo.
                logger.debug(f"AI per posizione {pos.ticket} ha raccomandato '{ai_verdict['action']}' (ignorato, solo HOLD è permesso). {ai_verdict.get('analysis_text', '')}")
                # Non facciamo nulla, la gestione SL/TP standard continuerà


        # Break-Even Stop Loss
        if CONFIG['break_even_activation_ratio'] > 0 and pos.sl != pos.price_open:
            if profit_pips >= initial_sl_pips * CONFIG['break_even_activation_ratio']:
                # Assicurati che il nuovo SL sia effettivamente migliore o uguale al prezzo di apertura
                if ((pos.type == mt5.ORDER_TYPE_BUY and (pos.sl is None or pos.sl < pos.price_open)) or
                   (pos.type == mt5.ORDER_TYPE_SELL and (pos.sl is None or pos.sl > pos.price_open))):
                    self._modify_sl_or_tp(pos.ticket, new_sl=pos.price_open, symbol_info=symbol_info)
                    logger.info(f"SL a Break-Even per {pos.ticket} ({symbol}).")

        # Trailing Stop (migliorato per attivazione anticipata e movimento continuo)
        trailing_distance = atr * CONFIG['atr_sl_multiplier']

        # Attiva il trailing stop se il trade è in profitto (anche minimo)
        if profit_pips > 0 and profit_pips >= initial_sl_pips * CONFIG['trailing_stop_activation_ratio']:
            new_sl_candidate = pos.sl # Inizializza con l'attuale SL
            if pos.type == mt5.ORDER_TYPE_BUY:
                potential_new_sl = current_price_for_sl_tp - trailing_distance
                # Assicurati che il nuovo SL sia migliore del precedente E almeno all'entry price
                if potential_new_sl > pos.sl and potential_new_sl > pos.price_open:
                    new_sl_candidate = potential_new_sl
            else: # SELL
                potential_new_sl = current_price_for_sl_tp + trailing_distance
                # Assicurati che il nuovo SL sia migliore del precedente E almeno all'entry price
                if potential_new_sl < pos.sl and potential_new_sl < pos.price_open:
                    new_sl_candidate = potential_new_sl

            new_sl = round(new_sl_candidate, symbol_info.digits)
            if new_sl != pos.sl: # Evita modifiche inutili
                self._modify_sl_or_tp(pos.ticket, new_sl=new_sl, symbol_info=symbol_info)

    def _modify_sl_or_tp(self, ticket, new_sl=None, new_tp=None, symbol_info=None):
        """Modifica Stop Loss o Take Profit di una posizione esistente."""
        if not symbol_info:
            logger.error(f"Informazioni simbolo non fornite per modifica ordine {ticket}.")
            return

        # Recupera l'oggetto mt5.TradePosition live
        current_pos_list = mt5.positions_get(ticket=ticket)
        if not current_pos_list or len(current_pos_list) == 0:
            logger.warning(f"Posizione {ticket} non trovata su MT5 per la modifica SL/TP. Potrebbe essere già chiusa. Rimuovo da lista interna.")
            # Rimuovi la posizione dalla lista interna del bot se non più presente su MT5
            self.bot.open_positions = [p for p in self.bot.open_positions if p['ticket'] != ticket]
            return
        
        current_pos = current_pos_list[0] # Questo è l'oggetto mt5.TradePosition

        pos_type = current_pos.type
        current_tick = mt5.symbol_info_tick(symbol_info.name)
        if not current_tick:
            logger.warning(f"Impossibile ottenere tick corrente per modifica SL/TP su {symbol_info.name}.")
            return

        # Recupera stops_level dal simbolo, con fallback
        stops_level_points = getattr(symbol_info, 'stops_level', 0)
        if stops_level_points == 0:
            stops_level_points = 10 # Default fallback
            logger.warning(f"stops_level non disponibile o zero per {symbol_info.name}. Usato fallback di {stops_level_points} punti.")
            
        stops_level_price_units = stops_level_points * symbol_info.point

        # Inizializza i valori per la richiesta con i valori attuali della posizione
        request_sl = current_pos.sl
        request_tp = current_pos.tp

        # Logica per il nuovo Stop Loss
        if new_sl is not None:
            # Calcola il limite di stops_level dal prezzo corrente
            if pos_type == mt5.ORDER_TYPE_BUY:
                # Per BUY: SL deve essere al di sotto del prezzo corrente (bid) di almeno stops_level
                min_acceptable_sl = current_tick.bid - stops_level_price_units
                
                # Il nuovo SL candidato deve essere almeno 'min_acceptable_sl'
                candidate_sl = max(new_sl, min_acceptable_sl)
                
                # Se l'attuale SL è migliore del candidato, manteniamo l'attuale SL
                # Un SL migliore per BUY è un valore più alto
                if current_pos.sl is not None and candidate_sl < current_pos.sl: # Se il candidato è peggiore dell'attuale
                    request_sl = current_pos.sl
                else:
                    request_sl = candidate_sl
            else: # SELL
                # Per SELL: SL deve essere al di sopra del prezzo corrente (ask) di almeno stops_level
                max_acceptable_sl = current_tick.ask + stops_level_price_units
                
                # Il nuovo SL candidato deve essere al massimo 'max_acceptable_sl'
                candidate_sl = min(new_sl, max_acceptable_sl)
                
                # Se l'attuale SL è migliore del candidato, manteniamo l'attuale SL
                # Un SL migliore per SELL è un valore più basso
                if current_pos.sl is not None and candidate_sl > current_pos.sl: # Se il candidato è peggiore dell'attuale
                    request_sl = current_pos.sl
                else:
                    request_sl = candidate_sl
            
            # Arrotonda il request_sl
            request_sl = round(request_sl, symbol_info.digits)
            if request_sl == 0.0: # MT5 non accetta SL a 0.0, significa che non è stato impostato
                request_sl = current_pos.sl # Mantieni il vecchio SL se il nuovo è 0.0
                logger.warning(f"SL calcolato per {ticket} è 0.0, mantenendo il vecchio SL. Potrebbe indicare un problema di calcolo.")


        # Logica per il nuovo Take Profit (simile a SL per stops_level, ma con la direzione opposta)
        if new_tp is not None:
            if pos_type == mt5.ORDER_TYPE_BUY:
                # Per BUY: TP deve essere al di sopra del prezzo corrente (ask) di almeno stops_level
                min_acceptable_tp = current_tick.ask + stops_level_price_units
                candidate_tp = max(new_tp, min_acceptable_tp)
                
                # Se il TP attuale è migliore del candidato (più vicino all'entry, ma comunque in profitto), mantenilo
                # Un TP migliore per BUY è un valore più basso (più vicino all'entry) se si sta riducendo il TP,
                # ma deve essere comunque un profitto. Qui vogliamo solo che si muova a nostro favore.
                if current_pos.tp is not None and candidate_tp < current_pos.tp: # Se il candidato è peggiore dell'attuale (più vicino all'entry)
                    request_tp = current_pos.tp
                else:
                    request_tp = candidate_tp
            else: # SELL
                # Per SELL: TP deve essere al di sotto del prezzo corrente (bid) di almeno stops_level
                max_acceptable_tp = current_tick.bid - stops_level_price_units
                candidate_tp = min(new_tp, max_acceptable_tp)

                # Se il TP attuale è migliore del candidato (più vicino all'entry, ma comunque in profitto), mantenilo
                # Un TP migliore per SELL è un valore più alto (più vicino all'entry)
                if current_pos.tp is not None and candidate_tp > current_pos.tp: # Se il candidato è peggiore dell'attuale (più vicino all'entry)
                    request_tp = current_pos.tp
                else:
                    request_tp = candidate_tp
            
            # Arrotonda il request_tp
            request_tp = round(request_tp, symbol_info.digits)
            if request_tp == 0.0: # MT5 non accetta TP a 0.0
                request_tp = current_pos.tp # Mantieni il vecchio TP se il nuovo è 0.0
                logger.warning(f"TP calcolato per {ticket} è 0.0, mantenendo il vecchio TP. Potrebbe indicare un problema di calcolo.")


        # Prepara la richiesta finale
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": request_sl,
            "tp": request_tp,
            "comment": "Aggiornamento SL/TP FusionAI"
        }

        # Implementa un meccanismo di retry per l'invio dell'ordine
        max_retries = 3
        for attempt in range(max_retries):
            # Aggiungi log dettagliati prima di inviare la richiesta
            logger.debug(f"Tentativo {attempt + 1}/{max_retries} di modificare SL/TP per ticket {ticket}:")
            logger.debug(f"  Posizione Corrente: SL={current_pos.sl}, TP={current_pos.tp}, Tipo={pos_type}")
            logger.debug(f"  Tick Corrente: Bid={current_tick.bid}, Ask={current_tick.ask}")
            logger.debug(f"  stops_level_points: {stops_level_points}, stops_level_price_units: {stops_level_price_units}")
            logger.debug(f"  Nuovo SL (input a _modify_sl_or_tp): {new_sl}, Nuovo TP (input a _modify_sl_or_tp): {new_tp}")
            logger.debug(f"  SL Finale nella richiesta: {request['sl']}, TP Finale nella richiesta: {request['tp']}")

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Ordine {ticket} SL/TP aggiornato. SL: {request['sl']}, TP: {request['tp']}")
                # Dopo aver modificato con successo su MT5, aggiorna il dizionario interno del bot
                for i, p_dict in enumerate(self.bot.open_positions):
                    if p_dict['ticket'] == ticket:
                        self.bot.open_positions[i]['sl'] = request['sl']
                        self.bot.open_positions[i]['tp'] = request['tp']
                        break
                return # Successo, esci dalla funzione
            else:
                logger.warning(f"Tentativo {attempt + 1}/{max_retries} di aggiornamento SL/TP fallito per ticket {ticket}: {result.comment if result else mt5.last_error()}")
                time.sleep(0.5) # Breve pausa prima di riprovare

        logger.error(f"Aggiornamento SL/TP fallito definitivamente per ticket {ticket} dopo {max_retries} tentativi.")


    def stop(self): self.running = False

# --- CLASSE PRINCIPALE ---

class FusionAITrader:
    """
    Classe principale che orchestra tutti i componenti del sistema di trading.
    """
    def __init__(self):
        self.digital_twin = DigitalTwin()
        self.pattern_engine = PatternEngine()
        self.risk_manager = RiskManager()
        self.trade_journal = TradeJournal()
        self.market_sentiment = MarketSentiment()
        self.strategy_engine = StrategyEngine(self.digital_twin, self.pattern_engine)
        self.backtesting_engine = BacktestingEngine()
        self.backtesting_engine.set_bot_dependencies(self.digital_twin, self.strategy_engine, self.risk_manager)
        
        self.ai_brain = None
        self.market_monitor = MarketMonitor()
        self.opportunity_scout = OpportunityScout()
        self.position_manager = PositionManager()
        
        self.market_monitor.set_bot_instance(self)
        self.opportunity_scout.set_bot_instance(self)
        self.position_manager.set_bot_instance(self)

        self.notifications = []
        self.account_info = None
        self.open_positions = [] # Ora conterrà dizionari con metadati
        self.mt5_login = None
        self.mt5_password = None
        self.mt5_server = None
        self.initial_equity = 0
        self.initial_weekly_equity = 0
        self.last_daily_reset = datetime.now(pytz.timezone('Europe/Rome')).replace(hour=0, minute=0, second=0, microsecond=0)
        self.last_weekly_reset = datetime.now(pytz.timezone('Europe/Rome')).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=datetime.now(pytz.timezone('Europe/Rome')).weekday())
        self.lock = threading.Lock()

        # Questi contatori sono ora utilizzati per la gestione dinamica del lotto e l'AI
        self.consecutive_sl_count = 0
        self.consecutive_tp_count = 0

    def initialize_mt5(self, login, password, server, reconnect=False):
        """Inizializza la connessione a MetaTrader 5."""
        if not reconnect:
            self.mt5_login = login
            self.mt5_password = password
            self.mt5_server = server

        try:
            login_int = int(login)
        except Exception as e:
            logger.error(f"Il login di MT5 non è un intero valido: {login} ({e})")
            return False

        if mt5.terminal_info() is not None:
            mt5.shutdown()
            time.sleep(1)

        success = mt5.initialize(
            path=CONFIG['mt5_path'],
            login=login_int,
            password=password,
            server=server
        )
        if not success:
            logger.error(f"Inizializzazione di MT5 fallita: {mt5.last_error()}")
            return False

        self.update_account_state()
        if self.initial_equity == 0 and self.account_info:
            self.initial_equity = self.account_info.equity
            self.initial_weekly_equity = self.account_info.equity
            logger.info(f"Equity iniziale giornaliera e settimanale impostata a: {self.initial_equity}")

        if not self.digital_twin.load_historical_data():
            logger.error("Caricamento dati storici fallito. Il bot potrebbe non funzionare correttamente.")

        self.start_background_services()
        logger.info("FusionAI Trader inizializzato.")
        return True

    def initialize_ai(self, gemini_api_key, openai_api_key=None):
        """Inizializza il cervello AI."""
        if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
            logger.error("Chiave API Gemini non valida. AI Brain non inizializzato.")
            self.handle_notification({"type": "ERROR", "message": "Chiave API Gemini mancante o non valida."})
            return False

        self.ai_brain = AIBrain(gemini_api_key, openai_api_key)
        logger.info("AI Brain inizializzato.")
        self.start_background_services()
        return True

    def start_background_services(self):
        """Avvia i thread di background se non sono già in esecuzione."""
        if not self.market_monitor.is_alive():
            self.market_monitor.start()
            logger.info("MarketMonitor avviato.")

        if self.ai_brain and not self.opportunity_scout.is_alive():
            self.opportunity_scout.start()
            logger.info("OpportunityScout avviato.")

        if not self.position_manager.is_alive():
            self.position_manager.start()
            logger.info("PositionManager avviato.")

    def update_account_state(self):
        """Aggiorna lo stato del conto e le posizioni aperte dal broker."""
        with self.lock:
            self.account_info = mt5.account_info()
            current_mt5_positions = mt5.positions_get()
            
            new_managed_positions = []
            if current_mt5_positions:
                for pos_mt5_obj in current_mt5_positions:
                    pos_dict = pos_mt5_obj._asdict()
                    pos_dict['is_hedged'] = False # Default
                    pos_dict['open_time_m5_bars'] = 0 # Default

                    for existing_pos_dict in self.open_positions:
                        if existing_pos_dict.get('ticket') == pos_mt5_obj.ticket:
                            pos_dict['is_hedged'] = existing_pos_dict.get('is_hedged', False)
                            pos_dict['open_time_m5_bars'] = existing_pos_dict.get('open_time_m5_bars', 0)
                            pos_dict['open_timestamp'] = existing_pos_dict.get('open_timestamp', time.time()) # Mantieni il timestamp di apertura
                            pos_dict['strategy'] = existing_pos_dict.get('strategy', 'N/A') # Mantieni la strategia
                            break
                    new_managed_positions.append(pos_dict)
                
            self.open_positions = new_managed_positions # Sostituisci l'elenco delle posizioni gestite

            # Ricalcola is_hedged per le posizioni gestite
            for p_dict in self.open_positions:
                for other_p_dict in self.open_positions:
                    if (p_dict['ticket'] != other_p_dict['ticket'] and
                        p_dict['symbol'] == other_p_dict['symbol'] and
                        p_dict['type'] != other_p_dict['type']):
                        p_dict['is_hedged'] = True
                        other_p_dict['is_hedged'] = True
            
            now = datetime.now(pytz.timezone('Europe/Rome'))
            if now.day != self.last_daily_reset.day:
                self.initial_equity = self.account_info.equity if self.account_info else 0
                self.last_daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info(f"Reset drawdown/profitto giornaliero. Nuova equity iniziale: {self.initial_equity}")
                self.consecutive_sl_count = 0
                self.consecutive_tp_count = 0

            if (now.weekday() == 0 and now.hour == 0 and now.minute <= 5 and
                (now - self.last_weekly_reset).days >= 7):
                self.initial_weekly_equity = self.account_info.equity if self.account_info else 0
                self.last_weekly_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info(f"Reset drawdown/profitto settimanale. Nuova equity iniziale settimanale: {self.initial_weekly_equity}")

    def calculate_weekly_performance(self):
        """Calcola la performance settimanale."""
        if not self.account_info or self.initial_weekly_equity <= 0:
            return 0, 0
        current_equity = self.account_info.equity
        drawdown_percent = ((self.initial_weekly_equity - current_equity) / self.initial_weekly_equity) * 100
        profit_percent = ((current_equity - self.initial_weekly_equity) / current_equity) * 100 # Calcola profitto sul capitale corrente
        return max(0, drawdown_percent), max(0, profit_percent)

    def handle_notification(self, notification):
        """Gestisce le notifiche interne e le azioni di trading basate sull'AI."""
        with self.lock:
            notification_data = {"timestamp": datetime.now(italy_tz).isoformat(), "data": notification}
            self.notifications.insert(0, notification_data)
            self.notifications = self.notifications[:50]

        logger.info(f"Notifica: {notification.get('analysis_text', notification.get('message', 'N/A'))} [Confidenza: {notification.get('confidence_score', 'N/A')}]")

        # Logica di esecuzione basata sulla CONFIENZA AI FINALE
        if CONFIG['auto_trade_enabled'] and notification.get('action') in ["BUY", "SELL"] and \
           notification.get('confidence_score', 0) >= CONFIG['auto_trade_min_confidence']:
            logger.info(f"Auto-trading attivato da notifica con confidenza {notification['confidence_score']}.")
            self.execute_trade_from_ai(notification)
        elif notification.get('action') == "WAIT":
            logger.info(f"L'AI raccomanda ATTESA per {notification.get('symbol')}.")
        # Rimosse le azioni CLOSE_PARTIAL e REVERSE da qui
        # elif notification.get('action') == "CLOSE_PARTIAL":
        #     logger.info(f"L'AI raccomanda azione sul trade: {notification.get('action')} per {notification.get('symbol')}.")
        #     positions_for_symbol = [pos_dict for pos_dict in self.open_positions if pos_dict['symbol'] == notification.get('symbol')]
        #     if not positions_for_symbol:
        #         logger.warning(f"L'AI ha raccomandato {notification.get('action')} per {notification.get('symbol')}, ma nessuna posizione trovata.")
        #         return
        #
        #     for pos_to_close_dict in positions_for_symbol:
        #         # Applica il cooldown AI anche qui per CLOSE_PARTIAL
        #         if (time.time() - pos_to_close_dict.get('open_timestamp', 0) < CONFIG['ai_decision_cooldown_seconds']):
        #             logger.info(f"AI ha raccomandato CLOSE_PARTIAL per {pos_to_close_dict['symbol']} (Ticket: {pos_to_close_dict['ticket']}), ma è in cooldown. Ignorato.")
        #             continue # Salta questa chiusura parziale se in cooldown
        #
        #         volume_to_close_ratio = notification.get('volume_to_close_ratio', 0.5) # Default a 0.5 se non specificato dall'AI
        #         volume_to_close = pos_to_close_dict['volume'] * volume_to_close_ratio
        #         
        #         symbol_info = mt5.symbol_info(pos_to_close_dict['symbol'])
        #         if symbol_info:
        #             volume_to_close = round(volume_to_close / symbol_info.volume_step) * symbol_info.volume_step
        #             volume_to_close = max(volume_to_close, symbol_info.volume_min) # Assicurati che sia almeno il volume minimo
        #         else:
        #             volume_to_close = round(volume_to_close, 2) # Fallback
        #
        #         if volume_to_close > 0 and volume_to_close < pos_to_close_dict['volume']:
        #             self.close_position(pos_to_close_dict['ticket'], pos_to_close_dict['symbol'], pos_to_close_dict['type'],
        #                                  volume_to_close, "AI_CLOSE_PARTIAL", notification)
        #         else:
        #             logger.warning(f"Volume per chiusura parziale non valido per {pos_to_close_dict['ticket']}: {volume_to_close:.2f}")
        # elif notification.get('action') == "REVERSE": # L'AI raccomanda solo REVERSE, non REVERSE_TRADE
        #     logger.info(f"L'AI raccomanda REVERSE per {notification.get('symbol')}.")
        #     pass # L'AI ha solo dato un'indicazione, non un comando diretto di chiusura qui.

    def close_position(self, ticket, symbol, trade_type_enum, volume, reason, ai_verdict):
        """Funzione centralizzata per chiudere una posizione."""
        # Lista delle modalità di riempimento da provare in ordine di preferenza
        filling_modes_to_try = [
            get_symbol_filling_mode(symbol), # La modalità preferita dal broker
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN
        ]
        # Rimuovi duplicati e NaN se get_symbol_filling_mode restituisce lo stesso valore di un fallback
        filling_modes_to_try = list(dict.fromkeys(filling_modes_to_try))

        current_tick = mt5.symbol_info_tick(symbol)
        if not current_tick:
            logger.error(f"Impossibile ottenere tick corrente per chiusura posizione {ticket} su {symbol}. Annullato.")
            self.handle_notification({"type": "ERROR", "message": f"Errore chiusura: Nessun tick live per {symbol}."})
            return

        price_to_use = current_tick.ask if trade_type_enum == mt5.ORDER_TYPE_BUY else current_tick.bid

        order_sent_successfully = False
        final_result = None
        closed_profit = 0.0 # Inizializza il profitto a 0.0

        for fill_mode in filling_modes_to_try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if trade_type_enum == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": price_to_use,
                "deviation": CONFIG['slippage_points'],
                "comment": f"FusionAI {reason}",
                "type_filling": fill_mode, # Utilizzo dinamico con reintento
            }
            
            logger.info(f"Tentativo di chiudere posizione {ticket} ({symbol}) con filling mode {fill_mode}. Volume: {volume:.2f}.")
            result = mt5.order_send(request)
            final_result = result # Salva l'ultimo risultato

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                order_sent_successfully = True
                # RECUPERA IL PROFITTO DAL DEAL
                if result.deal:
                    deals = mt5.history_deals_get(ticket=result.deal)
                    if deals:
                        closed_profit = deals[0].profit
                        logger.info(f"Profitto recuperato dal deal {result.deal}: {closed_profit:.2f}")
                    else:
                        logger.warning(f"Deal {result.deal} non trovato per la posizione {ticket} dopo la chiusura.")
                else:
                    logger.warning(f"Nessun deal ticket in OrderSendResult per la posizione {ticket} dopo la chiusura.")
                break # Ordine inviato con successo, esci dal ciclo delle modalità di riempimento
            elif result and result.retcode != mt5.TRADE_RETCODE_DONE: # Se non si è completato, prova il prossimo modo
                logger.warning(f"Modalità di riempimento {fill_mode} fallita per chiusura {symbol} con retcode {result.retcode}. Tentativo con la prossima.")
                time.sleep(0.5) # Breve pausa prima di riprovare
                continue # Prova il prossimo filling mode
            else:
                break # Altro errore, esci dal ciclo delle modalità di riempimento

        if order_sent_successfully:
            logger.info(f"Posizione {ticket} ({symbol}) chiusa per '{reason}'. Volume: {volume:.2f}. P/L: {closed_profit:.2f}.")
            # Aggiorna i contatori di vincite/perdite consecutive
            if closed_profit > 0:
                self.consecutive_tp_count += 1
                self.consecutive_sl_count = 0
            elif closed_profit < 0:
                self.consecutive_sl_count += 1
                self.consecutive_tp_count = 0
            
            journal_entry = {
                "timestamp": datetime.now(italy_tz).isoformat(),
                "symbol": symbol,
                "action": reason,
                "type": "Closed",
                "volume": volume,
                "entry_price": ai_verdict.get('entry_price', 'N/A'), # Potrebbe non essere disponibile per azioni su posizioni
                "close_price": final_result.price,
                "profit_loss": closed_profit, # Usa il profitto recuperato
                "ai_analysis": ai_verdict.get('analysis_text', ''),
                "mt5_result": final_result._asdict(),
                "ticket": ticket
            }
            self.trade_journal.log_trade(journal_entry)
            
            # NUOVO: Aggiorna la performance della strategia dopo la chiusura del trade
            if 'strategy' in ai_verdict: # Se l'AI ha fornito la strategia che ha generato il trade
                self.strategy_engine.update_strategy_performance(ai_verdict['strategy'], closed_profit)

            self.handle_notification({"type": "INFO", "message": f"Posizione {symbol} (Ticket: {ticket}) chiusa da AI. P/L: {closed_profit:.2f}.", "action": reason})
        else:
            logger.error(f"Chiusura posizione {ticket} ({symbol}) fallita per '{reason}': {final_result.comment if final_result else mt5.last_error()}")
            self.handle_notification({"type": "ERROR", "message": f"Errore chiusura posizione {symbol} (Ticket: {ticket}): {final_result.comment if final_result else mt5.last_error()}", "action": reason})

    def execute_trade_from_ai(self, ai_decision):
        """Esegue un trade basato sulla decisione dell'AI."""
        symbol = ai_decision['symbol']
        action = ai_decision['action']
        strategy_name = ai_decision.get('strategy', 'AI_Decision') # Recupera il nome della strategia

        if action not in ["BUY", "SELL"]: return

        # --- GESTIONE POSIZIONI OPPOSTE PRIMA DI APRIRE NUOVO TRADE ---
        # Se l'AI decide di aprire un trade, controlla se ci sono posizioni aperte nel senso opposto
        # e chiudile prima di procedere.
        opposing_positions = []
        if action == "BUY":
            opposing_positions = [pos_dict for pos_dict in self.open_positions if pos_dict['symbol'] == symbol and pos_dict['type'] == mt5.ORDER_TYPE_SELL]
        elif action == "SELL":
            opposing_positions = [pos_dict for pos_dict in self.open_positions if pos_dict['symbol'] == symbol and pos_dict['type'] == mt5.ORDER_TYPE_BUY]

        if opposing_positions:
            logger.info(f"Trovate posizioni opposte per {symbol}. Chiusura prima di aprire il nuovo trade {action}.")
            for pos_to_close_dict in opposing_positions:
                self.close_position(pos_to_close_dict['ticket'], pos_to_close_dict['symbol'], pos_to_close_dict['type'],
                                     pos_to_close_dict['volume'], f"AI_CLOSE_OPPOSITE_FOR_{action}", {"analysis_text": f"Chiusura posizione opposta per aprire {action}.", "strategy": pos_to_close_dict.get('strategy', 'N/A')})
            # Diamo un piccolo momento per l'esecuzione della chiusura prima di procedere con l'apertura
            time.sleep(1)
            # Ricarica lo stato delle posizioni dopo la chiusura
            self.update_account_state()


        current_trades_for_symbol = len([pos_dict for pos_dict in self.open_positions if pos_dict['symbol'] == symbol])
        if current_trades_for_symbol >= CONFIG['max_open_trades_per_symbol']:
            logger.info(f"Raggiunto limite di {CONFIG['max_open_trades_per_symbol']} trade aperti per {symbol}. Salto apertura.")
            self.handle_notification({"type": "INFO", "message": f"Limite trade per {symbol} raggiunto ({CONFIG['max_open_trades_per_symbol']}). Operazione saltata."})
            return

        trade_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

        tick = self.digital_twin.get_live_tick(symbol)
        if not tick:
            logger.error(f"Impossibile aprire trade per {symbol}, nessun tick live. Annullato.")
            self.handle_notification({"type": "ERROR", "message": f"Errore trade: Nessun tick live per {symbol}."})
            return

        entry_price = tick['ask'] if trade_type == mt5.ORDER_TYPE_BUY else tick['bid']

        # --- CONTROLLI PRE-TRADE (CENTRALIZZATI QUI) ---
        # 1. Controllo Spread Anomalo
        symbol_info_for_spread_check = mt5.symbol_info(symbol)
        current_spread_points = (tick['ask'] - tick['bid']) / symbol_info_for_spread_check.point if symbol_info_for_spread_check and isinstance(tick['ask'], (int, float)) and isinstance(tick['bid'], (int, float)) else 0
        max_spread = CONFIG['max_spread_points_allowed'].get(symbol, CONFIG['max_spread_points_allowed']['DEFAULT'])
        if current_spread_points > max_spread:
            logger.warning(f"Blocco trade per {symbol}: Spread ({current_spread_points:.2f} punti) supera il limite consentito ({max_spread} punti).")
            self.handle_notification({"type": "WARNING", "message": f"Blocco trade per {symbol}: Spread troppo alto ({current_spread_points:.2f} punti)."
                                                                   + f" L'AI raccomanda ATTESA. (Motivo: Spread {current_spread_points:.2f} > {max_spread})."})
            return

        # 2. Controllo Volatilità Anomala
        atr_data_for_vol_check = self.digital_twin.get_chart_data(symbol, "M5")
        if not atr_data_for_vol_check.empty and len(atr_data_for_vol_check) > CONFIG['volatility_check_period_candles'] and 'ATR' in atr_data_for_vol_check.columns:
            recent_atr_avg = atr_data_for_vol_check['ATR'].iloc[-CONFIG['volatility_check_period_candles']:-1].mean()
            current_atr_for_check = atr_data_for_vol_check['ATR'].iloc[-1]
            
            vol_rules = CONFIG['volatility_check_atr_multiplier'].get(symbol, CONFIG['volatility_check_atr_multiplier']['DEFAULT'])
            min_multiplier = vol_rules['min_multiplier']
            max_multiplier = vol_rules['max_multiplier']

            if not np.isnan(recent_atr_avg) and recent_atr_avg > 0:
                if current_atr_for_check > recent_atr_avg * max_multiplier:
                    warning_message = (
                        f"Blocco trade per {symbol}: Volatilità ({current_atr_for_check:.5f}) troppo alta rispetto alla media ({recent_atr_avg:.5f}). "
                        f"L'AI raccomanda ATTESA. (Motivo: ATR {current_atr_for_check:.5f} > {recent_atr_avg:.5f} * {max_multiplier})."
                    )
                    self.handle_notification({"type": "WARNING", "message": warning_message})
                    return
                elif current_atr_for_check < recent_atr_avg * min_multiplier:
                    warning_message = (
                        f"Blocco trade per {symbol}: Bassa volatilità anomala. "
                        f"L'AI raccomanda ATTESA. (Motivo: ATR {current_atr_for_check:.5f} < {recent_atr_avg:.5f} * {min_multiplier})."
                    )
                    self.handle_notification({"type": "WARNING", "message": warning_message})
                    return
        else:
             logger.warning(f"Dati ATR insufficienti per controllo volatilità pre-trade per {symbol}. Salto il controllo.")


        # Fine Controlli Pre-Trade

        atr_data = self.digital_twin.get_chart_data(symbol, "M5")
        if atr_data.empty or 'ATR' not in atr_data.columns or atr_data.iloc[-1]['ATR'] <= 0:
            logger.error(f"ATR non valido per {symbol} sul timeframe M5, trade annullato.")
            self.handle_notification({"type": "ERROR", "message": f"Errore trade: ATR non valido per {symbol}."})
            return
        atr_value = atr_data.iloc[-1]['ATR']

        sl_price, primary_tp_price_calculated = self.risk_manager.calculate_dynamic_sl_tp(symbol, trade_type, entry_price, atr_value)
        if sl_price is None or primary_tp_price_calculated is None:
            logger.error(f"Calcolo SL/TP fallito per {symbol}, trade annullato.")
            self.handle_notification({"type": "ERROR", "message": f"Errore trade: Calcolo SL/TP fallito per {symbol}."})
            return
        
        # Usa il TP suggerito dall'AI se disponibile e valido, altrimenti il calcolato
        # (Questo è per il TP1 dell'AI, ma verrà sovrascritto dai TP fissi se applicabili)
        primary_tp_price = ai_decision.get('suggested_tp_price')
        if primary_tp_price is None:
            primary_tp_price = primary_tp_price_calculated
        else: # Se l'AI ha suggerito un TP, assicurati che sia nella direzione giusta e abbia senso rispetto all'SL
            if trade_type == mt5.ORDER_TYPE_BUY and primary_tp_price <= entry_price:
                logger.warning(f"AI suggerimento TP ({primary_tp_price:.5f}) non valido per BUY. Usando TP calcolato: {primary_tp_price_calculated:.5f}.")
                primary_tp_price = primary_tp_price_calculated
            elif trade_type == mt5.ORDER_TYPE_SELL and primary_tp_price >= entry_price:
                logger.warning(f"AI suggerimento TP ({primary_tp_price:.5f}) non valido per SELL. Usando TP calcolato: {primary_tp_price_calculated:.5f}.")
                primary_tp_price = primary_tp_price_calculated


        # Passa i contatori di SL/TP consecutivi al calcolo del lotto
        lot_size = self.risk_manager.calculate_lot_size(
            symbol, 
            abs(entry_price - sl_price) / mt5.symbol_info(symbol).point, 
            self.account_info.equity,
            self.consecutive_sl_count,
            self.consecutive_tp_count
        )
        if lot_size <= 0:
            logger.error(f"Lot size calcolata di {lot_size} non valida per {symbol}. Trade annullato.")
            self.handle_notification({"type": "ERROR", "message": f"Errore trade: Lot size non valida per {symbol}."})
            return

        # Lista delle modalità di riempimento da provare in ordine di preferenza
        filling_modes_to_try = [
            get_symbol_filling_mode(symbol), # La modalità preferita dal broker
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN
        ]

        # Rimuovi duplicati e NaN se get_symbol_filling_mode restituisce lo stesso valore di un fallback
        filling_modes_to_try = list(dict.fromkeys(filling_modes_to_try))

        order_sent_successfully = False
        final_result = None

        # NUOVO: Determinazione dei TP per i sotto-trade
        tp_for_sub_trades = []
        if symbol in CONFIG['xauusd_btc_fixed_tps']:
            fixed_tps = CONFIG['xauusd_btc_fixed_tps'][symbol][action] # Usa BUY/SELL per scegliere i TP
            tp_for_sub_trades.append(fixed_tps["TP1"])
            tp_for_sub_trades.append(fixed_tps["TP2"])
            tp_for_sub_trades.append(fixed_tps["TP3"])
        else:
            # Usa i moltiplicatori standard se non ci sono TP fissi
            for multiplier in CONFIG['tp_levels_multiplier']:
                current_tp_distance = (atr_value * CONFIG['atr_sl_multiplier']) * multiplier
                tp_price = entry_price + current_tp_distance if trade_type == mt5.ORDER_TYPE_BUY else entry_price - current_tp_distance
                tp_for_sub_trades.append(round(tp_price, mt5.symbol_info(symbol).digits))
        
        # NUOVO: Calcolo dei volumi per la chiusura parziale a TP1
        lot_tp1_portion = round(lot_size * CONFIG['tp1_partial_close_ratio'], 2)
        lot_remainder_portion = round(lot_size - lot_tp1_portion, 2)

        # Assicurati che i volumi siano validi rispetto ai minimi del broker
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            lot_tp1_portion = max(lot_tp1_portion, symbol_info.volume_min)
            lot_remainder_portion = max(lot_remainder_portion, symbol_info.volume_min)
            # Se il lotto rimanente è troppo piccolo, aggiungilo al lotto TP1
            if lot_remainder_portion < symbol_info.volume_min:
                lot_tp1_portion += lot_remainder_portion
                lot_remainder_portion = 0
            # Se il lotto TP1 è troppo piccolo, aggiungilo al lotto rimanente
            if lot_tp1_portion < symbol_info.volume_min and lot_remainder_portion > 0:
                lot_remainder_portion += lot_tp1_portion
                lot_tp1_portion = 0
            # Se entrambi sono troppo piccoli, forza il lotto minimo per il primo trade
            if lot_tp1_portion < symbol_info.volume_min and lot_remainder_portion < symbol_info.volume_min:
                lot_tp1_portion = symbol_info.volume_min
                lot_remainder_portion = 0 # O gestisci come errore se non si può aprire neanche il minimo
        else: # Fallback se info simbolo non disponibile
            lot_tp1_portion = round(lot_tp1_portion, 2)
            lot_remainder_portion = round(lot_remainder_portion, 2)


        # --- Apertura del primo sotto-trade (per TP1) ---
        if lot_tp1_portion > 0:
            current_tp_price = tp_for_sub_trades[0] # TP1
            sub_trade_comment = f"AI {action} C:{ai_decision.get('confidence_score', 'N/A')}% TP1"
            if len(sub_trade_comment) > 25: sub_trade_comment = sub_trade_comment[:22] + "..."

            for fill_mode in filling_modes_to_try:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_tp1_portion,
                    "type": trade_type,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": current_tp_price,
                    "deviation": CONFIG['slippage_points'],
                    "magic": 12345, # Magic number per il primo trade
                    "comment": sub_trade_comment,
                    "type_filling": fill_mode,
                }
                logger.info(f"Tentativo di aprire SOTTO-TRADE TP1 con filling mode {fill_mode}: {action} {lot_tp1_portion} di {symbol} @ {entry_price}. SL={sl_price}, TP={current_tp_price}")
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SOTTO-TRADE TP1 APERTO: {action} {lot_tp1_portion} lotti di {symbol} @ {result.price}. Ticket: {result.order}. SL={sl_price}, TP={current_tp_price}")
                    self.trade_journal.log_trade({
                        "timestamp": datetime.now(italy_tz).isoformat(), "symbol": symbol, "action": "Open Sub-Trade TP1",
                        "type": action, "volume": lot_tp1_portion, "entry_price": result.price, "sl": sl_price, "tp": current_tp_price,
                        "ai_confidence": ai_decision.get('confidence_score', 'N/A'), "ai_analysis": ai_decision.get('analysis_text', ''),
                        "mt5_result": result._asdict(), "strategy": strategy_name
                    })
                    self.consecutive_tp_count += 1
                    self.consecutive_sl_count = 0
                    self.handle_notification({"type": "SUCCESS", "message": f"Apertura trade iniziata per {symbol}: {action} {lot_tp1_portion} lotti (TP1)."})
                    order_sent_successfully = True
                    break
                else:
                    logger.warning(f"Modalità di riempimento {fill_mode} fallita per SOTTO-TRADE TP1 con retcode {result.retcode}. Tentativo con la prossima.")
                    final_result = result # Salva l'ultimo risultato fallito
                    time.sleep(0.5)
            
            if not order_sent_successfully:
                error_msg = f"APERTURA SOTTO-TRADE TP1 FALLITA DEFINITIVAMENTE: {final_result.comment if final_result else mt5.last_error()}"
                logger.error(error_msg)
                self.trade_journal.log_trade({
                    "timestamp": datetime.now(italy_tz).isoformat(), "symbol": symbol, "action": "Failed Open Sub-Trade TP1",
                    "type": action, "volume": lot_tp1_portion, "entry_price": entry_price, "sl": sl_price, "tp": current_tp_price,
                    "ai_confidence": ai_decision.get('confidence_score', 'N/A'), "ai_analysis": ai_decision.get('analysis_text', ''),
                    "mt5_result": final_result._asdict() if final_result else {"error": mt5.last_error()}, "strategy": strategy_name
                })
                self.consecutive_sl_count += 1
                self.consecutive_tp_count = 0
                self.handle_notification({"type": "ERROR", "message": f"Errore nell'apertura del trade TP1 per {symbol}: {final_result.comment if final_result else mt5.last_error()}"})
                return # Se il primo trade fallisce, non tentare il secondo

        # --- Apertura del secondo sotto-trade (per il resto, con TP più lontano per trailing) ---
        if lot_remainder_portion > 0:
            current_tp_price = tp_for_sub_trades[-1] # Ultimo TP per il resto (per trailing)
            sub_trade_comment = f"AI {action} C:{ai_decision.get('confidence_score', 'N/A')}% REM"
            if len(sub_trade_comment) > 25: sub_trade_comment = sub_trade_comment[:22] + "..."

            order_sent_successfully_remainder = False
            final_result_remainder = None

            for fill_mode in filling_modes_to_try:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_remainder_portion,
                    "type": trade_type,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": current_tp_price,
                    "deviation": CONFIG['slippage_points'],
                    "magic": 12346, # Magic number diverso per il secondo trade
                    "comment": sub_trade_comment,
                    "type_filling": fill_mode,
                }
                logger.info(f"Tentativo di aprire SOTTO-TRADE REM con filling mode {fill_mode}: {action} {lot_remainder_portion} di {symbol} @ {entry_price}. SL={sl_price}, TP={current_tp_price}")
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"SOTTO-TRADE REM APERTO: {action} {lot_remainder_portion} lotti di {symbol} @ {result.price}. Ticket: {result.order}. SL={sl_price}, TP={current_tp_price}")
                    self.trade_journal.log_trade({
                        "timestamp": datetime.now(italy_tz).isoformat(), "symbol": symbol, "action": "Open Sub-Trade Remainder",
                        "type": action, "volume": lot_remainder_portion, "entry_price": result.price, "sl": sl_price, "tp": current_tp_price,
                        "ai_confidence": ai_decision.get('confidence_score', 'N/A'), "ai_analysis": ai_decision.get('analysis_text', ''),
                        "mt5_result": result._asdict(), "strategy": strategy_name
                    })
                    order_sent_successfully_remainder = True
                    break
                else:
                    logger.warning(f"Modalità di riempimento {fill_mode} fallita per SOTTO-TRADE REM con retcode {result.retcode}. Tentativo con la prossima.")
                    final_result_remainder = result # Salva l'ultimo risultato fallito
                    time.sleep(0.5)
            
            if not order_sent_successfully_remainder:
                error_msg = f"APERTURA SOTTO-TRADE REM FALLITA DEFINITIVAMENTE: {final_result_remainder.comment if final_result_remainder else mt5.last_error()}"
                logger.error(error_msg)
                self.trade_journal.log_trade({
                    "timestamp": datetime.now(italy_tz).isoformat(), "symbol": symbol, "action": "Failed Open Sub-Trade Remainder",
                    "type": action, "volume": lot_remainder_portion, "entry_price": entry_price, "sl": sl_price, "tp": current_tp_price,
                    "ai_confidence": ai_decision.get('confidence_score', 'N/A'), "ai_analysis": ai_decision.get('analysis_text', ''),
                    "mt5_result": final_result_remainder._asdict() if final_result_remainder else {"error": mt5.last_error()}, "strategy": strategy_name
                })
                # Non incrementiamo consecutive_sl_count qui perché il primo trade è già stato gestito.


    def handle_user_query(self, user_msg, view_context):
        """
        Gestisce le query dell'utente tramite la chat, fornendo risposte contestualizzate
        e gestendo le richieste di modifica della configurazione.
        """
        if not self.ai_brain: return {"analysis_text": "AI non configurata. Per favore, inizializza il mio cervello con le API keys!", "action": "ERROR", "symbol": "N/A", "confidence_score": 0}

        symbol = view_context.get('symbol') if view_context.get('symbol') != "" else CONFIG['symbols'][0]
        timeframe = view_context.get('timeframe') if view_context.get('timeframe') != "" else "M5"

        chart_data = self.digital_twin.get_chart_data(symbol, timeframe)
        if chart_data.empty or len(chart_data) < 200:
            return {"analysis_text": f"Mmmh, sembra che non abbia abbastanza dati per {symbol} su {timeframe} per fare un'analisi sensata. Dammi un po' di tempo per raccogliere informazioni, ok?", "action": "WAIT", "symbol": symbol, "confidence_score": 0}

        last_candle = chart_data.iloc[-1]
        patterns = self.pattern_engine.detect_all_patterns(chart_data)
        current_sentiment = self.market_sentiment.get_sentiment(symbol)
        active_signals = self.strategy_engine.generate_signals(symbol, timeframe)
        
        live_tick = self.digital_twin.get_live_tick(symbol)
        live_price_info = "Prezzo Live: N/D"
        if live_tick:
            bid_price = live_tick.get('bid')
            ask_price = live_tick.get('ask')
            spread_value = live_tick.get('spread', 'N/D')
            
            if isinstance(bid_price, (int, float)) and isinstance(ask_price, (int, float)):
                live_price_info = f"Prezzo Live (Bid/Ask): {bid_price:.5f}/{ask_price:.5f}, Spread: {spread_value}"
            else:
                live_price_info = f"Prezzo Live (Bid/Ask): N/A/N/A, Spread: {spread_value}"


        higher_tf_data_h4 = self.digital_twin.get_chart_data(symbol, "H4")
        higher_tf_data_d1 = self.digital_twin.get_chart_data(symbol, "D1")
        
        sr_levels = self.digital_twin._calculate_support_resistance(self.digital_twin.get_chart_data(symbol, "H1"), symbol)
        sr_info = f"Livelli Supporto/Resistenza (H1): {sr_levels}" if sr_levels else "Nessun livello S/R rilevato."

        # Gestione sicura per MFI, OBV, VPT
        current_mfi = last_candle['MFI'] if 'MFI' in last_candle else np.nan
        current_obv = last_candle['OBV'] if 'OBV' in last_candle else np.nan
        current_vpt = last_candle['VPT'] if 'VPT' in last_candle else np.nan

        account_info_text = "Stato Conto: Non disponibile"
        daily_drawdown_info = "Drawdown Giornaliero: N/A"
        daily_profit_info = "Profitto Giornaliero: N/A"
        weekly_drawdown_info = "Drawdown Settimanale: N/A"
        weekly_profit_info = "Profitto Settimanale: N/A"
        open_positions_summary = "Nessuna posizione aperta."

        if self.account_info:
            account_info_text = f"Stato Conto: Equity={self.account_info.equity:.2f}, Posizioni Aperte={len(self.open_positions)}"
            
            initial_equity_val = self.initial_equity if isinstance(self.initial_equity, (int, float)) else 0
            current_equity_val = self.account_info.equity if isinstance(self.account_info.equity, (int, float)) else 0
            
            if initial_equity_val > 0:
                current_daily_drawdown_percentage = ((initial_equity_val - current_equity_val) / initial_equity_val) * 100
                current_daily_profit_percentage = ((current_equity_val - initial_equity_val) / initial_equity_val) * 100
            else:
                current_daily_drawdown_percentage = 0
                current_daily_profit_percentage = 0

            daily_drawdown_info = f"Drawdown Giornaliero ({current_daily_drawdown_percentage:.2f}% >= {CONFIG['max_daily_drawdown_percentage']}%): {current_daily_drawdown_percentage >= CONFIG['max_daily_drawdown_percentage']}"
            daily_profit_info = f"Profitto Giornaliero ({current_daily_profit_percentage:.2f}% >= {CONFIG['daily_profit_target_percentage']}%): {current_daily_profit_percentage >= CONFIG['daily_profit_target_percentage']}"

            weekly_drawdown, weekly_profit = self.calculate_weekly_performance()
            weekly_drawdown_info = f"Drawdown Settimanale ({weekly_drawdown:.2f}% >= {CONFIG['max_weekly_drawdown_percentage']}%): {weekly_drawdown >= CONFIG['max_weekly_drawdown_percentage']}"
            weekly_profit_info = f"Profitto Settimanale ({weekly_profit:.2f}% >= {CONFIG['weekly_profit_target_percentage']}%): {weekly_profit >= CONFIG['weekly_profit_target_percentage']}"

            if self.open_positions:
                open_positions_summary = "Posizioni Aperte:\n" + "\n".join([
                    f"- Ticket: {p['ticket']}, Simbolo: {p['symbol']}, Tipo: {'BUY' if p['type'] == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {p['volume']:.2f}, Entry: {p['price_open']:.5f}, SL: {p['sl']:.5f}, TP: {p['tp']:.5f}, Profitto Corrente: {p['profit']:.2f}"
                    for p in self.open_positions
                ])


        context = (f"Domanda utente: '{user_msg}'\n"
                   f"Contesto Vista: Simbolo={symbol}, Timeframe={timeframe}\n"
                   f"{live_price_info}\n"
                   f"{account_info_text}\n"
                   f"{open_positions_summary}\n"
                   f"Dati Ultima Candela ({timeframe}):\n{last_candle.to_string()}\n"
                   f"Pattern Candlestick Rilevati: {', '.join([p['type'] for p in patterns]) if patterns else 'Nessuno'}.\n"
                   f"Segnali Strategici Attivi: {json.dumps(active_signals) if active_signals else 'Nessuno'}.\n"
                   f"Contesto Multi-Timeframe (H4): RSI={higher_tf_data_h4.iloc[-1]['RSI']:.2f if 'RSI' in higher_tf_data_h4.columns else 'N/A'}, SMA_50={higher_tf_data_h4.iloc[-1]['SMA_50']:.5f if 'SMA_50' in higher_tf_data_h4.columns else 'N/A'} (se disponibili)\n"
                   f"Contesto Multi-Timeframe (D1): SMA_200={higher_tf_data_d1.iloc[-1]['SMA_200']:.5f if 'SMA_200' in higher_tf_data_d1.columns else 'N/A'} (se disponibili)\n"
                   f"MFI corrente ({timeframe}): {current_mfi:.2f} (se disponibile)\n"
                   f"OBV corrente ({timeframe}): {current_obv:.2f} (se disponibile)\n"
                   f"VPT corrente ({timeframe}): {current_vpt:.2f} (se disponibile)\n"
                   f"{sr_info}\n"
                   f"Limiti Operativi Attivi: \n"
                   f"  {daily_drawdown_info}.\n"
                   f"  {daily_profit_info}.\n"
                   f"  {weekly_drawdown_info}.\n"
                   f"  {weekly_profit_info}.\n"
                   f"SL consecutivi: {self.consecutive_sl_count}\n" # Passa il conteggio SL consecutivi
                   f"TP consecutivi: {self.consecutive_tp_count}\n" # Passa il conteggio TP consecutivi
                   )
        
        user_msg_lower = user_msg.lower()
        ai_response_message_prefix = ""
        ai_action_override = None
        config_update_request = None

        # Riconoscimento delle intenzioni dell'utente per la chat
        if "ciao" in user_msg_lower or "generale" in user_msg_lower or "come la vedi" in user_msg_lower or "cosa ne pensi" in user_msg_lower or "consigli" in user_msg_lower:
            ai_response_message_prefix = "Certo, ecco la mia visione del mercato e del tuo conto. Preparati, sono sempre sul pezzo!"
        elif "analisi" in user_msg_lower and ("simbolo" in user_msg_lower or symbol.lower() in user_msg_lower):
            ai_response_message_prefix = f"Certamente, analizziamo in dettaglio {symbol}. Vediamo cosa bolle in pentola..."
        elif ("apri trade" in user_msg_lower or "opera" in user_msg_lower) and "si" not in user_msg_lower and "no" not in user_msg_lower:
            ai_response_message_prefix = "Capito, vuoi una raccomandazione operativa. Lascia che valuti il setup più promettente al momento... non ti deluderò, spero!"
        elif "spiega" in user_msg_lower and ("non operi" in user_msg_lower or "non apri" in user_msg_lower or "wait" in user_msg_lower):
            ai_response_message_prefix = "Certo, il motivo per cui preferisco attendere è semplice: la pazienza è la virtù dei forti (e dei conti in profitto!)."
            ai_action_override = "WAIT"
        elif "storico" in user_msg_lower or "trade passati" in user_msg_lower:
            ai_response_message_prefix = "Ah, il passato! Diamo un'occhiata al nostro glorioso (o meno) registro operazioni recenti. Spero tu abbia fatto i compiti!"
        elif "conti" in user_msg_lower or "saldo" in user_msg_lower or "equity" in user_msg_lower or "posizioni aperte" in user_msg_lower:
            ai_response_message_prefix = "Certo, ecco lo stato attuale del tuo conto. Spero che i numeri ti sorridano quanto a me la vista di un trend pulito!"
        elif "chiudi trade" in user_msg_lower:
            ai_response_message_prefix = f"Comprendo. Valuterò la chiusura delle posizioni su {symbol} in base alla situazione attuale. A volte bisogna saper mollare la presa, sai?"
            ai_action_override = "CLOSE_FULL"
        elif "chiudi parziale" in user_msg_lower or "riduci posizione" in user_msg_lower:
            ai_response_message_prefix = f"Ricevuto. Valuto la possibilità di chiudere parzialmente la posizione su {symbol}. Meglio un uovo oggi che una gallina domani, no?"
            ai_action_override = "CLOSE_PARTIAL"
        elif "inverti" in user_msg_lower and "trade" in user_msg_lower:
            ai_response_message_prefix = f"Ricevuto. Analizzo la possibilità di invertire le posizioni su {symbol}. A volte cambiare idea è segno di intelligenza... o di follia, dipende dal risultato!"
            ai_action_override = "REVERSE" # L'AI raccomanda "REVERSE", non "REVERSE_TRADE"
        elif "cambia parametro" in user_msg_lower or "modifica" in user_msg_lower or "setta" in user_msg_lower:
            # Tenta di estrarre chiave e valore dalla richiesta
            match = re.search(r"(cambia|modifica|setta)\s+(.+?)\s+a\s+([0-9\.]+)", user_msg_lower)
            if match:
                key = match.group(2).strip().replace(" ", "_") # Sostituisce spazi con underscore per matchare le chiavi config
                value_str = match.group(3).strip()
                try:
                    # Tenta di convertire al tipo corretto
                    if key in ["slippage_points", "max_open_trades_per_symbol", "volatility_check_period_candles", "consecutive_losses_for_risk_reduction", "consecutive_wins_for_risk_increase", "max_trade_duration_bars_m5", "auto_trade_min_confidence", "tp1_max_points_xau_btc", "ai_decision_cooldown_seconds"]:
                        value = int(value_str)
                    else:
                        value = float(value_str)
                    
                    if key in CONFIG or (key.split('.')[0] == 'strategy_params' and key.split('.')[1] in CONFIG['strategy_params']):
                        config_update_request = {"config_key": key, "config_value": value}
                        ai_response_message_prefix = f"Capito, vuoi che modifichi il parametro '{key}'. Vediamo se è una buona idea... (scherzo, mi fido di te!). Ti suggerisco questo valore:"
                        ai_action_override = "UPDATE_CONFIG"
                    else:
                        ai_response_message_prefix = f"Mmmh, non sono sicuro di quale parametro '{key}' tu stia parlando. Potresti essere più specifico? Il mio database di configurazione è molto preciso!"
                except ValueError:
                    ai_response_message_prefix = "Non sono riuscito a capire il valore che vuoi impostare. Assicurati che sia un numero valido, per favore. Non sono un mago, sai!"
            else:
                ai_response_message_prefix = "Interessante, vuoi modificare un parametro. Quale e a che valore? Sii preciso, i dettagli fanno la differenza (e i profitti!)."
        else:
            ai_response_message_prefix = "Interessante, lascia che ci rifletta un attimo. Cosa ti piacerebbe approfondire? Sono qui per te (e per i mercati!)."

        response = self.ai_brain.get_strategic_analysis(context)
        
        # Prepend the AI's conversational prefix
        response['analysis_text'] = ai_response_message_prefix + " " + response['analysis_text']
        
        # Override action if a specific user intent was detected
        if ai_action_override:
            response['action'] = ai_action_override
            # If it's a config update request, add the config details to the response
            if ai_action_override == "UPDATE_CONFIG" and config_update_request:
                response['config_key'] = config_update_request['config_key']
                response['config_value'] = config_update_request['config_value']

        return response

    def shutdown(self):
        """Esegue lo shutdown controllato del bot e delle sue componenti."""
        logger.info("Avvio shutdown di FusionAI Trader...")
        if self.market_monitor: self.market_monitor.stop()
        if self.opportunity_scout: self.opportunity_scout.stop()
        if self.position_manager: self.position_manager.stop()

        if self.market_monitor and self.market_monitor.is_alive(): self.market_monitor.join(5)
        if self.opportunity_scout and self.opportunity_scout.is_alive(): self.opportunity_scout.join(5)
        if self.position_manager and self.position_manager.is_alive(): self.position_manager.join(5)

        # Salva la performance delle strategie prima dello shutdown
        self.strategy_engine._save_strategy_performance()

        if mt5.terminal_info() is not None:
            mt5.shutdown()
            logger.info("Disconnesso da MetaTrader 5.")
        logger.info("FusionAI Trader spento.")

# --- APPLICAZIONE FLASK ---
app = Flask(__name__)
CORS(app)
bot = FusionAITrader()

@app.route('/login_mt5', methods=['POST'])
def login_mt5_api():
    """API per il login a MetaTrader 5."""
    data = request.json
    login = data.get('login')
    password = data.get('password')
    server = data.get('server')

    if not all([login, password, server]):
        return jsonify({"success": False, "message": "Credenziali MT5 mancanti."}), 400

    result = bot.initialize_mt5(login, password, server)
    if result:
        return jsonify({"success": True, "message": "MT5 connesso e servizi avviati."})
    else:
        return jsonify({"success": False, "message": f"Login MT5 fallito: {mt5.last_error()}"}), 500

@app.route('/configure_ai', methods=['POST'])
def configure_ai_api():
    """API per configurare il cervello AI."""
    data = request.json
    gemini_api_key = data.get('gemini_api_key')
    openai_api_key = data.get('openai_api_key')

    if bot.initialize_ai(gemini_api_key, openai_api_key):
        return jsonify({"success": True, "message": "AI Brain configurato."})
    return jsonify({"success": False, "message": "Configurazione AI fallita."}), 500

@app.route('/chat_with_fusionai', methods=['POST'])
def chat_with_fusionai():
    """API per interagire con l'AI tramite chat."""
    data = request.json
    user_message = data.get('message', 'Analisi generale del mercato.')
    view_context = data.get('view_context', {})
    
    symbol_from_context = view_context.get('symbol') if view_context.get('symbol') != "" else CONFIG['symbols'][0]
    timeframe_from_context = view_context.get('timeframe') if view_context.get('timeframe') != "" else "M5"
    
    view_context['symbol'] = symbol_from_context
    view_context['timeframe'] = timeframe_from_context

    if not bot.ai_brain:
        return jsonify({"analysis_text": "AI non configurata. Per favorere, inizializza il mio cervello con le API keys!", "action": "ERROR", "symbol": view_context['symbol'], "confidence_score": 0}), 400

    response = bot.handle_user_query(user_message, view_context)
    
    # Se l'AI ha raccomandato un UPDATE_CONFIG, esegui l'aggiornamento
    if response.get('action') == "UPDATE_CONFIG" and response.get('config_key') and response.get('config_value') is not None:
        key = response['config_key']
        value = response['config_value']
        
        # Prepara un payload per la funzione update_config
        update_payload = {key: value}
        
        # Chiamata interna alla funzione update_config
        update_success_response = update_config_internal(update_payload)
        
        if update_success_response.json['success']:
            response['analysis_text'] += f" Parametro '{key}' aggiornato a '{value}'. Fatto! Ora sono ancora più affinato."
        else:
            response['analysis_text'] += f" Ops, non sono riuscito ad aggiornare il parametro '{key}'. Errore: {update_success_response.json['message']}"
        
        # Rimuovi i campi specifici di config_key/value dalla risposta finale all'utente
        response.pop('config_key', None)
        response.pop('config_value', None)
        response['action'] = "INFO" # Cambia l'azione a INFO per non scatenare azioni di trading

    return jsonify(response)

@app.route('/get_notifications', methods=['GET'])
def get_notifications():
    """API per recuperare le notifiche recenti."""
    with bot.lock: return jsonify({"notifications": bot.notifications[:] if bot.notifications else []})

@app.route('/status', methods=['GET'])
def get_status():
    """API per ottenere lo stato attuale del bot e del conto."""
    with bot.lock:
        account_info_obj = mt5.account_info()
        account_status = account_info_obj._asdict() if account_info_obj else None
        
        if account_status:
            initial_equity_val = bot.initial_equity if isinstance(bot.initial_equity, (int, float)) else 0
            current_equity_val = account_info_obj.equity if isinstance(account_info_obj.equity, (int, float)) else 0

            if initial_equity_val > 0:
                account_status['current_daily_drawdown_percentage'] = ((initial_equity_val - current_equity_val) / initial_equity_val) * 100
                account_status['current_daily_profit_percentage'] = ((current_equity_val - initial_equity_val) / initial_equity_val) * 100
            else:
                account_status['current_daily_drawdown_percentage'] = 0
                account_status['current_daily_profit_percentage'] = 0


            weekly_drawdown, weekly_profit = bot.calculate_weekly_performance()
            account_status['current_weekly_drawdown_percentage'] = weekly_drawdown
            account_status['current_weekly_profit_percentage'] = weekly_profit
            account_status['max_weekly_drawdown_limit'] = CONFIG['max_weekly_drawdown_percentage']
            account_status['weekly_profit_target_limit'] = CONFIG['weekly_profit_target_percentage']
            account_status['daily_profit_target_limit'] = CONFIG['daily_profit_target_percentage']
            account_status['max_daily_drawdown_limit'] = CONFIG['max_daily_drawdown_percentage']


        return jsonify({
            "mt5_connected": mt5.terminal_info() is not None,
            "account_info": account_status,
            "open_positions": bot.open_positions[:] if bot.open_positions else [],
            "ai_initialized": bot.ai_brain is not None,
            "threads_alive": {
                "monitor": bot.market_monitor.is_alive() if bot.market_monitor else False,
                "scout": bot.opportunity_scout.is_alive() if bot.opportunity_scout else False,
                "position_manager": bot.position_manager.is_alive() if bot.position_manager else False,
            },
            "configuration": CONFIG,
            "latest_sentiment": {s: bot.market_sentiment.get_sentiment(s) for s in CONFIG['symbols']}
        })

@app.route('/get_chart_data/<symbol>/<timeframe>', methods=['GET'])
def get_chart_data(symbol, timeframe):
    """API per recuperare i dati del grafico con indicatori e overlay."""
    if symbol not in CONFIG['symbols'] or timeframe not in CONFIG['timeframes']:
        return jsonify({"success": False, "message": "Simbolo o timeframe non validi."}), 400

    df = bot.digital_twin.get_chart_data(symbol, timeframe)
    if df.empty:
        return jsonify({"success": True, "data": [], "overlays": {"support_resistance": [], "pivot_points": {}}})

    data_to_send = []
    
    # Include il calcolo di Supporto/Resistenza e Punti Pivot
    support_resistance_levels = bot.digital_twin._calculate_support_resistance(df, symbol)
    
    pivot_points = {}
    if not df.empty and len(df) > 1:
        # Per i pivot points, è più accurato usare i dati del giorno precedente (D1)
        df_d1 = bot.digital_twin.get_chart_data(symbol, "D1")
        if not df_d1.empty and len(df_d1) > 1:
            last_day_candle = df_d1.iloc[-1]
            last_day_close = last_day_candle['close']
            last_day_high = last_day_candle['high']
            last_day_low = last_day_candle['low']
            
            pivot_classic = (last_day_high + last_day_low + last_day_close) / 3
            r1_classic = (2 * pivot_classic) - last_day_low
            s1_classic = (2 * pivot_classic) - last_day_high
            r2_classic = pivot_classic + (last_day_high - last_day_low)
            s2_classic = pivot_classic - (last_day_high - last_day_low)
            
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info else 5

            pivot_points = {
                "P": round(pivot_classic, digits),
                "R1": round(r1_classic, digits),
                "S1": round(s1_classic, digits),
                "R2": round(r2_classic, digits),
                "S2": round(s2_classic, digits)
            }
            # Rimuovi eventuali NaN dai Punti Pivot
            pivot_points = {k:v for k,v in pivot_points.items() if not np.isnan(v)}
        
    for index, row in df.iterrows():
        record = row.to_dict()
        record['time'] = index.timestamp()
        data_to_send.append(record)

    response_data = {
        "success": True,
        "data": data_to_send,
        "overlays": {
            "support_resistance": support_resistance_levels,
            "pivot_points": pivot_points
        }
    }

    return jsonify(response_data)

@app.route('/get_trade_journal', methods=['GET'])
def get_trade_journal():
    """API per recuperare il journal delle operazioni."""
    return jsonify({"trades": bot.trade_journal.get_trades()})

@app.route('/toggle_autotrade', methods=['POST'])
def toggle_autotrade():
    """API per abilitare/disabilitare l'auto-trading."""
    data = request.json
    enable = data.get('enable')
    if isinstance(enable, bool):
        CONFIG['auto_trade_enabled'] = enable
        logger.info(f"Auto-trading configurato a: {CONFIG['auto_trade_enabled']}")
        return jsonify({"success": True, "auto_trade_enabled": CONFIG['auto_trade_enabled'], "message": f"Auto-trading {'abilitato' if enable else 'disabilitato'}."})
    return jsonify({"success": False, "message": "Valore 'enable' non valido."}), 400

# Funzione interna per l'aggiornamento della configurazione, riutilizzabile dalla chat
def update_config_internal(data):
    """Funzione interna per aggiornare la configurazione."""
    updated_keys = []
    for key, value in data.items():
        if key in CONFIG:
            if key in ["atr_sl_multiplier", "risk_reward_ratio", "slippage_points", "max_open_trades_per_symbol", "max_lot_xau_btc", "max_lot_other_symbols", "max_spread_points_allowed", "volatility_check_atr_multiplier", "volatility_check_period_candles", "consecutive_losses_for_risk_reduction", "consecutive_wins_for_risk_increase", "max_trade_duration_bars_m5", "auto_trade_min_confidence", "tp1_max_points_xau_btc", "ai_decision_cooldown_seconds"]:
                if isinstance(value, str) and value.strip() == '':
                    logger.warning(f"Tentativo di aggiornare {key} con stringa vuota. Ignorato.")
                    continue

                try:
                    if key in ["slippage_points", "max_open_trades_per_symbol", "volatility_check_period_candles", "consecutive_losses_for_risk_reduction", "consecutive_wins_for_risk_increase", "max_trade_duration_bars_m5", "auto_trade_min_confidence", "tp1_max_points_xau_btc", "ai_decision_cooldown_seconds"]:
                        value = int(value)
                    else:
                        value = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Tentativo di aggiornare {key} con tipo non corrispondente ('{type(value).__name__}'). Ignorato.")
                    continue

            elif key == "strategy_params" and isinstance(value, dict):
                for skey, svalue in value.items():
                    if skey in CONFIG['strategy_params'] and isinstance(svalue, type(CONFIG['strategy_params'][skey])):
                        CONFIG['strategy_params'][skey] = svalue
                        updated_keys.append(f"strategy_params.{skey}")
                    else:
                        logger.warning(f"Tentativo di aggiornare strategy_params.{skey} con tipo non corrispondente o chiave non valida. Ignorato.")
            elif key == "tp_levels_multiplier" and isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    CONFIG[key] = value
                    updated_keys.append(key)
                    logger.info(f"Configurazione aggiornata: {key} = {value}")
                else:
                    logger.warning(f"Tentativo di aggiornare {key} con lista non numerica. Ignorato.")
            elif key == "xauusd_btc_fixed_tps" and isinstance(value, dict): # Gestione TP fissi
                for sym_key, tp_data in value.items():
                    if sym_key in CONFIG['xauusd_btc_fixed_tps'] and isinstance(tp_data, dict):
                        for trade_type_key, prices in tp_data.items():
                            if trade_type_key in ["BUY", "SELL"] and isinstance(prices, dict):
                                for tp_level, price_val in prices.items():
                                    if tp_level in ["TP1", "TP2", "TP3"] and isinstance(price_val, (int, float)):
                                        CONFIG['xauusd_btc_fixed_tps'][sym_key][trade_type_key][tp_level] = price_val
                                        updated_keys.append(f"xauusd_btc_fixed_tps.{sym_key}.{trade_type_key}.{tp_level}")
                                    else:
                                        logger.warning(f"Tentativo di aggiornare xauusd_btc_fixed_tps.{sym_key}.{trade_type_key}.{tp_level} con valore non valido. Ignorato.")
                            else:
                                logger.warning(f"Tentativo di aggiornare xauusd_btc_fixed_tps.{sym_key}.{trade_type_key} con tipo non valido. Ignorato.")
                    else:
                        logger.warning(f"Tentativo di aggiornare xauusd_btc_fixed_tps.{sym_key} con tipo non valido o chiave non esistente. Ignorato.")
            elif isinstance(CONFIG[key], type(value)):
                CONFIG[key] = value
                updated_keys.append(key)
                logger.info(f"Configurazione aggiornata: {key} = {value}")
            else:
                logger.warning(f"Tentativo di aggiornare {key} con tipo non corrispondente. Ignorato.")
        else:
            logger.warning(f"Tentativo di aggiornare chiave non esistente: {key}. Ignorato.")

    if updated_keys:
        return jsonify({"success": True, "message": f"Configurazione aggiornata per le chiavi: {', '.join(updated_keys)}."})
    else:
        return jsonify({"success": False, "message": "Nessun parametro valido fornito per l'aggiornamento."}), 400

@app.route('/update_config', methods=['POST'])
def update_config_api():
    """API per aggiornare la configurazione del bot."""
    data = request.json
    return update_config_internal(data)


@app.route('/run_backtest', methods=['POST'])
def run_backtest_api():
    """API per avviare un backtest."""
    data = request.json
    symbol = data.get('symbol')
    timeframe = data.get('timeframe')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    initial_capital = data.get('initial_capital', 10000.0)
    strategies_to_test = data.get('strategies_to_test', [])

    if not all([symbol, timeframe, start_date_str, end_date_str, strategies_to_test]) or initial_capital is None or initial_capital <= 0:
        return jsonify({"success": False, "message": "Parametri backtest mancanti o non validi. Assicurati che capitale iniziale e strategie siano forniti correttamente."}), 400

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=italy_tz)
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=italy_tz)
    except ValueError:
        return jsonify({"success": False, "message": "Formato data non valido (YYYY-MM-DD)."}), 400

    backtest_id = f"backtest_{symbol}_{timeframe}_{start_date_str}_{end_date_str}_{'_'.join(sorted(strategies_to_test))}" # Ordina per ID consistente

    threading.Thread(target=lambda: bot.backtesting_engine.run_backtest(
        symbol, timeframe, start_date, end_date, initial_capital, strategies_to_test
    )).start()

    return jsonify({"success": True, "message": "Backtest avviato. Controlla /get_backtest_results/<id> per lo stato.", "backtest_id": backtest_id})

@app.route('/get_backtest_results/<backtest_id>', methods=['GET'])
def get_backtest_results_api(backtest_id):
    """API per recuperare i risultati di un backtest."""
    results = bot.backtesting_engine.get_backtest_results(backtest_id)
    if results:
        serializable_results = results.copy()
        if 'equity_curve' in serializable_results:
            serializable_results['equity_curve'] = [round(e, 2) for e in serializable_results['equity_curve']]
        if 'trade_log' in serializable_results:
            for trade in serializable_results['trade_log']:
                if isinstance(trade.get('time'), datetime):
                    trade['time'] = trade['time'].isoformat()
        return jsonify({"success": True, "results": serializable_results})
    return jsonify({"success": False, "message": "Backtest non trovato o non ancora completato."}), 404

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """API per spegnere il server Flask e il bot."""
    logger.info("Richiesta di shutdown ricevuta tramite API.")
    bot.shutdown()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Non in esecuzione con il server Werkzeug')
    func()
    return jsonify({"success": True, "message": "Server spent."})


if __name__ == '__main__':
    import signal
    def signal_handler(sig, frame):
        logger.info("Segnale di interruzione ricevuto. Avvio shutdown...")
        bot.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
