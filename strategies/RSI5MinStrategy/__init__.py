from jesse.strategies import Strategy, cached
from jesse import utils
import jesse.indicators as ta
import numpy as np


class RSI5MinStrategy(Strategy):
    """
    Sophisticated 5-minute RSI trading strategy with comprehensive risk management.
    
    This strategy uses RSI with multiple confirmations:
    - RSI overbought/oversold levels
    - RSI momentum confirmation
    - Trend filtering with EMAs
    - Volume filtering (optional)
    - Higher timeframe alignment
    - ATR-based risk management
    """

    def hyperparameters(self):
        """
        15 optimizable hyperparameters for comprehensive strategy optimization
        """
        return [
            # RSI Configuration
            {'name': 'rsi_period', 'type': int, 'min': 10, 'max': 20, 'default': 14},
            {'name': 'rsi_overbought', 'type': int, 'min': 65, 'max': 85, 'default': 70},
            {'name': 'rsi_oversold', 'type': int, 'min': 15, 'max': 35, 'default': 30},
            
            # Entry Confirmation
            {'name': 'rsi_momentum_period', 'type': int, 'min': 2, 'max': 5, 'default': 3},
            {'name': 'entry_confirmation_candles', 'type': int, 'min': 1, 'max': 3, 'default': 2},
            
            # Trend Filter
            {'name': 'trend_ema_fast', 'type': int, 'min': 20, 'max': 50, 'default': 34},
            {'name': 'trend_ema_slow', 'type': int, 'min': 50, 'max': 200, 'default': 100},
            {'name': 'use_trend_filter', 'type': 'categorical', 'options': [True, False], 'default': True},
            {'name': 'higher_tf_period', 'type': int, 'min': 50, 'max': 200, 'default': 100},
            
            # Risk Management
            {'name': 'risk_percentage', 'type': float, 'min': 0.5, 'max': 3.0, 'default': 1.0},
            {'name': 'atr_period', 'type': int, 'min': 10, 'max': 20, 'default': 14},
            {'name': 'stop_loss_atr_mult', 'type': float, 'min': 1.5, 'max': 4.0, 'default': 2.5},
            {'name': 'take_profit_atr_mult', 'type': float, 'min': 2.0, 'max': 6.0, 'default': 3.5},
            
            # Advanced Features
            {'name': 'use_volume_filter', 'type': 'categorical', 'options': [True, False], 'default': False},
            {'name': 'max_hold_candles', 'type': int, 'min': 20, 'max': 100, 'default': 50}
        ]

    # ========== INDICATOR PROPERTIES ==========
    
    @property
    @cached
    def rsi(self):
        """Current RSI value"""
        return ta.rsi(self.candles, period=self.hp['rsi_period'])
    
    @property
    @cached
    def rsi_seq(self):
        """RSI sequence for momentum analysis"""
        return ta.rsi(self.candles, period=self.hp['rsi_period'], sequential=True)
    
    @property
    @cached
    def ema_fast(self):
        """Fast EMA for trend detection"""
        return ta.ema(self.candles, period=self.hp['trend_ema_fast'])
    
    @property
    @cached
    def ema_slow(self):
        """Slow EMA for trend detection"""
        return ta.ema(self.candles, period=self.hp['trend_ema_slow'])
    
    @property
    @cached
    def atr(self):
        """Average True Range for volatility-based stops"""
        return ta.atr(self.candles, period=self.hp['atr_period'])
    
    @property
    @cached
    def volume_sma(self):
        """Volume moving average for volume filter"""
        return ta.sma(self.candles, period=20, source_type="volume")
    
    @property
    @cached
    def higher_tf_ema(self):
        """Higher timeframe EMA for trend alignment"""
        # Use 15min candles for higher timeframe analysis on 5min strategy
        try:
            htf_candles = self.get_candles(self.exchange, self.symbol, '15m')
            return ta.ema(htf_candles, period=self.hp['higher_tf_period'])
        except:
            # Fallback to current timeframe if higher timeframe not available
            return ta.ema(self.candles, period=self.hp['higher_tf_period'])

    # ========== HELPER METHODS ==========
    
    def rsi_momentum_bullish(self):
        """Check if RSI has bullish momentum over recent candles"""
        if len(self.rsi_seq) < self.hp['rsi_momentum_period'] + 1:
            return False
        
        recent_rsi = self.rsi_seq[-self.hp['rsi_momentum_period']-1:]
        # Check if RSI is generally increasing over the momentum period
        momentum_count = 0
        for i in range(1, len(recent_rsi)):
            if recent_rsi[i] > recent_rsi[i-1]:
                momentum_count += 1
        
        # Require majority of recent candles to show upward RSI momentum
        return momentum_count >= (len(recent_rsi) - 1) // 2
    
    def rsi_momentum_bearish(self):
        """Check if RSI has bearish momentum over recent candles"""
        if len(self.rsi_seq) < self.hp['rsi_momentum_period'] + 1:
            return False
        
        recent_rsi = self.rsi_seq[-self.hp['rsi_momentum_period']-1:]
        # Check if RSI is generally decreasing over the momentum period
        momentum_count = 0
        for i in range(1, len(recent_rsi)):
            if recent_rsi[i] < recent_rsi[i-1]:
                momentum_count += 1
        
        # Require majority of recent candles to show downward RSI momentum
        return momentum_count >= (len(recent_rsi) - 1) // 2
    
    @property
    def trend_direction(self):
        """Determine overall trend direction: 1=bullish, -1=bearish, 0=neutral"""
        if not self.hp['use_trend_filter']:
            return 0  # No trend filter
        
        # Primary trend from fast/slow EMA
        primary_trend = 1 if self.ema_fast > self.ema_slow else -1
        
        # Higher timeframe confirmation
        htf_trend = 1 if self.close > self.higher_tf_ema else -1
        
        # Both timeframes must align for strong trend signal
        if primary_trend == htf_trend:
            return primary_trend
        
        return 0  # Conflicting signals = neutral
    
    @property
    def volume_confirmation(self):
        """Check if current volume is above average (bullish confirmation)"""
        if not self.hp['use_volume_filter']:
            return True  # Volume filter disabled
        
        return self.volume > self.volume_sma * 1.2  # 20% above average
    
    def get_position_entry_candle(self):
        """Get the candle index when position was opened"""
        if hasattr(self, '_entry_candle_index'):
            return self._entry_candle_index
        return None

    # ========== ENTRY LOGIC ==========
    
    def should_long(self) -> bool:
        """
        Comprehensive long entry conditions:
        1. RSI oversold
        2. RSI showing bullish momentum
        3. Trend alignment (if enabled)
        4. Volume confirmation (if enabled)
        """
        # Core RSI condition - must be oversold
        if self.rsi >= self.hp['rsi_oversold']:
            return False
        
        # RSI momentum confirmation - RSI must be turning up
        if not self.rsi_momentum_bullish():
            return False
        
        # Trend filter - only long when trend is bullish or neutral
        if self.hp['use_trend_filter'] and self.trend_direction == -1:
            return False
        
        # Volume confirmation (if enabled)
        if not self.volume_confirmation:
            return False
        
        # Additional confirmation: price should be near or above fast EMA for bullish bias
        if self.hp['use_trend_filter'] and self.close < self.ema_fast * 0.998:
            return False
        
        return True
    
    def should_short(self) -> bool:
        """
        Comprehensive short entry conditions:
        1. RSI overbought
        2. RSI showing bearish momentum
        3. Trend alignment (if enabled)
        4. Volume confirmation (if enabled)
        """
        # Core RSI condition - must be overbought
        if self.rsi <= self.hp['rsi_overbought']:
            return False
        
        # RSI momentum confirmation - RSI must be turning down
        if not self.rsi_momentum_bearish():
            return False
        
        # Trend filter - only short when trend is bearish or neutral
        if self.hp['use_trend_filter'] and self.trend_direction == 1:
            return False
        
        # Volume confirmation (if enabled)
        if not self.volume_confirmation:
            return False
        
        # Additional confirmation: price should be near or below fast EMA for bearish bias
        if self.hp['use_trend_filter'] and self.close > self.ema_fast * 1.002:
            return False
        
        return True

    # ========== POSITION MANAGEMENT ==========
    
    def go_long(self):
        """Execute long position with proper risk management"""
        # Calculate entry price (market order)
        entry_price = self.close
        
        # Calculate stop loss based on ATR
        stop_loss_price = entry_price - (self.atr * self.hp['stop_loss_atr_mult'])
        
        # Calculate position size based on risk percentage
        qty = utils.risk_to_qty(
            self.balance,
            self.hp['risk_percentage'],
            entry_price,
            stop_loss_price,
            fee_rate=self.fee_rate
        )
        
        # Execute market buy order
        self.buy = qty, entry_price
        
        # Store entry candle for time-based exit
        self._entry_candle_index = self.index
    
    def go_short(self):
        """Execute short position with proper risk management"""
        # Calculate entry price (market order)
        entry_price = self.close
        
        # Calculate stop loss based on ATR
        stop_loss_price = entry_price + (self.atr * self.hp['stop_loss_atr_mult'])
        
        # Calculate position size based on risk percentage
        qty = utils.risk_to_qty(
            self.balance,
            self.hp['risk_percentage'],
            entry_price,
            stop_loss_price,
            fee_rate=self.fee_rate
        )
        
        # Execute market sell order
        self.sell = qty, entry_price
        
        # Store entry candle for time-based exit
        self._entry_candle_index = self.index
    
    def should_cancel_entry(self) -> bool:
        """Cancel entry orders if conditions change"""
        return True
    
    # ========== POSITION UPDATES ==========
    
    def on_open_position(self, order) -> None:
        """Set stop loss and take profit when position opens"""
        entry_price = self.position.entry_price
        
        if self.is_long:
            # Long position stops
            stop_loss_price = entry_price - (self.atr * self.hp['stop_loss_atr_mult'])
            take_profit_price = entry_price + (self.atr * self.hp['take_profit_atr_mult'])
        else:
            # Short position stops
            stop_loss_price = entry_price + (self.atr * self.hp['stop_loss_atr_mult'])
            take_profit_price = entry_price - (self.atr * self.hp['take_profit_atr_mult'])
        
        # Set stop loss and take profit
        self.stop_loss = self.position.qty, stop_loss_price
        self.take_profit = self.position.qty, take_profit_price
    
    def update_position(self) -> None:
        """Manage open positions with dynamic exits"""
        if not self.position.is_open:
            return
        
        # Time-based exit - close position if held too long
        entry_candle = self.get_position_entry_candle()
        if entry_candle is not None:
            candles_held = self.index - entry_candle
            if candles_held >= self.hp['max_hold_candles']:
                self.liquidate()
                return
        
        # RSI-based exits
        if self.is_long:
            # Exit long when RSI reaches overbought or shows strong bearish momentum
            if (self.rsi >= self.hp['rsi_overbought'] or 
                (self.rsi > 60 and self.rsi_momentum_bearish())):
                self.liquidate()
                return
        
        elif self.is_short:
            # Exit short when RSI reaches oversold or shows strong bullish momentum
            if (self.rsi <= self.hp['rsi_oversold'] or 
                (self.rsi < 40 and self.rsi_momentum_bullish())):
                self.liquidate()
                return
        
        # Trend reversal exit
        if self.hp['use_trend_filter']:
            if self.is_long and self.trend_direction == -1:
                # Strong bearish trend developed, exit long
                if self.close < self.ema_fast:
                    self.liquidate()
                    return
            
            elif self.is_short and self.trend_direction == 1:
                # Strong bullish trend developed, exit short
                if self.close > self.ema_fast:
                    self.liquidate()
                    return
    
    # ========== OPTIONAL: WATCHLIST FOR MONITORING ==========
    
    def watch_list(self):
        """Return watchlist for live trading monitoring"""
        return [
            ('RSI', round(self.rsi, 2)),
            ('Trend', self.trend_direction),
            ('ATR', round(self.atr, 4)),
            ('Entry Signal Long', self.should_long()),
            ('Entry Signal Short', self.should_short()),
        ]