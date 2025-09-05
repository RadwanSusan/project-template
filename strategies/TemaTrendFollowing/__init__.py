from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils

class TemaTrendFollowing(Strategy):
    def hyperparameters(self):
        return [
            # TEMA periods
            {'name': 'tema_short_period', 'type': int, 'min': 5, 'max': 20, 'default': 10},
            {'name': 'tema_medium_period', 'type': int, 'min': 50, 'max': 120, 'default': 80},
            {'name': 'tema_long_4h_short', 'type': int, 'min': 10, 'max': 40, 'default': 20},
            {'name': 'tema_long_4h_long', 'type': int, 'min': 50, 'max': 100, 'default': 70},
            
            # Indicator thresholds
            {'name': 'adx_threshold', 'type': int, 'min': 20, 'max': 60, 'default': 40},
            {'name': 'cmo_upper_threshold', 'type': int, 'min': 20, 'max': 60, 'default': 40},
            {'name': 'cmo_lower_threshold', 'type': int, 'min': -60, 'max': -20, 'default': -40},
            
            # ATR multipliers
            {'name': 'entry_atr_offset', 'type': float, 'min': 0.5, 'max': 2.0, 'default': 1.0, 'step': 0.1},
            {'name': 'stop_loss_atr_mult', 'type': float, 'min': 2.0, 'max': 6.0, 'default': 4.0, 'step': 0.5},
            {'name': 'take_profit_atr_mult', 'type': float, 'min': 1.5, 'max': 5.0, 'default': 3.0, 'step': 0.5},
            
            # Risk management
            {'name': 'risk_percentage', 'type': float, 'min': 1.0, 'max': 5.0, 'default': 3.0, 'step': 0.5},
            {'name': 'position_multiplier', 'type': int, 'min': 1, 'max': 5, 'default': 3},
        ]
    
    @property
    def candles_4h(self):
        """Cache 4h candles to avoid multiple API calls"""
        if not hasattr(self, '_candles_4h'):
            self._candles_4h = self.get_candles(self.exchange, self.symbol, '4h')
        return self._candles_4h
    
    @property
    def short_term_trend(self):
        tema_short = ta.tema(self.candles, self.hp['tema_short_period'])
        tema_medium = ta.tema(self.candles, self.hp['tema_medium_period'])
        return 1 if tema_short > tema_medium else -1
    
    @property
    def long_term_trend(self):
        tema_short_4h = ta.tema(self.candles_4h, self.hp['tema_long_4h_short'])
        tema_long_4h = ta.tema(self.candles_4h, self.hp['tema_long_4h_long'])
        return 1 if tema_short_4h > tema_long_4h else -1
    
    @property
    def tema_short(self):
        return ta.tema(self.candles, self.hp['tema_short_period'])
    
    @property
    def tema_medium(self):
        return ta.tema(self.candles, self.hp['tema_medium_period'])
    
    @property
    def tema_short_4h(self):
        return ta.tema(self.candles_4h, self.hp['tema_long_4h_short'])
    
    @property
    def tema_long_4h(self):
        return ta.tema(self.candles_4h, self.hp['tema_long_4h_long'])
    
    @property
    def atr(self):
        return ta.atr(self.candles)
    
    @property
    def adx(self):
        return ta.adx(self.candles)
    
    @property
    def cmo(self):
        return ta.cmo(self.candles)
    
    def should_long(self) -> bool:
        return (
            self.short_term_trend == 1 and 
            self.long_term_trend == 1 and 
            self.adx > self.hp['adx_threshold'] and 
            self.cmo > self.hp['cmo_upper_threshold']
        )
    
    def should_short(self) -> bool:
        return (
            self.short_term_trend == -1 and 
            self.long_term_trend == -1 and 
            self.adx > self.hp['adx_threshold'] and 
            self.cmo < self.hp['cmo_lower_threshold']
        )
    
    def go_long(self):
        entry_price = self.price - (self.atr * self.hp['entry_atr_offset'])
        stop_loss_price = entry_price - (self.atr * self.hp['stop_loss_atr_mult'])
        
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.hp['risk_percentage'], 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        self.buy = qty * self.hp['position_multiplier'], entry_price
    
    def go_short(self):
        entry_price = self.price + (self.atr * self.hp['entry_atr_offset'])
        stop_loss_price = entry_price + (self.atr * self.hp['stop_loss_atr_mult'])
        
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.hp['risk_percentage'], 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        self.sell = qty * self.hp['position_multiplier'], entry_price
    
    def should_cancel_entry(self) -> bool:
        return True
    
    def on_open_position(self, order) -> None:
        atr_stop = self.atr * self.hp['stop_loss_atr_mult']
        atr_tp = self.atr * self.hp['take_profit_atr_mult']
        
        if self.is_long:
            self.stop_loss = self.position.qty, self.position.entry_price - atr_stop
            self.take_profit = self.position.qty, self.position.entry_price + atr_tp
        elif self.is_short:
            self.stop_loss = self.position.qty, self.position.entry_price + atr_stop
            self.take_profit = self.position.qty, self.position.entry_price - atr_tp