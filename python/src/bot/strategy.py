import pprint

from bot.analyzer import TechnicalAnalyzer, TrendAnalyzer


pp = pprint.PrettyPrinter()


class ToggleStrategy:
    def __init__(self, provider):
        self.provider = provider
        self.next_action = 'buy'

    def signal(self):
        contract = self.provider.contract
        price = self.provider.trailing(1)[0].close
        action = self.next_action

        if self.next_action == 'buy':
            self.next_action = 'sell'
        else:
            self.next_action = 'buy'

        return Signal(contract, action, 1, price)


class EmaCrossoverStrategy:
    def __init__(self, provider):
        self.provider = provider
        self.analyzer = TechnicalAnalyzer(provider)

    def signal(self):
        contract = self.provider.contract
        analysis = self.analyzer.analysis()
        cross_signal = analysis['cross_signal']
        if not cross_signal:
            return None
        action = {'buy': 'buy', 'sell': 'sell'}[cross_signal]
        price = self.provider.trailing(1)[0].close
        return Signal(contract, action, 1, price)


class TrendStrategy:
    def __init__(self, provider, portfolio, max_position=10):
        self.provider = provider
        self.portfolio = portfolio
        self.technical_analyzer = TechnicalAnalyzer(provider)
        self.trend_analyzer = TrendAnalyzer(provider)
        self.max_position = max_position
        self.analyzers = [
            self.technical_analyzer,
            self.trend_analyzer,
        ]

        self.provider.subscribe(self)
        self.portfolio.subscribe(self)

        self.states = {}

    @property
    def contract(self):
        return self.provider.contract

    def on_provider_update(self):
        technical_info = self.technical_analyzer.analysis()
        trend_info = self.trend_analyzer.analysis()

        print('trend_info', trend_info)

        price = self.provider.trailing(1)[0].close

        obv_state = None
        if trend_info['obv_slope_percent']:
            if trend_info['obv_slope_percent'] > .1:
                obv_state = 'strong_up_trend'
            elif trend_info['obv_slope_percent'] > .05:
                obv_state = 'weak_up_trend'
            elif trend_info['obv_slope_percent'] < .1:
                obv_state = 'strong_down_trend'
            elif trend_info['obv_slope_percent'] < .05:
                obv_state = 'weak_down_trend'
            else:
                obv_state = 'flat'

        self.states['obv_trend'] = obv_state
        self.states['obv_slope_percent'] = trend_info['obv_slope_percent']

    def on_portfolio_update(self):
        position = self.portfolio.get_position(self.contract)

        if position > 10:
            position_state = 'long'
        elif position > 0:
            position_state = 'long'
        elif position < 10:
            position_state = 'short'
        elif position < 0:
            position_state = 'short'
        else:
            position_state = 'none'

        self.states['position'] = position_state

    def signal(self):
        contract = self.contract
        position = self.portfolio.get_position(self.contract)
        price = self.provider.trailing(1)[0].close

        print('get signal based on state: %s' % self.states)

        position = self.portfolio.get_position(self.contract)

        if position > 0 and self.states.get('obv_trend') in ['strong_down_trend', 'weak_down_trend', 'flat']:
            return Signal(contract, 'sell', position, price)
        if position < self.max_position and self.states.get('obv_trend') in ['weak_up_trend']:
            return Signal(contract, 'buy', int(self.max_position / 5), price)
        if position < self.max_position and self.states.get('obv_trend') in ['strong_up_trend']:
            return Signal(contract, 'buy', self.max_position, price)

        return None
