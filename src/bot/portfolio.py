from collections import defaultdict

from bot.pnl import SignalLogger, Signal


class Portfolio:
    subscribers = []

    def __init__(self, ib):
        self.ib = ib
        self.logger = SignalLogger()
        self.positions = defaultdict(int)

    @property
    def pnl(self):
        return self.logger.pnl

    def get_position(self, contract):
        return self.positions[contract.conId]

    def subscribe(self, obj):
        self.subscribers.append(obj)

    def notify_subscribers(self):
        for subscriber in self.subscribers:
            subscriber.on_portfolio_update()

    def execute(self, signal):
        # print('execute', signal)
        if not signal:
            return

        self.positions[signal.contract.conId] += {'buy': 1, 'sell': -1}[signal.action] * signal.quantity
        self.logger.log_signal(signal)
        self.notify_subscribers()

    def close_positions(self, providers=[]):
        for conId, position in self.positions.items():
            if not position:
                continue

            contract_provider = None
            for provider in providers:
                if provider.contract.conId == conId:
                    contract_provider = provider
            if not contract_provider:
                raise Exception('data source missing')

            last_price = contract_provider.last().close
            if position > 0:
                action = 'sell'
            elif position < 0:
                action = 'buy'
            signal = Signal(contract_provider.contract, action, abs(position), last_price)
            print('close with signal', signal)
            self.execute(signal)
