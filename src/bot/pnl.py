class Signal:
    def __init__(self, contract, action, quantity, price, datetime):
        if not contract.localSymbol:
            raise Exception('no localSymbol on contract, make sure to qualify')

        self.contract = contract
        self.action = action
        self.quantity = quantity
        self.price = price
        self.datetime = datetime

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '[Signal: %s %s, %s@%s %s]' % (self.contract.localSymbol, self.action, self.quantity, self.price, self.datetime)


class Position:
    pass


# Log signals as if they executed at desired price
class SignalLogger:
    def __init__(self):
        self.events = []
        self.pnl = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '[Logger: PnL=%s]' % self.pnl

    def get_quantity(self, contract):
        quantity = 0
        for event in self.events:
            signal = event['signal']
            if signal.contract.localSymbol == contract.localSymbol:
                quantity += signal.quantity * {
                    'buy': 1,
                    'sell': -1,
                }[signal.action]
        # if quantity:
        #     import pdb ; pdb.set_trace()
        return quantity

    def log_signal(self, signal):
        if not signal:
            return

        value = signal.quantity * signal.price
        if signal.action == 'buy':
            value = -value
        self.pnl += value

        self.events.append({
            'pnl': self.pnl,
            'signal': signal
        })

    def print_summary(self):
        for event in self.events:
            print('%.2f\t%s' % (event['pnl'], event['signal']))
