import asyncio
import math
import time
import pprint
import ib_insync as ibs

from collections import defaultdict

from bot.analyzer import (
    BuyLineAnalyzer,
    CurrentBarAnalyzer,
    GroupedAnalyzer,
    ObvSpikeAnalyzer,
    TechnicalAnalyzer,
    TrendAnalyzer,

    ObvStateAnalyzer,
    AdxStateAnalyzer,
    DiPlusStateAnalyzer,
    DiMinusStateAnalyzer,

    EmaPeakAnalyzer,

    get_trailing_obv,
    get_full_analayzer_group,
)
from bot.contracts import suggest_options
from bot.provider import BacktestDoneException, BacktestTickProvider, LiveTickProvider
from bot import ranges
from bot import util
from bot import orders
from bot import pnl

pp = pprint.PrettyPrinter()


class StateChangeListener:
    def __init__(self, symbol, state_name, from_value, to_value, callback):
        self.symbol = symbol
        self.state_name = state_name
        self.from_value = from_value
        self.to_value = to_value
        self.callback = callback

    def eval(self, symbol, state_name, from_value, to_value):
        if self.symbol not in  ['*', symbol]:
            return
        if self.state_name not in  ['*', state_name]:
            return
        if self.from_value not in ['*', from_value]:
            return
        if self.to_value not in ['*', to_value]:
            return
        self.callback(symbol)


class StateMachine:
    def __init__(self, listeners):
        self.last_infos = None
        self.listeners = listeners
        self.memory = []
        self.memory_seconds = 5 * 60
        self.bull_indicators = ['obv', 'di+']
        self.bear_indicators = ['di-']
        self.trend_indicators = ['adx']

    def transition(self, infos, dt):
        if self.last_infos:
            self._transition_internal(infos, dt)
        self.last_infos = infos

        print('State transition memory')
        for item in self.memory:
            delta = dt - item['datetime']
            print('- [%s] %s %s->%s, %s seconds ago' % (
                item['symbol'], item['indicator'], item['last_q'], item['q'], delta.seconds))

        # sell, buy, trend = self.signals(dt)
        # import pdb ; pdb.set_trace()

    def _transition_internal(self, infos, dt):
        for key, value in infos.items():
            s = key.split('.')
            # print(key, s)
            if len(s) > 1:
                symbol = s[0]
                key_part = s[1]
            else:
                symbol = ''
                key_part = s[0]

            if not key_part.startswith('state:'):
                continue

            last_value = self.last_infos.get(key)
            if last_value is None or last_value == value:
                continue

            print('State change: %s: %s -> %s' % (key, last_value, value))
            # for listener in self.listeners:
            #     listener.eval(symbol, key_part, last_value, value)

            if last_value and value and last_value.startswith('p') and value.startswith('p'):
                self.handle_percentile_transition(symbol, key_part.replace('state:', ''), last_value, value, dt)

    def handle_percentile_transition(self, symbol, indicator, last_value, value, dt):
        last_q = int(last_value.replace('p', ''))
        q = int(value.replace('p', ''))
        magnitude = q - last_q
        item = {
            'symbol': symbol,
            'indicator': indicator,
            'last_q': last_q,
            'q': q,
            'magnitude': magnitude,
            'datetime': dt,
        }
        # print('State change [%s] percentile, %s -> %s, magnitude %s' % (indicator, last_q, q, magnitude))
        # print(item)
        self.memory.append(item)
        # sell, buy, trend = self.signals(dt, symbol)
        # pprint(self.memory)
        # import pdb ; pdb.set_trace()

    def signals(self, dt, symbol):
        # Prune memory
        new_memory = []
        buy_signal = 0
        sell_signal = 0
        trend_signal = 0
        for item in self.memory:
            diff = dt - item['datetime']
            # print('diff:', diff)
            diff_seconds = diff.seconds
            if diff_seconds > self.memory_seconds:
                # print('drop item')
                continue
            new_memory.append(item)

        self.memory = new_memory
        for item in self.memory:
            if item['symbol'] != symbol:
                continue

            factor = (self.memory_seconds - diff_seconds) / float(self.memory_seconds)
            v = factor * item['magnitude'] * item['q']
            if item['indicator'] in self.bull_indicators:
                buy_signal += v
            if item['indicator'] in self.bear_indicators:
                sell_signal += v
            if item['indicator'] in self.trend_indicators:
                trend_signal += v
        # print('State signals: buy=%.1f\tsell=%.1f\ttrend=%.1f' % (buy_signal, sell_signal, trend_signal))
        return buy_signal, sell_signal, trend_signal


class Trader:
    def __init__(self, ib, account, contracts, bar_size='1 min', backtest=False,
        backtest_start=None, backtest_duration='3 D', use_obv=False, use_ema=False):
        self.ib = ib
        self.account = account
        self.backtest = backtest
        self.backtest_start = backtest_start
        self.logger = pnl.SignalLogger()
        self.use_obv = use_obv
        self.use_ema = use_ema
        self.targets = {}

        self.state_machine = StateMachine([])

        self.contracts = contracts
        start = time.perf_counter()
        self.contracts =  self.ib.qualifyContracts(*self.contracts)
        print(f"qualifying contracts async took {time.perf_counter() - start}")
        if self.backtest:
            provider_class = BacktestTickProvider
        else:
            provider_class = LiveTickProvider

        print('create the providers')
        self.providers = []
        num_failures = 0
        for i, contract in enumerate(self.contracts):
            print('processing %s/%s of contracts' % (i, len(self.contracts)))

            # self.trade_to_target_position(contract, 0)
            # self.take_profit_pnl(contract, 'p95')
            # self.take_profit('p95')
            # import pdb ; pdb.set_trace()
            # self.trade_to_target_position(contract, 4)
            # self.trade_to_target_position(contract, 4)
            # self.ib.sleep(1)
            # self.take_profit(1.5)
            # orders.pretty_print_open_trades(self.ib, self.account)
            # import pdb ; pdb.set_trace()

            try:
                provider = provider_class(ib, contract, bar_size=bar_size, duration=backtest_duration)
                self.providers.append(provider)
            except:
                num_failures += 1
                print('%s failures' % num_failures)
                if num_failures > 50:
                    raise
                print('\n\t!!! failed to get provider for %s, skip\n' % contract)
                raise

        # for provider in self.providers:
        #     get_trailing_obv(provider.trailing(500))

        print('self.backtest_start', self.backtest_start)
        if self.backtest and self.backtest_start:
            print('fast forward backtest')
            for provider in self.providers:
                while not provider.done():
                    bar = provider.last()
                    print('%s > %s ?' % (bar.datetime, self.backtest_start))
                    if bar.datetime >= self.backtest_start:
                        break
                    provider.incr()

        self.obv_analyzer_groups = []
        # self.states = {}

        self.analyzers = []
        for provider in self.providers:
            self.analyzers.append(GroupedAnalyzer(provider, [
                CurrentBarAnalyzer,
                TechnicalAnalyzer,
                ObvStateAnalyzer,
                AdxStateAnalyzer,
                DiPlusStateAnalyzer,
                DiMinusStateAnalyzer,
                EmaPeakAnalyzer,
            ]))

        for provider in self.providers:
            self.obv_analyzer_groups.append(get_full_analayzer_group(provider))

    def _get_contracts(self, contract_type):
        raise

    def run(self):
        print('run trader')

        self._last_bar_by_contract = {}
        self._target_position_for_contract = {}
        self._bar_position_for_contract = {}

        if self.backtest:
            sleep_incr = .01
        else:
            sleep_incr = 1
        while self.ib.sleep(sleep_incr):
            try:
                if self.use_obv:
                    self.iterate_obv()
                elif self.use_ema:
                    self.iterate_ema()
                else:
                    self.iterate()
            except BacktestDoneException:
                break

        if self.backtest:
            print('flatten')
            self.flatten_backtest()
            print('done flattening')
            self.logger.print_summary()
            # import pdb ; pdb.set_trace()

    async def run_async(self):
        print('run trader')

        self._last_bar_by_contract = {}
        self._target_position_for_contract = {}
        self._bar_position_for_contract = {}

        if self.backtest:
            sleep_incr = .01
        else:
            sleep_incr = 1
        while asyncio.sleep(sleep_incr, result=True):
            try:
                if self.use_obv:
                    self.iterate_obv()
                elif self.use_ema:
                    self.iterate_ema()
                else:
                    self.iterate()
            except BacktestDoneException:
                break

        if self.backtest:
            print('flatten')
            self.flatten_backtest()
            print('done flattening')
            self.logger.print_summary()
            # import pdb ; pdb.set_trace()

    def iterate(self):
        if not self.backtest and util.seconds_to_next_market_close() < 200:
            print('market closing soon, flatten positions and exit')
            util.flatten_positions(self.ib, self.account)
            print('done flattening, exit trading')
            return

        print('\n== iter ==\n')
        infos = self.analyzer.analysis()

        # pp.pprint(infos)
        print_states(infos)
        dt = self.analyzer.analyzers[0].provider.last().datetime
        self.state_machine.transition(infos, dt)
        [provider.incr() for provider in self.providers]

        did_trade = False

        for provider in self.providers:
            contract = provider.contract
            buy, sell, trend = self.state_machine.signals(dt, contract.localSymbol)

            contract_id = contract.conId
            target_position = self._target_position_for_contract.get(contract_id, 0)
            if buy > 0 and buy > sell:
                if buy > 100:
                    target_position = 2
                if buy > 500:
                    target_position = 5
                if buy > 1000:
                    target_position = 10
            elif sell > 0 and sell > buy:
                target_position = 0
                if sell > 100:
                    target_position = -2
                if sell > 500:
                    target_position = -5
                if sell > 1000:
                    target_position = -10

            current_position = self._get_quantity(contract)

            print('State signals: [%s]\tbuy=%.1f\tsell=%.1f\ttrend=%.1f\ttarget_qty=%s\tcurrent_qty=%s' % (contract.localSymbol, buy, sell, trend, target_position, current_position))

            current_bar = provider.last()
            if target_position > current_position:
                quantity = target_position - current_position
                self.buy(contract, quantity)
                did_trade = True
                if self.backtest:
                    signal = pnl.Signal(provider.contract, 'buy', quantity, current_bar.close, current_bar.datetime)
                    self.logger.log_signal(signal)
            elif target_position < current_position:
                quantity = current_position - target_position
                self.sell(contract, quantity, short_sell_ok=False)
                did_trade = True
                if self.backtest:
                    signal = pnl.Signal(provider.contract, 'sell', quantity, current_bar.close, current_bar.datetime)
                    self.logger.log_signal(signal)

        # HACK! Fix this with proper trade execution tracking
        if did_trade and not self.backtest:
            self.ib.sleep(5)

        return

    def iterate_ema(self):
        for provider, analyzer in zip(self.providers, self.analyzers):
            buy_line, sell_line, current_bar, reason, infos = self.analyze_ema(provider, analyzer)
            ema_peak = infos.get('ema_peak', 0) * 100
            print('%s\t%s\t%s\t%.1f%%\t%.1f%%' % (current_bar, buy_line, sell_line, infos.get('percent_increase', 0) * 100, ema_peak))

            target_position = None
            if buy_line:
                target_position = 2
            elif sell_line:
                target_position = -2

            if target_position is not None:
                reason = '%s, ema_peak=%.1f%%' % (reason, 100 * ema_peak)
                self.trade_to_target_position(provider.contract, target_position, reason=reason, backtest_bar=provider.last())

            provider.incr()

        if not self.backtest:
            self.take_profit('p95')
            self.ib.sleep(2)

    def iterate_obv(self):
        for provider, analyzer_group in zip(self.providers, self.obv_analyzer_groups):
            buy_line, sell_line, current_bar = self.analyze_obv(provider, analyzer_group)

            # buy_line = 1
            # sell_line = 0

            # if last_bar != current_bar:
            #     import pdb ; pdb.set_trace()
            # last_bar = current_bar
            print('current_bar', current_bar)

            contract_id = provider.contract.conId
            last_bar = self._last_bar_by_contract.get(contract_id, None)
            l = None
            c = None
            if last_bar:
                l = last_bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            if current_bar:
                c = current_bar.datetime.strftime('%Y-%m-%d %H:%M:%S')
            print('l, c', l, c)
            if current_bar and last_bar and l != c:
                self._target_position_for_contract[contract_id] = 0
                self._bar_position_for_contract[contract_id] = 0
            self._last_bar_by_contract[contract_id] = current_bar

            position_quantity = self._get_quantity(provider.contract)

            if buy_line and int(buy_line):
                print('BUY!')
                # target = max(int(buy_line), self._target_position_for_contract.get(contract_id, 0))
                # current = self._bar_position_for_contract.get(contract_id, 0)
                # self._target_position_for_contract[contract_id] = target
                # quantity = target - current
                # print('quantity', quantity)

                if not position_quantity:
                    buy_quantity = 10
                    self.buy(provider.contract, buy_quantity)
                    # self._bar_position_for_contract[contract_id] = quantity
                    if self.backtest:
                        signal = pnl.Signal(provider.contract, 'buy', buy_quantity, current_bar.close, current_bar.datetime)
                        self.logger.log_signal(signal)

            if sell_line and position_quantity > 0:
                print('SELL!')
                self.sell(provider.contract, position_quantity)
                if self.backtest:
                    signal = pnl.Signal(provider.contract, 'sell', position_quantity, current_bar.close, current_bar.datetime)
                    self.logger.log_signal(signal)

        if not self.backtest:
            self.take_profit('p75')
            # self.ib.sleep(1)

        provider.incr()

    def analyze_ema(self, provider, analyzer):
        contract = provider.contract
        infos = {
            'contract': contract,
            'last_bar': provider.last,
        }
        infos = analyzer.analysis()

        # pp.pprint(infos)
        # import pdb ; pdb.set_trace()

        buy_line = 0
        sell_line = 0
        reason = ''
        position_qty = self._get_quantity(contract)
        ema_diff_percent_ranges = ranges.get_ranges(self.ib, contract, '5 mins', 'cross_diff_percent_abs', 14)
        ema_diff_percent = infos.get('ema_diff_percent')
        cutoff = ema_diff_percent_ranges['p05']['min_value']

        significant_buy_cross = infos.get('cross_signal_int') == 1 and abs(ema_diff_percent) > cutoff
        significant_sell_cross = infos.get('cross_signal_int') == -1 and abs(ema_diff_percent) > cutoff
        if significant_buy_cross:
            buy_line = 1
            reason = 'ema LONG cross, (dist=%.2f%%, p05=%.4f%%)' % (ema_diff_percent * 100, cutoff * 100)

        if significant_sell_cross:
            sell_line = 1
            reason = 'ema SHORT cross (dist=%.2f%%, p05=%.4f%%)' % (ema_diff_percent * 100, cutoff * 100)

        return buy_line, sell_line, infos.get('current_bar'), reason, infos


    def analyze_obv(self, provider, analyzer_group):
        contract = analyzer_group[0].provider.contract
        # state = self.states[contract.conId]
        infos = {
            'contract': contract,
            'last_bar': analyzer_group[0].provider.last,
        }
        for analyzer in analyzer_group:
            try:
                info = analyzer.analysis()
                if not info:
                    info = {}
                infos.update(info)
            except:
                print('exception in analyzer %s for %s, lets keep going though...' % (analyzer, contract))
                raise
                continue

        # pp.pprint(infos)

        buy_line = infos.get('obv_buy_line')
        sell_line = infos.get('cross_signal') == -1

        # import pdb ; pdb.set_trace()
        # if infos.get('trailing_volume', 0) < 50:
        #     buy_line = 0

        return buy_line, sell_line, infos.get('current_bar')

    def handle_sell(self, symbol):
        print('TOOD')
        import pdb ; pdb.set_trace()

    def handle_buy(self, symbol):
        print('TOOD')
        import pdb ; pdb.set_trace()

    def buy(self, contract, quantity):
        position_qty = self._get_quantity(contract)
        print('buying when position is %s' % position_qty)
        if position_qty > 20:
            return
        trade = orders.buy(self.ib, self.account, contract, quantity, backtest=self.backtest)
        if trade:
            print('buy trade', orders.pretty_string_trade(trade))
        if trade and not self.backtest:
            util.slack_message('Executed BUY: %s' % orders.pretty_string_trade(trade))

    def sell(self, contract, quantity, short_sell_ok=False):
        position_qty = self._get_quantity(contract)
        if position_qty <= 0 and not short_sell_ok:
            print('position qty is %s, not selling' % position_qty)
            return
        quantity = min(quantity, position_qty)
        trade = orders.sell(self.ib, self.account, contract, quantity, backtest=self.backtest)
        if trade:
            print('sell trade', orders.pretty_string_trade(trade))
        if trade and not self.backtest:
            util.slack_message('Executed SELL: %s' % orders.pretty_string_trade(trade))

    def trade_to_target_position(self, contract, target_position, reason='', backtest_bar=None):
        open_trades = orders.open_trades_for_contract(self.ib, contract)
        if open_trades:
            for trade in open_trades:
                print('cancel order', trade.order)
                result = self.ib.cancelOrder(trade.order)
                # print('cancel result', result)
            self.ib.sleep(.1)

        current_position = self._get_quantity(contract)
        delta = target_position - current_position
        qty = abs(delta)
        print('current position is %s, target is %s' % (current_position, target_position))

        if delta > 0:
            orders.buy(self.ib, self.account, contract, qty, backtest=self.backtest)
            if self.backtest:
                signal = pnl.Signal(contract, 'buy', qty, backtest_bar.close, backtest_bar.datetime)
                # import pdb ; pdb.set_trace()
                self.logger.log_signal(signal)
        elif delta < 0:
            orders.sell(self.ib, self.account, contract, qty, backtest=self.backtest)
            if self.backtest:
                signal = pnl.Signal(contract, 'sell', abs(qty), backtest_bar.close, backtest_bar.datetime)
                # import pdb ; pdb.set_trace()
                self.logger.log_signal(signal)

        existing_target = self.targets.get(contract.conId)
        if existing_target is None or existing_target != target_position:
            if not self.backtest:
                bid, ask = util.last_bid_ask(self.ib, contract)
                util.slack_message('Set target position of %s for %s (bid=%s, ask=%s) %s' % (contract.localSymbol, target_position, bid, ask, reason))
        self.targets[contract.conId] = target_position

    def take_profit_pnl(self, contract, target_atr_percentile='p95'):
        atr_ranges = ranges.get_ranges(self.ib, contract, '1 hour', 'atr', 14)
        atr_value = atr_ranges[target_atr_percentile]['min_value']
        self.ib.qualifyContracts(contract)
        position = None
        for p in self.ib.positions():
            if p.account != self.account:
                continue
            if p.contract.conId != contract.conId:
                continue
            position = p
        avg_unit_cost = position.avgCost
        if position.contract.multiplier:
            avg_unit_cost = avg_unit_cost / float(position.contract.multiplier)
        take_profit_pnl = max(4 * atr_value, 0.015 - avg_unit_cost)
        return take_profit_pnl

    def take_profit(self, target_atr_percentile):
        print('take profit above', target_atr_percentile)

        positions = []
        for p in self.ib.positions():
            if p.account != self.account:
                continue

            if not self.trading_contract(p.contract):
                continue

            positions.append(p)

        contracts = [p.contract for p in positions]
        # print('qualify')
        # self.ib.qualifyContracts(*contracts)
        # print('get tickers')
        # tickers = self.ib.reqTickers(*contracts)

        i = 0

        summary = '\n\n'
        summary += '-' * 40
        summary += '\n'
        for position, contract in zip(positions, contracts):
            if position.position == 0:
                continue

            if position.position > 0:
                direction = 'long'
            else:
                direction = 'short'

            i += 1
            print('evaluate take profit for', position, i, len(positions))
            ticker = self._get_ticker_for(contract)
            bid, ask = None, None
            tries = 10
            while self.ib.sleep(.05):
                tries -= 1
                if not tries:
                    break
                bid_ticker, ask_ticker = ticker.bid, ticker.ask
                if bid_ticker > 0 and ask_ticker > 0:
                    bid = bid_ticker
                    ask = ask_ticker
                    break

            avg_unit_cost = position.avgCost

            if not avg_unit_cost or not bid or not ask:
                continue

            if position.contract.multiplier:
                avg_unit_cost = avg_unit_cost / float(position.contract.multiplier)

            atr_ranges = ranges.get_ranges(self.ib, contract, '1 hour', 'atr', 14)
            atr_value = atr_ranges[target_atr_percentile]['min_value']

            pnl_single = self.ib.reqPnLSingle(self.account, '', contract.conId)
            self.ib.sleep(.1)

            # print('%s: bid is %s and avg position cost is %s (%.2f), pnl: %s' % (position.contract.localSymbol, bid, avg_unit_cost, bid/avg_unit_cost, pnl_single))
            self.ib.cancelPnLSingle(self.account, '', contract.conId)

            upnl_per_position = pnl_single.unrealizedPnL / abs(position.position)
            take_profit_pnl = self.take_profit_pnl(contract)

            summary += '\t%s:\tavg cost=%0.2f\tposition=%s\tdaily PnL=%0.2f\tuPnL per pos=%0.2f (%.3f%%)\ttake_profit_pnl=%.2f (%.3f%%)\n' % (position.contract.localSymbol, avg_unit_cost, position.position, pnl_single.dailyPnL, upnl_per_position, (100 * pnl_single.unrealizedPnL / position.position) / avg_unit_cost, take_profit_pnl, (100 * take_profit_pnl)/avg_unit_cost)

            # import pdb ; pdb.set_trace()

            if not math.isnan(pnl_single.unrealizedPnL) and upnl_per_position >= take_profit_pnl:
                if not self.backtest:
                    util.slack_message('Take profit on %s (%s)' % (contract.localSymbol, upnl_per_position))
                print('Take profit!')
                self.trade_to_target_position(contract, 0)
        summary += '-' * 40
        summary += '\n\n'
        print(summary)
        # self.ib.sleep(1)

    def flatten_backtest(self):
        for provider in self.providers:
            position_quantity = self.logger.get_quantity(provider.contract)
            last_bar = provider.last()
            if position_quantity > 0:
                signal = pnl.Signal(provider.contract, 'sell', position_quantity, last_bar.close, last_bar.datetime)
            elif position_quantity < 0:
                signal = pnl.Signal(provider.contract, 'buy', -position_quantity, last_bar.close, last_bar.datetime)
            else:
                signal = None
            if signal:
                print('flatten w/ signal:', signal)
                self.logger.log_signal(signal)

    def trading_contract(self, contract):
        return contract.conId in [c.conId for c in self.contracts]

    def _get_quantity(self, contract):
        if self.backtest:
            return self.logger.get_quantity(contract)

        position_qty = 0
        for p in self.ib.positions():
            if p.account != self.account:
                continue
            if p.contract.conId == contract.conId:
                position_qty = p.position
        return position_qty

    def _get_ticker_for(self, contract, cache={}):
        self.ib.qualifyContracts(contract)
        # print('get ticker for', contract.conId, contract, cache)
        # if cache.get(contract.conId):
        #     return cache[contract.conId]
        # print('qualify', contract)
        # print('req tickers', contract)
        [ticker] = self.ib.reqTickers(contract)
        # cache[contract.conId] = ticker
        return ticker


def print_states(infos):
    for key, value in infos.items():
        s = key.split('.')
        # print(key, s)
        if len(s) > 1:
            symbol = s[0]
            key_part = s[1]
        else:
            symbol = ''
            key_part = s[0]

        if not key_part.startswith('state:'):
            continue

        print('State: [%s]\t%s=%s' % (symbol, key_part, value))


def test_callback():
    import pdb ; pdb.set_trace()
