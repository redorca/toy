import time
import requests
import requests_cache
import datetime
import re
import pandas as pd

import ib_insync as ibs

from bot import util


requests_cache.install_cache('provider')

class NoBarsException(Exception):
    pass


class BacktestDoneException(Exception):
    pass


class Bar:
    def __init__(self, o, h, l, c, v, d, ibs_bar):
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.datetime = d
        self.ibs_bar = ibs_bar

        if ibs_bar:
            self.datetime += datetime.timedelta(hours=2)  # CA timezone adjustment 

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '[Bar o=%s, c=%s, h=%s, l=%s, v=%s, dt=%s]' % (self.open, self.close, self.high, self.low, self.volume, self.datetime)

    # Compatibility for crypto
    @property
    def date(self):
        return self.datetime


class TickProvider:
    subscribers = []

    def next(self, *args, **kwargs):
        raise NotImplemented()

    def last(self, *args, **kwargs):
        raise NotImplemented()

    def subscribe(self, obj):
        self.subscribers.append(obj)

    def notify_subscribers(self):
        if self.done():
            return

        for subscriber in self.subscribers:
            subscriber.on_provider_update()


class BacktestTickProvider(TickProvider):
    def __init__(self, ib, contract, bar_size='5 mins', duration='1 D', end_datetime=None, use_rth=True):
        self.ib = ib
        self.contract = contract
        self.bar_size = bar_size
        self.use_rth = use_rth

        m = re.search(r'(\d+) (.)', duration)
        num = int(m.group(1))
        unit = m.group(2)

        if unit != 'D':
            raise Exception('unexpected')

        self.bars = []
        if not end_datetime:
            end_datetime = datetime.datetime.now()
        for i in range(num + 20):
            dt = end_datetime - datetime.timedelta(days=i)
            date_str = dt.strftime('%Y-%m-%d')
            historical_bars  = pull_historical_bars(ib, contract, date_str, bar_size, use_rth=self.use_rth)
            self.bars += historical_bars

        # print('sort')
        self.bars.sort(key=lambda x: x.datetime)
        # print('done sort')
        # print('have %s bars' % len(self.bars))

        # import pdb ; pdb.set_trace()

        cutoff_dt = end_datetime - datetime.timedelta(days=(num - 1))
        cutoff_index = None
        found = False
        for i, bar in enumerate(self.bars):
            cutoff_index = i
            # print(bar.datetime.strftime('%Y-%m-%d'), cutoff_dt.strftime('%Y-%m-%d'))
            if bar.datetime.strftime('%Y-%m-%d') >= cutoff_dt.strftime('%Y-%m-%d'):
                found = True
                break

        if not found:
            cutoff_index = len(self.bars)

        # import pdb ; pdb.set_trace()

        self.start_index = cutoff_index
        self.index = cutoff_index
        self.iteration = 0
        self.window = 500

    def done(self):
        # print('done?', self.index, len(self.bars))
        return self.index >= len(self.bars)

    def incr(self):
        self.index += 1
        self.notify_subscribers()

    def last(self):
        if self.index < len(self.bars):
            return self.bars[self.index]
        else:
            return self.bars[-1]

    def trailing(self, num, offset=0):
        if self.done():
            raise BacktestDoneException()

        if not num:
            return self.bars[self.start_index - 1:self.index]

        period_bars = self.bars[(self.index - self.window):self.index]
        if num > len(period_bars):
            return []
        start = len(period_bars) - (num + offset)
        end = len(period_bars) - offset
        return period_bars[start:end]


class LiveTickProvider(TickProvider):
    def __init__(self, ib, contract, bar_size='5 mins', duration='1 D'):
        self.ib = ib
        self.contract = contract
        self.bar_size = bar_size
        self.duration = duration

        self._trailing = None

        # Initialize historical data and subscription
        bars = self.ib.reqHistoricalData(
            contract=self.contract,
            endDateTime='',
            durationStr=self.duration,
            barSizeSetting=self.bar_size,
            whatToShow='TRADES',
            useRTH=False,
            keepUpToDate=True)
        print('got %s historical bars for %s' % (len(bars), self.contract.localSymbol))
        self.on_bar_update(bars, True)
        bars.updateEvent += self.on_bar_update

    def last(self):
        if not self._trailing:
            return None
        return self._trailing[-1]

    def on_bar_update(self, bars, has_new_bar):
        if not bars:
            return
        print('!! bar update', bars[-1], has_new_bar)
        trailing = []
        for bar in bars:
            trailing.append(Bar(bar.open, bar.high, bar.low, bar.close, bar.volume, bar.date, bar))
        self._trailing = trailing

    def trailing(self, num, offset=0):
        if not self._trailing:
            return []

        if not num:
            return self._trailing

        start = len(self._trailing) - (num + offset)
        end = len(self._trailing) - offset
        return self._trailing[start:end]

    def done(self):
        return False

    def incr(self):
        pass


def pull_historical_bars(ib, contract, date_str, bar_size, use_rth=True):
    # print('pull historical bars', date_str, bar_size)
    if ib != 'crypto':
        ib.qualifyContracts(contract)
    today_date_str = datetime.datetime.now().strftime('%Y-%m-%d')

    should_cache = date_str != today_date_str
    cache_key = 'historical_bars_%s_%s_%s' % (contract.localSymbol, date_str, bar_size)
    cached_bars = util.get_cached_pickle(cache_key)
    # print('check cache', cache_key)
    # print('should_cache?', should_cache, date_str, today_date_str)
    if should_cache and cached_bars is not None:
        # print('cache hit! for %s' % cache_key)
        return cached_bars

    # import pdb ; pdb.set_trace()

    year, month, day = [int(s) for s in date_str.split('-')]
    print(year, month, day)
    end_dt = datetime.datetime(year=year, month=month, day=day,
                               hour=23, minute=59, second=0)

    # print('pull', cache_key)
    if ib == 'crypto':
        start_dt = datetime.datetime(year=year, month=month, day=day,
                                     hour=0, minute=0, second=0)
        start_dt -= datetime.timedelta(days=1)
        use_end_dt = end_dt + datetime.timedelta(days=1)
        ib_bars = []
        while start_dt < use_end_dt:
            batch_end_dt = start_dt + datetime.timedelta(hours=6)
            ib_bars += coinbase_bars(contract.pair, bar_size, start_dt, batch_end_dt)
            start_dt = batch_end_dt
    else:
        ib_bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime=end_dt,
            durationStr='2 D',
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=use_rth)

    ib_bars.sort(key=lambda x:x.date)

    matching_bars = []
    at_least_one_previous_day = False
    seen = set()
    for ib_bar in ib_bars:
        bar = Bar(ib_bar.open, ib_bar.high, ib_bar.low, ib_bar.close, ib_bar.volume, ib_bar.date, ib_bar)
        if ib_bar.date.isoformat() in seen:
            continue
        seen.add(ib_bar.date.isoformat())
        # print('checking bar', bar.datetime, date_str)
        if bar.datetime.strftime('%Y-%m-%d') < date_str:
            at_least_one_previous_day = True
        if bar.datetime.strftime('%Y-%m-%d') != date_str:
            continue
        # print('adding bar', bar.datetime)
        matching_bars.append(bar)

    print('returning %s matching bars, should_cache %s with key %s' % (len(matching_bars), should_cache, cache_key))
    # import pdb ; pdb.set_trace()

    if should_cache:
        util.put_cached_pickle(cache_key, matching_bars)
    return matching_bars


def pull_today_bars(ib, contract, bar_size):
    ib.qualifyContracts(contract)

    now = datetime.datetime.now()
    # date_str = datetime.datetime.now().strftime('%Y-%m-d')
    # cache_key = 'today_bars_%s_%s_%s' % (contract.localSymbol, date_str, bar_size)
    # cached_bars = util.get_cached_pickle(cache_key)

    if util.is_before_market_open():
        end_dt = datetime.datetime(year=now.year, month=now.month, day=now.day,
                                   hour=16, minute=0, second=0).astimezone(eastern)
        end_dt -= datetime.timedelta(days=1)

    ib_bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime=None,
        durationStr='360 S',
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True)
    print(ib_bars)


def coinbase_bars(pair, bar_size, start_dt, end_dt, retries=5):
    granularity = {
        '1 min': 60,
        '5 mins': 300,
    }[bar_size]
    url = 'https://api.pro.coinbase.com/products/%s/candles?granularity=%s&start=%s&end=%s' % (pair, granularity, start_dt.isoformat(), end_dt.isoformat())
    resp = requests.get(url)
    bars = []
    data = resp.json()
    if 'message' in data:
        if retries > 0:
            time.sleep(1)
            return coinbase_bars(pair, bar_size, start_dt, end_dt, retries-1)
        else:
            raise Exception(data['message'])
    for r in data:
        try:
            t, low, high, o, close, volume = r
            bars.append(Bar(o, high, low, close, volume, datetime.datetime.fromtimestamp(t), None))
        except:
            import pdb ; pdb.set_trace()
    return bars
