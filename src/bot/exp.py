import datetime

import pandas as pd
import ib_insync as ibs

from bot import util
from bot.analyzer import get_nparrays


def get_premiums(ib, option):
    print('get_premiums', option)
    [ticker] = ib.reqTickers(option)
    print('market price:', ticker.marketPrice())
    import pdb ; pdb.set_trace()


def underlying_price(ib, option, cache={}):
    key = '%s:%s' % (option.symbol, option.exchange)
    if key in cache:
        return cache[key]
    stock = ibs.Stock(option.symbol, option.exchange, 'USD')
    [ticker] = ib.reqTickers(stock)
    val = ticker.marketPrice()
    cache[key] = val
    return val


def last_historical_ask(ib, contract):
    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime=None,
        durationStr='1 D',
        barSizeSetting='30 secs',
        whatToShow='TRADES',
        useRTH=False)
    import pdb; pdb.set_trace()


def get_spikes(ib, contract, bar_size, duration_str, end_date_time=None):
    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime=end_date_time,
        durationStr='1 D',
        barSizeSetting='30 secs',
        whatToShow='TRADES',
        useRTH=True)


    spike_window_size = 10
    spike_delta = .25

    high, low, close, volume = get_nparrays(bars)

    for i in range(len(bars) - spike_window_size):

        spike_window = bars[i:i + spike_window_size]

        spike_window_close = close
        # print('== spike window ==')
        initial_price = spike_window[0].open
        spike_price = None
        max_percent_delta = 0
        for bar in spike_window:
            percent_delta = (bar.close - initial_price) / initial_price
            if abs(percent_delta) > max_percent_delta:
                max_percent_delta = percent_delta
                spike_price = bar.close
        if percent_delta > spike_delta:
            print('spike! %s: %.1f%%, %s, %s->%s' % (util.contract_pretty(contract), percent_delta * 100., spike_window[0].date, initial_price, spike_price))


def get_spike_windows(out, trailing_spike_window_size, result_window_size):
    df = pd.DataFrame(out)
    # print(df)

    threshold = .15

    spike_results = []
    no_spike_results = []

    for i in range(len(df)):
        cross_signal = df['cross_signal'][i]
        if cross_signal:
            max_percent_delta = 0
            for j in range(trailing_spike_window_size):
                if i + j >= len(df):
                    continue
                mpd = df['max_percent_delta'][i + j]
                if abs(mpd) > abs(max_percent_delta):
                    max_percent_delta = mpd

            start = max(0, i - result_window_size)
            end = min(i + result_window_size, len(df))
            result = {
                'cross_signal': cross_signal,
                'cross_index': i,
                'window': df[start:end],
                'max_percent_delta': max_percent_delta,
            }
            if abs(max_percent_delta) > threshold:
                print('%s, %.2f' % (cross_signal, max_percent_delta))
                spike_results.append(result)
            else:
                no_spike_results.append(result)
    return spike_results, no_spike_results
