import time
import re
import os
import math
import datetime
# import talib as tal
import tulipy as tal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pprint
import joblib

# from pyculiarity import detect_ts
from scipy import signal

from bot import state_config
from bot import conf
from bot import util
from bot import ranges
from bot.provider import BacktestTickProvider, pull_historical_bars

bars_for_date_path = 'cache/bars'


pp = pprint.PrettyPrinter()


def analyze_obv(ib, contract, days_range=5, bar_size='30 secs'):
    mn, mx, mean, std, recent_dt = get_recent_obv_range(ib, contract, days_range=days_range-1)
    bars = []

    first_bar_of_today = None
    for i in range(days_range):
        date = datetime.datetime.now() - datetime.timedelta(days=days_range-i-1)
        print('get bars for date', date)
        day_bars = get_bars_for_date(ib, contract, date, bar_size)
        if not day_bars:
            continue
        # print('day_bars', day_bars)
        first_bar_of_today = day_bars[0]
        bars += day_bars

    print('first', bars[0])
    print('last ', bars[-1])
    # import pdb ; pdb.set_trace()

    high, low, close, volume = get_nparrays(bars)
    obv = tal.OBV(close, volume)

    df = pd.DataFrame({
        'date': [b.datetime for b in bars],
        'obv': obv,
        'mean_line': [mean] * len(obv),
        'min_line': [mn] * len(obv),
        'max_line': [mx] * len(obv),
    })
    ax = df.set_index('date').plot(title=contract.localSymbol)
    fig = ax.get_figure()
    plt.axvline(recent_dt, color='r')
    plt.axvline(first_bar_of_today.datetime, color='g')
    plt.show()
    plt.close(fig)


def get_trailing_obv(ib, contract, bars, should_plot=False):
    high, low, close, volume = get_nparrays(bars)
    obv = tal.OBV(close, volume)
    # adx = tal.ADX(high, low, close, timeperiod=27)
    # di_plus = tal.PLUS_DI(high, low, close, timeperiod=27)
    # di_minus = tal.MINUS_DI(high, low, close, timeperiod=27)

    window_size = conf.INDICATOR_WINDOW_SIZE
    moving_mean = []
    moving_std_dev = []
    delta_from_mean = []
    buy_line = []

    for i in range(window_size):
        moving_mean.append(0)
        delta_from_mean.append(0)
        moving_std_dev.append(0)
        buy_line.append(0)
    for i in range(len(obv) - window_size):
        window = obv[i:i+window_size]
        # mean = window.mean()
        mean = ema(window, len(window) - 1)[0]
        c_obv = obv[i + window_size]
        delta = c_obv - mean

        c_std = np.std(window)
        moving_std_dev.append(c_std)
        moving_mean.append(mean)
        delta_from_mean.append(delta)

        if delta > 2 * c_std:
            buy_line.append(delta / c_std)
        else:
            buy_line.append(0)

    return {}

    def norm(data):
        return data / (max(data) - min(data))

    # adx_peaks, _ = signal.find_peaks(adx, height=0, width=10)

    data_dict = {
        # 'datetime': [b.datetime for b in bars],
        # 'adx': adx,
        # 'close': close,
        'obv_buy_line': np.array(buy_line),
    }

    df = pd.DataFrame(data_dict).fillna(0)
    
    # print('obv std dev:', np.std(obv))
    # print(df)

    if should_plot:
        min_value = min(close)
        max_value = max(close)

        def norm_to_close(series):
            series_min = np.array(pd.Series(series).dropna().min())
            series_max = np.array(pd.Series(series).dropna().max())
            series_min -= .2 * (series_max - series_min)
            series_max += .2 * (series_max - series_min)

            # X falls between A and B, and you would like Y to fall between C and D
            # Y = (X-A)/(B-A) * (D-C) + C
            normalized = ((series - series_min) / (series_max - series_min)) * (max_value - min_value) + min_value
            return normalized

        plot_data_dict = {
            'datetime': data_dict['datetime'],
            'close': data_dict['close'],
        }

        # import pdb ; pdb.set_trace()

        for key in data_dict.keys():
            if key in ['datetime', 'close']:
                continue
            plot_data_dict[key] = norm_to_close(data_dict[key])

        plot_df = pd.DataFrame(plot_data_dict)

        print('plot_df')
        print(plot_df)

        plot_df.set_index('datetime').plot()
        plt.plot(plot_df['datetime'][adx_peaks], plot_df['adx'][adx_peaks], 'x')
        plt.show()

    record = df.to_dict(orient='records')[-1]
    result = {k: record[k] for k in ['obv_buy_line']}
    return result


def get_bars_for_date(ib, contract, date, bar_size):
    datestr = date.strftime('%Y%m%d')
    cached_path = '%s/%s_%s_%s_%s.pkl' % (bars_for_date_path, contract.symbol, contract.conId, datestr, bar_size.replace(' ', ''))

    os.makedirs(bars_for_date_path, exist_ok=True)
    if os.path.isfile(cached_path):
        print('reading from cache', cached_path)
        result = pickle.load(open(cached_path, 'rb'))
        return result

    start_dt = datetime.datetime(
        year=date.year, month=date.month, day=date.day,
        hour=6, minute=30, second=0)
    end_dt = datetime.datetime(
        year=date.year, month=date.month, day=date.day,
        hour=13, minute=30, second=0)
    provider = BacktestTickProvider(ib, contract, bar_size=bar_size, duration='3 D',
                                    end_datetime=end_dt + datetime.timedelta(days=1))
    bars = []
    while not provider.done():
        bar = provider.last()
        # print('bar', bar.datetime, start_dt, end_dt, bar.datetime >= start_dt, bar.datetime <= end_dt)
        if bar.datetime >= start_dt and bar.datetime <= end_dt:
            bars.append(bar)
        provider.incr()

    print('returing %s bars' % len(bars))

    print('saving to cache %s' % cached_path)
    pickle.dump(bars, open(cached_path, 'wb'))

    return bars


def get_recent_obv_range(ib, contract, days_range=3, bar_size='30 secs', end_dt=None):
    if end_dt is None:
        end_dt = datetime.datetime.now()

    recent_range_end_dt = datetime.datetime(
        year=end_dt.year, month=end_dt.month, day=end_dt.day,
        hour=13, minute=30, second=0)
    recent_range_end_dt -= datetime.timedelta(days=1)

    print('recent_range_end_dt', recent_range_end_dt)

    bars = []
    for i in range(days_range):
        date = recent_range_end_dt - datetime.timedelta(days=days_range-i-1)
        bars += get_bars_for_date(ib, contract, date, bar_size)

    high, low, close, volume = get_nparrays(bars)
    obv = tal.OBV(close, volume)
    result = (min(obv), max(obv), np.mean(obv), np.std(obv), recent_range_end_dt)
    # import pdb ; pdb.set_trace()
    return result


def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = np.array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema


class TrendAnalyzer:
    def __init__(self, provider):
        self.provider = provider
        self.fast = 9
        self.slow = 21
        self.adx_period = 27

    def analysis(self, offset=0, infos={}):
        bars = self.provider.trailing(100, offset)
        if not bars:
            return None

        high, low, close, volume = get_nparrays(bars)

        adx = tal.ADX(high, low, close, timeperiod=self.adx_period)
        obv = tal.OBV(close, volume)
        di_minus = tal.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        di_plus = tal.PLUS_DI(high, low, close, timeperiod=self.adx_period)

        datas = [
            (adx, 'adx'),
            (obv, 'obv'),
            (di_minus, 'di-'),
            (di_plus, 'di+'),
        ]

        # print('datas', datas)

        result = {}
        start_index = (max(self.slow, self.adx_period) * 2 - 1)
        for data, name in datas:
            data = data[start_index:]
            # print('name', name)
            # print('data', data)
            if data[-1]:
                slope_window = 32
                if slope_window > len(data):
                    raise Exception('slope window too large')

                slope_data = data[len(data) - slope_window:]
                index = [i for i in range(len(slope_data))]
                order = 1
                coeffs = np.polyfit(index, list(slope_data), order)
                slope = coeffs[0]
                slope_percent = coeffs[0] / data[-1]
            else:
                slope = None
                slope_percent = None

            result['%s_slope' % name] = slope
            result['%s_slope_percent' % name] = slope_percent
            result['%s_last' % name] = data[-1]
            if slope is None:
                result['%s_up' % name] = False
            else:
                result['%s_up' % name] = slope > 0
            
            last_value = data[len(data) - 1]
            before_last_value = data[len(data) - 2]
            if last_value and before_last_value:
                delta = last_value - before_last_value
                percent_delta = delta / last_value
            else:
                delta = None
                percent_delta = None

            result['%s_delta' % name] = delta
            result['%s_percent_delta' % name] = percent_delta

        trend_buy_line = 1
        for _, name in datas:
            if name == 'di-':
                continue
            if not result['%s_up' % name]:
                trend_buy_line = 0

        result['trend_buy_line'] = trend_buy_line

        return result



# class TrailingSpikeAnalyzer:
#     def __init__(self, provider, window):
#         self.provider = provider
#         self.window = window

#     @property
#     def contract(self):
#         return self.provider.contract

#     def analysis_list(self, num=10):
#         l = []
#         for i in range(num):
#             l.append(self.analysis(num - i - 1))
#         return l

#     def analysis(self, offset=0):
#         bars = self.provider.trailing(self.window, offset)
#         if not bars:
#             return None

#         initial_bar = bars[0]
#         spike_bar = None

#         max_percent_delta = 0
#         for bar in bars:
#             percent_delta = (bar.close - initial_bar.open) / initial_bar.open
#             if abs(percent_delta) > max_percent_delta:
#                 max_percent_delta = percent_delta
#                 spike_bar = bar

#         result = {
#             'initial_bar': initial_bar,
#             'current_bar': bars[-1],
#             'close': bars[-1].close,
#             'volume': bars[-1].volume,
#             'datetime': bars[-1].datetime,
#             'initial_price': initial_bar.open,
#             'max_percent_delta': max_percent_delta,
#         }
#         if spike_bar:
#             result['spike_price'] = spike_bar.close
#             result['spike_bar'] = spike_bar
#         else:
#             result['spike_price'] = None
#             result['spike_bar'] = None

#         return result



class TechnicalAnalyzer:
    def __init__(self, provider):
        self.provider = provider

        self.fast = 9
        self.slow = 21
        assert(self.fast < self.slow)

    @property
    def contract(self):
        return self.provider.contract

    def analysis_list(self, num=10):
        l = []
        for i in range(num):
            l.append(self.analysis(num - i - 1))
        return l

    def analysis(self, offset=0, infos={}):
        window_size = 100
        # bars = self.provider.trailing(max(self.slow, window_size), offset)
        bars = self.provider.trailing(None)
        if not bars:
            return None
        min_bars = max(14, conf.ADX_PERIOD) + 1
        if len(bars) < min_bars:
            bars = self.provider.trailing(min_bars)
        high, low, close, volume = get_nparrays(bars)
        current_bar = self.provider.last()

        adx_array = tal.ADX(high, low, close, timeperiod=conf.ADX_PERIOD)
        adx = adx_array[-1]
        di_minus = tal.MINUS_DI(high, low, close, timeperiod=conf.ADX_PERIOD)[-1]
        di_plus_array = tal.PLUS_DI(high, low, close, timeperiod=conf.ADX_PERIOD)
        di_plus = di_plus_array[-1]
        obv_array = tal.OBV(close, volume)
        obv = obv_array[-1]
        atr_array = tal.ATR(high, low, close, timeperiod=14)
        atr = atr_array[-1]

        # import pdb ; pdb.set_trace()

        natr_array = tal.NATR(high, low, close, timeperiod=14)
        natr = natr_array[-1]

        trange_array = tal.TRANGE(high, low, close)
        trange = trange_array[-1]

        peak_width = 3

        adx_peaks, _ = signal.find_peaks(adx_array, height=0, width=3)
        has_adx_peak = 0
        if adx_peaks.any():
            if adx_peaks[-1] == len(adx_array) - peak_width - 1:
                has_adx_peak = 1

        # obv_peaks, _ = signal.find_peaks(obv_array, height=0, width=3)
        # has_obv_peak = 0
        # if obv_peaks.any():
        #     # if obv_peaks[-1] == len(obv_array) - peak_width - 1:
        #     if obv_peaks[-1] >= len(obv_array) - peak_width - 1:
        #         has_obv_peak = 1

        di_plus_peaks, _ = signal.find_peaks(di_plus_array, height=0, width=3)
        has_di_plus_peak = 0
        if di_plus_peaks.any():
            if di_plus_peaks[-1] == len(di_plus_array) - peak_width - 1:
                has_di_plus_peak = 1

        has_obv_peak = 1
        peak_obv = obv_array[-1]
        for i in range(peak_width):
            val_index = -2 - i
            if abs(val_index) >= len(obv_array):
                has_obv_peak = 0
                break
            val = obv_array[val_index]
            # print('peak, val', peak_obv, val)
            if val > peak_obv:
                peak_obv = val
            else:
                has_obv_peak = 0
                break

        # if bars[-1].datetime.day == 11 and bars[-1].datetime.hour == 7 and bars[-1].datetime.minute == 40:
        #     import pdb ; pdb.set_trace()

        ema_fast = tal.EMA(close, timeperiod=self.fast)
        ema_slow = tal.EMA(close, timeperiod=self.slow)
        ema_diff = ema_slow[-1] - ema_fast[-1]
        ema_diff_percent = ema_diff / ema_slow[-1]

        # print(ema_fast)
        # print(ema_slow)

        p = None
        cross_signal = None
        last_cross_signal = None
        last_cross_dt = None
        last_cross_delta = None
        now = bars[-1].datetime
        i = 0
        percent_increase = 0
        for ef, es, bar in zip(ema_fast, ema_slow, bars):
            t = None
            if es > 0 and ef > 0:
                if ef > es:
                    t = 'fast'
                elif ef == es:
                    t = 'eq'
                else:
                    t = 'slow'

            # print('t', t)

            cross = ''
            # print('debug', i, bar.datetime, bar, '\t', p, '->', t, ef, es)
            if p and p != t:
                cross = '*'
                # if bar.datetime.month == 11 and bar.datetime.day == 3 and bar.datetime.hour == 10 and bar.datetime.minute == 25:
                #     print('DEBUG', bar.datetime, bar, '\t', p, '->', t)
                #     import pdb ; pdb.set_trace()
            else:
                cross = None

            p = t
            i += 1

            if cross and t == 'fast':
                cross_signal = 'buy'
                cross_signal_int = 1
                percent_increase = 0
                entry_price = bar.close
            elif cross and t == 'slow':
                cross_signal = 'sell'
                cross_signal_int = -1
                percent_increase = 0
                entry_price = bar.close
            else:
                cross_signal = None
                cross_signal_int = 0
                if last_cross_signal:
                    percent_increase = (current_bar.close - entry_price) / entry_price
                    if last_cross_signal == 'sell':
                        percent_increase = -percent_increase
            if cross_signal:
                last_cross_dt = bar.datetime
                last_cross_delta = (now - bar.datetime).seconds
                last_cross_signal = cross_signal

        if last_cross_signal == 'buy':
            cross_buy_state = 1
        else:
            cross_buy_state = -1

        cross_diff_percent = None
        cross_diff_percent_abs = None
        if cross_signal_int != 0:
            cross_diff_percent = ema_diff_percent
            cross_diff_percent_abs = abs(ema_diff_percent)

        # if cross_signal == 'buy' and bars[-1].datetime.day == 11:
        #     import pdb ; pdb.set_trace()

        return {
            'adx': adx,
            'atr': atr,
            'natr': natr,
            'trange': trange,
            'obv': obv,
            'has_adx_peak': has_adx_peak,
            'has_obv_peak': has_obv_peak,
            'has_di_plus_peak': has_di_plus_peak,
            'di+': di_plus,
            'di-': di_minus,
            'ema_diff': ema_diff,
            'ema_diff_percent': ema_diff_percent,
            'ema_diff_abs': abs(ema_diff),
            'ema_diff_percent_abs': abs(ema_diff_percent),
            'cross_diff_percent': cross_diff_percent,
            'cross_diff_percent_abs': cross_diff_percent_abs,
            'cross_signal': cross_signal,
            'cross_signal_int': cross_signal_int,
            # 'last_cross_dt': last_cross_dt,
            # 'last_cross_delta': last_cross_delta,
            # 'last_cross_signal': last_cross_signal,
            'cross_buy_state': cross_buy_state,
            'percent_increase': percent_increase,
        }


class ObvSpikeAnalyzer:
    def __init__(self, provider):
        self.provider = provider

    def analysis(self, offset=0, infos={}):
        window_size = 100
        bars = self.provider.trailing(window_size, offset)
        if not bars:
            return None

        high, low, close, volume = get_nparrays(bars)
        adx_period = 21
        adx = tal.ADX(high, low, close, timeperiod=adx_period)
        di_minus = tal.MINUS_DI(high, low, close, timeperiod=adx_period)
        di_plus = tal.PLUS_DI(high, low, close, timeperiod=adx_period)
        obv = tal.OBV(close, volume)

        trailing_mean = obv.mean()
        trailing_std = obv.std()
        last_obv = obv[-1]
        delta_from_mean = abs(trailing_mean - last_obv)
        if delta_from_mean > trailing_std:
            obv_buy_line = delta_from_mean / trailing_std
        else:
            obv_buy_line = 0

        trailing_volume = volume[-60:].sum()
        trailing_median_volume = np.median(volume[-20:])
        last_adx = adx[-1]
        last_di_plus = di_plus[-1]

        if trailing_volume < 100:
            obv_buy_line = 0

        return {
            'trailing_volume': trailing_volume,
            'obv_buy_line': obv_buy_line,
            'delta_from_mean': delta_from_mean,
        }


class CurrentBarAnalyzer:
    def __init__(self, provider):
        self.provider = provider

    def analysis(self, offset=0, infos={}):
        bar = self.provider.last()
        if not bar:
            return None

        # if bar.datetime.month == 12 and bar.datetime.day == 1 and bar.datetime.hour == 15 and bar.datetime.minute == 0:
        #     import pdb ;pdb.set_trace()

        result = {
            'high': bar.high,
            'low': bar.low,
            'open': bar.open,
            'close': bar.close,
            'datetime': bar.datetime,
            'current_bar': bar,
        }
        return result


class OptionExpiryAnalyzer:
    def __init__(self, provider):
        self.provider = provider

    def analysis(self, offset=0, infos={}):
        if self.provider.contract.secType != 'OPT':
            return None
        m = re.search(r'(\d\d\d\d)(\d\d)(\d\d)', self.provider.contract.lastTradeDateOrContractMonth)
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        expiry = datetime.datetime(year=year, month=month, day=day,
                                   hour=4, minute=0, second=0)
        delta = expiry - self.provider.last().datetime
        return {
            'expiry_days': delta.days,
            'expiry_seconds': delta.days * 24 * 3600 + delta.seconds,
        }

# class TradingActiveAnalyzer:
#     def __init__(self, provider):
#         self.provider = provider

#     def analysis(self, offset=0, infos={}):
#         bar = self.provider.last()
#         if bar.datetime.strftime('%Y-%m-%d') == '2020-10-18':
#             import pdb ; pdb.set_trace()


class BuyLineAnalyzer:
    def __init__(self, analyzer_group):
        self.analyzer_group = analyzer_group

    def analysis(self, offset=0, infos={}):
        infos = {}
        for analyzer in self.analyzer_group:
            try:
                info = analyzer.analysis()
                if not info:
                    info = {}
                infos.update(info)
            except:
                print('exception in analyzer %s for %s, lets keep going though...' % (analyzer))

        buy_line = 0
        sell_line = 0
        if infos.get('cross_buy_state', 0) < 0:
            buy_line = 0
            sell_line = 1
        elif infos.get('cross_buy_state', 0) > 0 and infos.get('obv', 0) > 0:
            buy_line = infos.get('obv_buy_line', 0)
            buy_line *= min(1, infos['obv'] / 10.)

        return {
            'buy_line': min(1000, buy_line),  # In case of inf value
            'sell_line': sell_line,
        }


class IndicatorStateAnalyzer:
    def _init_for_indicator(self, provider, indicator_name):
        self.indicator_name = indicator_name
        self.provider = provider
        self.state = 'none'
        self.ranges = ranges.get_ranges(self.provider.ib, self.provider.contract, self.provider.bar_size, self.indicator_name, conf.RANGES_NUM_DAYS)

    def analysis(self, offset=0, infos={}):
        window_size = conf.INDICATOR_WINDOW_SIZE
        bars = self.provider.trailing(window_size, offset)
        if not bars:
            return None

        value = self.get_value(bars)
        state_name = None
        sorted_keys = list(self.ranges.keys())
        sorted_keys.sort()
        for name in sorted_keys:
            range_def = self.ranges[name]
            if value > range_def['min_value'] and (state_name is None or state_name < name):
                # print('setting state to %s, min_value %s' % (state_name, range_def['min_value']))
                state_name = name

        result = {}
        result['state:%s' % self.indicator_name] = state_name

        # import pdb ; pdb.set_trace()

        return result


class AdxStateAnalyzer(IndicatorStateAnalyzer):
    def __init__(self, provider):
        self._init_for_indicator(provider, 'adx')

    def get_value(self, bars):
        high, low, close, volume = get_nparrays(bars)
        adx = tal.ADX(high, low, close, timeperiod=conf.ADX_PERIOD)
        return adx[-1]


class DiPlusStateAnalyzer(IndicatorStateAnalyzer):
    def __init__(self, provider):
        self._init_for_indicator(provider, 'di+')

    def get_value(self, bars):
        high, low, close, volume = get_nparrays(bars)
        di_minus = tal.MINUS_DI(high, low, close, timeperiod=conf.ADX_PERIOD)
        return di_minus[-1]


class DiMinusStateAnalyzer(IndicatorStateAnalyzer):
    def __init__(self, provider):
        self._init_for_indicator(provider, 'di-')

    def get_value(self, bars):
        high, low, close, volume = get_nparrays(bars)
        di_plus = tal.PLUS_DI(high, low, close, timeperiod=conf.ADX_PERIOD)
        return di_plus[-1]


class ObvStateAnalyzer(IndicatorStateAnalyzer):
    def __init__(self, provider):
        self._init_for_indicator(provider, 'obv')

    def get_value(self, bars):
        high, low, close, volume = get_nparrays(bars)
        obv = tal.OBV(close, volume)
        return obv[-1]


# class AdxStateAnalyzer:
#     def __init__(self, provider):
#         self.provider = provider
#         self.state = 'none'
#         self.last_adx = None
#         self.adx_flat_count = 0
#         self.adx_up_count = 0
#         self.adx_down_count = 0

#         self.weak_trend_cutoff = state_config.WEAK_TREND_NUM_BARS
#         self.strong_trend_cutoff = state_config.STRONG_TREND_NUM_BARS

#     def analysis(self, offset=0, infos={}):
#         window_size = conf.INDICATOR_WINDOW_SIZE
#         bars = self.provider.trailing(window_size, offset)
#         if not bars:
#             return None

#         high, low, close, volume = get_nparrays(bars)
#         adx_period = 21
#         adx = tal.ADX(high, low, close, timeperiod=adx_period)
#         current_adx = adx[-1]

#         if self.last_adx is not None:
#             if current_adx > self.last_adx:
#                 self.adx_up_count += 1
#                 self.adx_down_count = 0
#                 self.adx_flat_count = 0
#             elif current_adx < self.last_adx:
#                 self.adx_up_count = 0
#                 self.adx_down_count += 1
#                 self.adx_flat_count = 0
#             else:
#                 self.adx_up_count = 0
#                 self.adx_down_count = 0
#                 self.adx_flat_count += 1

#         if self.adx_up_count >= self.weak_trend_cutoff:
#             self.state = 'weak_up'
#         if self.adx_up_count >= self.strong_trend_cutoff:
#             self.state = 'strong_up'

#         if self.adx_down_count >= self.weak_trend_cutoff:
#             self.state = 'weak_down'
#         if self.adx_down_count >= self.strong_trend_cutoff:
#             self.state = 'strong_down'

#         if self.adx_flat_count > 0:
#             self.state = 'flat'

#         if current_adx > state_config.ADX_HIGH_CUTOFF:
#             value_state = 'high'
#         else:
#             value_state = 'low'

#         self.last_adx = current_adx

#         return {
#             'state:adx_trend': self.state,
#             'state:adx_value': value_state,
#         }


class EmaPeakAnalyzer:
    def __init__(self, provider, num=10):
        self.provider = provider
        self.num = num
        self.trailing_infos = []
        self.clf = joblib.load('ema_model.joblib')
        # df = pd.read_pickle(filename)

    def analysis(self, offset=0, infos={}):
        bar = self.provider.last()
        if not bar:
            return None

        self.trailing_infos.append(infos)
        if len(self.trailing_infos) > self.num:
            self.trailing_infos = self.trailing_infos[1:]

        ema_peak = 0
        if len(self.trailing_infos) == self.num:
            x = flat_infos_feature_vector(self.trailing_infos, self.provider.contract.localSymbol, ['di+', 'percent_increase'])
            x = pd.DataFrame.from_dict([x])
            x[x == np.inf] = np.nan
            x.fillna(x.mean(), inplace=True)
            ema_peak = self.clf.predict(x)[0]

        return {'ema_peak': ema_peak}


class GroupedAnalyzer:
    def __init__(self, providers, analyzer_classes):
        # self.providers = providers
        # self.analyzers = analyzers
        if type(providers) == list:
            self.should_prefix = True
        else:
            self.should_prefix = False
            providers = [providers]
        self.analyzers = []
        for provider in providers:
            for analyzer_class in analyzer_classes:
                self.analyzers.append(analyzer_class(provider))

    def analysis(self, offset=0, infos={}):
        infos = {}
        for analyzer in self.analyzers:
            info = analyzer.analysis(offset=offset, infos=infos)
            if info is None:
                continue

            for key, value in info.items():
                if self.should_prefix:
                    symbol = analyzer.provider.contract.localSymbol
                    key = '%s.%s' % (symbol, key)
                infos[key] = value
        return infos


def get_nparrays(bars):
    high = np.array([b.high for b in bars])
    low = np.array([b.low for b in bars])
    close = np.array([b.close for b in bars])
    volume = np.array([float(b.volume) for b in bars])
    return high, low, close, volume


def get_full_analayzer_group(provider):
    spike_window_size = 20
    group = [
        CurrentBarAnalyzer(provider),
        TechnicalAnalyzer(provider),
        TrendAnalyzer(provider),
        ObvSpikeAnalyzer(provider),
        OptionExpiryAnalyzer(provider),
    ]
    full_group = group + [BuyLineAnalyzer(group)]
    return full_group


def pull_training_data(ib, contract, start_date, end_date, read_cache=True, bar_size='1 min', use_rth=True):
    print('pull training data %s-%s' % (start_date, end_date))
    dt = util.parse_date_str(start_date, hour=23)
    df = None

    print(dt, end_date)

    while dt.strftime('%Y-%m-%d') <= end_date:
        print('pull for', dt.strftime('%Y-%m-%d'))
        # time.sleep(2)
        part = pull_training_data_for_date(
            ib, contract, dt.strftime('%Y-%m-%d'), read_cache, bar_size, use_rth=use_rth)
        provider = BacktestTickProvider(ib, contract, bar_size=bar_size, duration='1 D', end_datetime=dt)

        if df is None:
            df = part
        else:
            df = pd.concat([df, part])
        dt += datetime.timedelta(days=1)

    return df


def pull_training_data_for_date(ib, contract, date_str, read_cache=True, bar_size='1 min', use_rth=True):
    cache_key = 'training_data_%s_%s' % (contract.localSymbol, date_str)
    if read_cache:
        cached = util.get_cached_pickle(cache_key)
        if cached is not None:
            # print('training data cache hit %s' % cache_key)
            return cached

    provider = BacktestTickProvider(ib, contract, bar_size=bar_size, duration='1 D',
                                    end_datetime=util.parse_date_str(date_str),
                                    use_rth=use_rth)

    analyzer_group = get_full_analayzer_group(provider)
    # analyzer_group = [
    #     CurrentBarAnalyzer(provider),
    #     TechnicalAnalyzer(provider),
    #     OptionExpiryAnalyzer(provider),
    # ]

    infos_list = []
    # print('provider done?', provider.done())
    while not provider.done():
        infos = {}
        for analyzer in analyzer_group:
            info = analyzer.analysis()
            if info is None:
                continue
            infos.update(info)
        # print(infos.get('current_bar'))
        infos_list.append(infos)
        # pp.pprint(infos)
        bar = infos['current_bar']
        provider.incr()

    lookahead = 40
    for i, info in enumerate(infos_list):
        end = i+lookahead+1
        future_close_sum = 0
        upcoming_high = 0
        upcoming_low = 999999999999
        if end >= len(infos_list):
            y = None
        else:
            future_infos = infos_list[i+1:i+lookahead+1]

            has_data = bool(infos_list[i].get('current_bar'))
            for future_info in future_infos:
                if not future_info.get('current_bar'):
                    has_data = False

            sum_to_sell_signal = None
            if has_data:
                current_close = infos_list[i]['current_bar'].close
                future_close_sum = sum([info['current_bar'].close - current_close for info in future_infos])
                upcoming_high = max([info['current_bar'].high for info in future_infos])
                upcoming_low = min([info['current_bar'].low for info in future_infos])

                for j in range(1000):
                    idx = i + j
                    if idx >= len(infos_list):
                        break
                    info = infos_list[idx]

                    # import pdb ; pdb.set_trace()
                    if sum_to_sell_signal is None:
                        sum_to_sell_signal = 0
                    sum_to_sell_signal += info['current_bar'].close - current_close

                    if info.get('sell_line'):
                        break

        infos_list[i]['upcoming_high'] = upcoming_high
        infos_list[i]['upcoming_low'] = upcoming_low
        uh_percent = upcoming_high / info['current_bar'].close
        infos_list[i]['upcoming_high_percent'] = uh_percent
        ul_percent = upcoming_low / info['current_bar'].close
        infos_list[i]['upcoming_low_percent'] = ul_percent
        infos_list[i]['future_close_sum'] = future_close_sum
        fcs_percent = future_close_sum / info['current_bar'].close
        infos_list[i]['future_close_sum_percent'] = fcs_percent
        if fcs_percent > 2:
            fcs_buy_signal = 1
        else:
            fcs_buy_signal = 0

        uh_buy_signal = 0
        uh_sell_signal = 0
        if uh_percent > 1.2:
            uh_buy_signal = 1
        elif ul_percent < .8:
            uh_sell_signal = 1

        infos_list[i]['fcs_buy_signal'] = fcs_buy_signal
        infos_list[i]['uh_buy_signal'] = uh_buy_signal
        infos_list[i]['uh_sell_signal'] = uh_sell_signal
        infos_list[i]['lookahead'] = lookahead

    data = pd.DataFrame(infos_list)
    util.put_cached_pickle(cache_key, data)
    return data


def flat_infos_feature_vector(infos, prefix, features):
    feature_list = []
    combined = {}
    # print(infos)
    for i, info in enumerate(infos):
        for feature in features:
            combined['from_peak_%s_%s' % (i, feature)] = info[feature]
    return combined
    
