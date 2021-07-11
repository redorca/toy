import pprint
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

pp = pprint.PrettyPrinter()


def plot_backtest(provider, analyzer_group):
    infos_list = []
    print('plot backtest')
    i = 0
    plot_fields = [
        # 'di+',
        # 'adx',
        'obv',
        # 'trend_buy_line',
        # 'buy_line',
        # 'sell_line',
        # 'has_di_plus_peak',
        # 'cross_buy_state',
        'has_obv_peak',
        'cross_signal_int',
        'obv_buy_line',
        # 'trailing_volume',
        # 'delta_from_mean',
    ]
    fields = [
        # 'di+',
        # 'adx',
        'has_obv_peak',
        'obv',
        'buy_line',
        'sell_line',
        'obv_buy_line',
        # 'trend_buy_line',
        # 'has_di_plus_peak',
        # 'cross_buy_state',
        'cross_signal_int',
        # 'trailing_volume',
        # 'delta_from_mean',
    ]
    data = defaultdict(list)
    bars = []

    while not provider.done():
        print('.', end='')
        sys.stdout.flush()
        bars.append(provider.last())

        i += 1

        if i % 50 == 0:
            print(i)

        if i > 2000:
            break

        infos = {
            # 'contract': contract,
            # 'last_bar': analyzer_group[0].provider.last,
        }
        for analyzer in analyzer_group:
            info = analyzer.analysis()
            if not info:
                info = {}
            infos.update(info)

        print('== infos ==')
        pp.pprint(infos)

        for field in fields:
            data[field].append(float(infos[field]))

        data['datetime'].append(provider.last().datetime)
        data['close'].append(provider.last().close)
        provider.incr()

    df = pd.DataFrame(data).fillna(0)

    min_value = min(data['close'])
    max_value = max(data['close'])

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
        'datetime': data['datetime'],
        'close': data['close'],
    }
    for key in data.keys():
        if key in ['datetime', 'close']:
            continue
        if key not in plot_fields:
            continue
        plot_data_dict[key] = norm_to_close(data[key])

    plot_df = pd.DataFrame(plot_data_dict)
    # for field in plot_fields:
    #     plt.plot(plot_df['datetime'], plot_df[field], '-', label=field)
    plot_df.set_index('datetime').plot()
    # adx_peaks = []
    # for i, x in enumerate(df['has_adx_peak']):
    #     print('i, x', i, x)
    #     if x:
    #         adx_peaks.append(i)

    # plt.plot(plot_df['datetime'][adx_peaks], plot_df['adx'][adx_peaks], 'o')

    print(df.to_string())

    plt.title(str(provider.contract), loc='center', wrap=True)
    plt.show()


def get_nparrays(bars):
    high = np.array([b.high for b in bars])
    low = np.array([b.low for b in bars])
    close = np.array([b.close for b in bars])
    volume = np.array([float(b.volume) for b in bars])
    return high, low, close, volume
