import datetime
import pprint

import ib_insync as ibs

from bot import TechnicalAnalyzer, TrendAnalyzer, TrailingSpikeAnalyzer, ObvSpikeAnalyzer, BuyLineAnalyzer, get_trailing_obv, get_recent_obv_range, analyze_obv
from bot.provider import BacktestTickProvider, LiveTickProvider
from bot.strategy import TrendStrategy
from bot.connect import connect
from bot.portfolio import Portfolio
from bot.contracts import suggest_options, suggest_futures, suggest_ranked_options, rank_contracts
from bot.plot import plot_backtest


test_client_id = 12

pp = pprint.PrettyPrinter()


def func(x):
    return x + 1


def test_fb_obv():
    ib = connect(client_id=test_client_id)
    contract = ibs.Option('FB', '20200821', 265, 'C', 'SMART')
    ib.qualifyContracts(contract)

    start_dt = datetime.datetime(year=2020, month=8, day=18, hour=11, minute=30)
    end_dt = datetime.datetime(year=2020, month=8, day=18, hour=12, minute=30)

    provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='1 D', end_datetime=end_dt)
    # analyzer = TechnicalAnalyzer(provider)
    analyzer = TrendAnalyzer(provider)

    bars_of_interest = []
    infos_of_interest = []
    while not provider.done():
        bar = provider.last()
        info = analyzer.analysis()
        provider.incr()

        print('bar', bar)
        if bar.datetime >= start_dt and bar.datetime < end_dt:
            bars_of_interest.append(bar)
            infos_of_interest.append(info)

    for bar, info in zip(bars_of_interest, infos_of_interest):
        print('bar of interest: %s\t%.2f\t%.2f\t%.2f\t%.2f' % (bar, info['obv_last'], info['obv_slope'], info['obv_delta'], info['obv_percent_delta']))
        print('bar of interest: %s\t%.2f\t%.2f\t%.3f' % (bar, info['obv_last'], info['obv_slope'], info['obv_slope_percent']))


def test_strategy_fb():
    ib = connect(client_id=test_client_id)
    contract = ibs.Option('FB', '20200821', 265, 'C', 'SMART')
    ib.qualifyContracts(contract)

    start_dt = datetime.datetime(year=2020, month=8, day=18, hour=11, minute=30)
    end_dt = datetime.datetime(year=2020, month=8, day=18, hour=12, minute=30)

    provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='1 D', end_datetime=end_dt)
    # analyzer = TechnicalAnalyzer(provider)
    # analyzer = TrendAnalyzer(provider)

    portfolio = Portfolio(ib)
    strategy = TrendStrategy(provider, portfolio)

    while not provider.done():
        bar = provider.last()
        signal = strategy.signal()
        print('bar: %s, signal: %s' % (bar, signal))
        portfolio.execute(signal)
        provider.incr()

    print('close positions')
    portfolio.close_positions(providers=[provider])
    print('pnl:', portfolio.pnl)


def test_crm_obv():
    ib = connect(client_id=test_client_id)
    contract = ibs.Option('CRM', '20200828', 270, 'C', 'SMART')
    ib.qualifyContracts(contract)

    start_dt = datetime.datetime(year=2020, month=8, day=26, hour=6, minute=30)
    end_dt = datetime.datetime(year=2020, month=8, day=26, hour=10, minute=30)

    provider = BacktestTickProvider(ib, contract, bar_size='1 min', duration='2 D', end_datetime=end_dt)
    # analyzer = TechnicalAnalyzer(provider)
    analyzer = TrendAnalyzer(provider)

    bars_of_interest = []
    infos_of_interest = []
    while not provider.done():
        bar = provider.last()
        info = analyzer.analysis()
        provider.incr()

        print('bar', bar)
        if bar.datetime >= start_dt and bar.datetime < end_dt:
            bars_of_interest.append(bar)
            infos_of_interest.append(info)

    print('bars_of_interest', bars_of_interest)

    for bar, info in zip(bars_of_interest, infos_of_interest):
        print('bar of interest: %s\t%.2f\t%.2f\t%.2f\t%.2f' % (bar, info['obv_last'], info['obv_slope'], info['obv_delta'], info['obv_percent_delta']))
        # print('bar of interest: %s\t%.2f\t%.2f\t%.3f' % (bar, info['obv_last'], info['obv_slope'], info['obv_slope_percent']))


def test_roku_obv():
    ib = connect(client_id=test_client_id)
    contract = ibs.Option('ROKU', '20200904', 185, 'C', 'SMART')
    ib.qualifyContracts(contract)

    start_dt = datetime.datetime(year=2020, month=8, day=26, hour=6, minute=30)
    end_dt = datetime.datetime(year=2020, month=9, day=1, hour=10, minute=30)

    provider = BacktestTickProvider(ib, contract, bar_size='1 min', duration='10 D', end_datetime=end_dt)

    bars_of_interest = []
    while not provider.done():
        bar = provider.last()
        provider.incr()

        if bar.datetime >= start_dt and bar.datetime < end_dt:
            bars_of_interest.append(bar)


    print('first', bars_of_interest[0])
    print('last ', bars_of_interest[-1])
    print('bars of interest', len(bars_of_interest))

    print('get_trailing_obv')

    get_trailing_obv(bars_of_interest)


def test_nkla_obv():
    ib = connect(client_id=test_client_id)
    # contract = ibs.Option('NKLA', '20200918', 50, 'C', 'SMART')
    # contract = ibs.Option('BYND', '20200911', 135, 'C', 'SMART')
    # contract = ibs.Stock('BYND', 'SMART')
    # contract = ibs.Option('AMZN', '20200918', 3120, 'C', 'SMART')
    # contract = ibs.Option('GOOGL', '20200918', 1535, 'P', 'SMART')
    # contract = ibs.Option('FB', '20200918', 275, 'P', 'SMART')
    # contract = ibs.Option('AAPL', '20200918', 116.25, 'C', 'SMART')
    # contract = ibs.Option('AAPL', '20200918', 115, 'C', 'SMART')
    # contract = ibs.Option('DIS', '20200918', 134, 'P', 'SMART')
    contract = ibs.Future('NQ', '20201218', 'GLOBEX')

    ib.qualifyContracts(contract)

    print('contract', contract)

    start_dt = datetime.datetime(year=2020, month=9, day=13, hour=6, minute=30)
    end_dt = datetime.datetime(year=2020, month=9, day=14, hour=13, minute=0)

    provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='1 D', end_datetime=None)

    bars_of_interest = []
    while not provider.done():
        bar = provider.last()
        print('bar', bar)
        provider.incr()

        # if bar.datetime >= start_dt and bar.datetime < end_dt:
        if bar.datetime >= start_dt:
            bars_of_interest.append(bar)

    print('first', bars_of_interest[0])
    print('last ', bars_of_interest[-1])
    print('bars of interest', len(bars_of_interest))

    print('get_trailing_obv')

    get_trailing_obv(ib, contract, bars_of_interest, should_plot=True)


def test_pyc():
    from pyculiarity import detect_ts
    import pandas as pd
    twitter_example_data = pd.read_csv('raw_data.csv',
                                    usecols=['timestamp', 'count'])
    print('twitter_example_data', twitter_example_data)
    import pdb ; pdb.set_trace()
    results = detect_ts(twitter_example_data,
                        max_anoms=0.02,
                        direction='both', only_last='day')


def test_get_range():
    ib = connect(client_id=test_client_id)
    contracts = suggest_options(ib)
    for contract in contracts:
        print(contract)
        mn, mx, mean, std = get_recent_obv_range(ib, contract)
        print('->', mn, mx, mean, std)


def test_suggest_options():
    ib = connect(client_id=test_client_id)
    contracts = suggest_options(ib)
    for contract in contracts:
        print(contract)
        provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='5 D')
        bars = []
        infos = []
        while not provider.done():
            bar = provider.last()
            provider.incr()
            bars.append(bar)
        print('contract', contract)
        get_trailing_obv(ib, contract, bars)


def test_analyze_obv():
    ib = connect(client_id=test_client_id)
    contracts = suggest_options(ib)
    for contract in contracts:
        print(contract)
        provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='5 D')
        bars = []
        infos = []
        while not provider.done():
            bar = provider.last()
            provider.incr()
            bars.append(bar)
        print('contract', contract)
        analyze_obv(ib, contract)


def test_mkt_data():
    ib = connect(client_id=test_client_id)
    # contract = ibs.Option('NKLA', '20200918', 50, 'C', 'SMART')
    # contract = ibs.Option('BYND', '20200911', 135, 'C', 'SMART')
    # contract = ibs.Stock('BYND', 'SMART')
    # contract = ibs.Option('AMZN', '20200918', 3120, 'C', 'SMART')
    # contract = ibs.Option('GOOGL', '20200918', 1535, 'P', 'SMART')
    # contract = ibs.Option('FB', '20200918', 275, 'P', 'SMART')
    # contract = ibs.Option('AAPL', '20200918', 116.25, 'C', 'SMART')

    # contract = ibs.Option('DIS', '20200918', 134, 'P', 'SMART')
    # contract = ibs.Stock('AAPL', 'SMART', 'USD')

    # contracts = suggest_futures(ib)
    contracts = suggest_ranked_options(ib, 100, 10)
    # contracts = [
    #     ibs.Option('AAPL', '20200925', 116.25, 'C', 'SMART'),
    #     ibs.Option('FB', '20200925', 250, 'P', 'SMART'),
    # ]
    [ib.qualifyContracts(c) for c in contracts]
    ranking = rank_contracts(ib, contracts)
    pp.pprint(ranking)


def test_plot():
    ib = connect(client_id=test_client_id)
    # contract = ibs.Future('NQ', '20201218', 'GLOBEX')
    # contract = ibs.Future('GC', '20201229', 'NYMEX')
    # contract = ibs.Option('AAPL', '20200918', 116.25, 'C', 'SMART')
    # contract = ibs.Option('FB', '20200925', 250, 'P', 'SMART')
    # contract = ibs.Option('AMD', '20200925', 77, 'P', 'SMART')
    # contract = ibs.Option('AMZN', '20201023', 3400, 'C', 'SMART')
    # contract = ibs.Option('FB', '20201023', 270, 'C', 'SMART')
    # contract = ibs.Option('AMZN', '20201023', 3300, 'P', 'SMART')

    # contract = ibs.Future('ES', '20201218', 'GLOBEX')
    contract = ibs.Future('NQ', '20201218', 'GLOBEX')

    ib.qualifyContracts(contract)
    # provider = BacktestTickProvider(ib, contract, bar_size='30 secs', duration='1 D', end_datetime=None)
    provider = BacktestTickProvider(ib, contract, bar_size='5 mins', duration='5 D', end_datetime=None)
    # import pdb ; pdb.set_trace()

    analyzer_group = [
        TechnicalAnalyzer(provider),
        TrendAnalyzer(provider),
        TrailingSpikeAnalyzer(provider, 20),
        ObvSpikeAnalyzer(provider)
    ]
    full_analyzer_group = analyzer_group + [BuyLineAnalyzer(analyzer_group)]

    plot_backtest(provider, full_analyzer_group)
