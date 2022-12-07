import os
from typing import Union

# ACCOUNT = 'DU2326883'  # arden9050 #2
# ACCOUNT = 'DU2856241'  # btcvc4842
selector = 1
ACCOUNT = "DU1825086"  # btcadv966
ACCOUNT = "DF3987931"  # btcjo7537
HOST = ("127.0.0.1", "btcjopaper.rockyahoo.com", "btchfpaper.rockyahoo.com")[selector]
PORT = (7497, 4002, 4002)[selector]
MIN_DELTA = 0.1
MAX_DELTA = 0.7
RANGES_NUM_DAYS = 14
INDICATOR_WINDOW_SIZE = 100
ADX_PERIOD = 27
PERCENTILE_WIDTH = 5
# 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen   for explanation see:
# https://interactivebrokers.github.io/tws-api/market_data_type.html
MARKET_DATA_TYPE = 4

gateway_hosts = {
    "btcjopaper": "btcjopaper.rockyahoo.com",
    "btchfpaper": "btchfpaper.rockyahoo.com",
    "nypaperib": "nypaperib.rockyahoo.com",
    "chpaperib": "chpaperib.rockyahoo.com",
}
