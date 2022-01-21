import datetime
import sys

sys.path.append("//src")
import asyncio
import pytest
import streamer
from streamer import connect
import sys
import warnings

for item in sys.path:
    print(item)


def test_gateway():
    gateway = connect.Btcjopaper()

    assert isinstance(gateway, connect.Gateway)
    assert gateway.host is not None
    assert isinstance(gateway.port, int)
    assert isinstance(gateway.timeout, float)


def test_connection_object():

    conn = connect.Connection()
    assert isinstance(conn, connect.Connection)
    assert isinstance(conn.gateway, connect.Gateway)
    assert isinstance(conn.gateway, connect.TradersWorkstation)


# @pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_connection():
    gateway = connect.Btcjopaper()
    conn = connect.Connection(gateway)
    try:
        ib = conn.connect()
        assert isinstance(ib.reqCurrentTime(), datetime.datetime)
    except AssertionError as e:
        raise e
    except Exception as e:
        assert False, e
    finally:
        conn.close()
