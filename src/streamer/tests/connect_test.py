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


def test_defaults():
    conn = connect.Connection()
    assert isinstance(conn.tws, dict)
    assert isinstance(conn.gw_defaults, dict)
    assert isinstance(conn.btcjo, dict)
    assert conn.tws["port"] == 7496


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_connection():
    conn = connect.Connection()
    conn.select(connect.Connection.btcjo)
    try:
        ib = conn.connect()
    except:
        assert False, "failed to connect"
