# standard library
import logging
import os
from typing import Optional

# PyPi
import ib_insync as ibs

# local modules
from streamer import config


class Gateway:
    """configuration details common to all gateways
    including a local TradersWorkstation
    """

    host: str = None
    port: int = 4002
    account: str = None
    timeout: float = 10.0

    def env_overrides(self):
        acct = os.getenv("TB_ACCOUNT")  # getenv returns None if not set
        if acct is not None:
            print(f"using account '{acct}' from environment variable TB_ACCOUNT")
            self.account = acct
        host = os.getenv("TB_HOST")
        if host is not None:
            print(f"connecting to '{host}' from environment variable TB_HOST")
            self.host = host
        port = os.getenv("TB_PORT")
        if port is not None:
            port = int(port)
            print(f"using port {port} from environment variable TB_PORT")
            self.port = port
        timeout = os.getenv("TB_TIMEOUT")

    def change_timeout(self, timeout: Optional[float] = None):
        if timeout is not None:
            self.timeout = float(timeout)


class TradersWorkstation(Gateway):
    host = "localhost"
    port = 7497
    timeout = 5.0

    def __init__(self):
        self.env_overrides()


class Btcjopaper(Gateway):
    host = config.gateway_hosts["btcjopaper"]

    def __init__(self):
        self.env_overrides()


class Btchfpaper(Gateway):
    host = config.gateway_hosts["btchfpaper"]

    def __init__(self):
        self.env_overrides()


class Nypaperib(Gateway):
    host = config.gateway_hosts["nypaperib"]

    def __init__(self):
        self.env_overrides()


class Chpaperib(Gateway):
    host = config.gateway_hosts["chpaperib"]

    def __init__(self):
        self.env_overrides()


class Connection:
    """
    this class provides a way to manage connection parameters,
    but the actual connection parameters are imported from streamer.config
    and passwords are kept in streamer.secrets.
    Streamer.secrets is NOT pushed to the repo.
    Use secrets_template.py as your guide to making your own.
    """

    client_id = 50

    def __init__(self, gateway: Optional[Gateway] = None):
        if gateway is None:
            self.gateway = TradersWorkstation()
        else:
            self.gateway = gateway
        self.ib = ibs.IB()

    def connect(self, client_id: Optional[int] = None):
        if client_id is None:
            self.client_id = Connection.client_id
            Connection.client_id += 1  # increment class variable
        self.ib.connect(
            host=self.gateway.host,
            port=int(self.gateway.port),
            clientId=self.client_id,
            timeout=float(self.gateway.timeout),
        )
        # log unsuccessful connect
        return self.ib

    def close(self):
        if ib.isConnected():
            self.ib.disconnect()

    async def connect_async(self, client_id=None):
        if client_id is None:
            client_id = Connection.client_id
            Connection.client_id += 1
        await self.ib.connectAsync(
            host=self.gateway.host,
            port=int(self.gateway.port),
            clientId=Connection.client_id,
            timeout=float(self.gateway.timeout),
        )
        return self.ib


if __name__ == "__main__":
    conn = Connection(Btcjopaper())
    ib = conn.connect()

    print("done")
    ib.disconnect()
