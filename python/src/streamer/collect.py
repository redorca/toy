# standard library
import logging
import os
from typing import Optional, Union

# PyPi
import ib_insync as ibi

# local modules
from streamer import config
from streamer import davelogging as dl

logger = dl.logger(__name__, dl.DEBUG, dl.logformat)


class Gateway:
    """
        configuration details common to all gateways
        including a local TradersWorkstation
    """

    host: str = None
    port: int = None
    account: str = None
    timeout: float = None

    def __init__(self, host=None, port=None,  account=None, timeout=None):
        self.environment_overrider()
        if self.host is None:
            self.host = config.gateway_hosts.get(host)
        if self.port is None:
            self.port = port
        if self.account is None:
            self.account = account
        if self.timeout is None:
            self.timeout = timeout
  
    def environment_overrider(self):
        if (account_ := os.getenv("TB_ACCOUNT")) is not None:
            self.account = account_
            print(f"using account '{self.account}' from environment variable TB_ACCOUNT")
        if(host_ := os.getenv("TB_HOST")) is not None:
            self.host = config.gateway_hosts.get(host_)
            print(f"connecting to '{self.host}' from environment variable TB_HOST")
        if(port_ := os.getenv("TB_PORT")) is not None:
            self.port = int(port_)
            print(f"using port {self.port} from environment variable TB_PORT")
        if (timeout_ := os.getenv("TB_TIMEOUT")) is not None:
            self.timeout = float(timeout_)
            print(f"using port {self.timeout} from environment variable TB_PORT")

    def set_timeout(self, timeout: Union[int, float]):
        self.timeout = float(timeout)


class TradersWorkstation(Gateway):
    host = "localhost"
    port = 7497
    timeout = 5.0


class Btcjopaper(Gateway):
    host = config.gateway_hosts.get("btcjopaper", "btcjopaper.rockyahoo.com")
    account = "DF3987931"


class Btchfpaper(Gateway):
    host = config.gateway_hosts.get("btchfpaper", "btchfpaper.rockyahoo.com")


class Nypaperib(Gateway):
    host = config.gateway_hosts.get("nypaperib", "nypaperib.rockyahoo.com")


class Chpaperib(Gateway):
    host = config.gateway_hosts.get("chpaperib", "chpaperib.rockyahoo.com")


class GatewayFromEnvironment(Gateway):
    pass


class Connection:
    """
        this class provides a way to manage connection parameters,
        but the actual connection parameters are imported from streamer.config
        and passwords are kept in streamer.secrets.
        Streamer.secrets is NOT pushed to the repo.
        Use secrets_template.py as your guide to making your own.
    """

    client_id = 10

    def __init__(self, gateway: Optional[Gateway] = None):
        if gateway is None:
            self.gateway = TradersWorkstation()
        else:
            self.gateway = gateway
        self.ib = ibi.IB()

    def connect(self, client_id: Optional[int] = None):
        if client_id is None:
            self.client_id = Connection.client_id
            Connection.client_id += 1  # increment class variable
        self.ib.connect(
            host=self.gateway.host,
            port=int(self.gateway.port),
            clientId=self.client_id,
            timeout=float(self.gateway.timeout),
            account=self.gateway.account,
        )
        # if self.ib.isConnected():
        #     logger.info(f"successful connection to {self.gateway.host}")
        return self.ib

    async def connect_async(self, client_id=None):
        if client_id is None:
            self.client_id = Connection.client_id
            Connection.client_id += 1
        await self.ib.connectAsync(
            host=self.gateway.host,
            port=int(self.gateway.port),
            clientId=Connection.client_id,
            timeout=float(self.gateway.timeout),
        )
        return self.ib

    def close(self):
        if ib.isConnected():
            self.ib.disconnect()


if __name__ == "__main__":
    conn = Connection(Btcjopaper())
    ib = conn.connect()

    print("done")
    ib.disconnect()
