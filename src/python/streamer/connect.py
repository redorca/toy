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
    timeout: float = 10.0

    def __init__(self):
        self.environment_overrider()

    def environment_overrider(self):
        acct_ = os.getenv("TB_ACCOUNT")  # getenv returns None if not set
        if acct_ is not None:
            print(f"using account '{acct_}' from environment variable TB_ACCOUNT")
            self.account = acct_
        host_ = os.getenv("TB_HOST")
        if host_ is not None:
            print(f"connecting to '{host_}' from environment variable TB_HOST")
            self.host = host_
        port_ = os.getenv("TB_PORT")
        if port_ is not None:
            port_ = int(port_)
            print(f"using port {port_} from environment variable TB_PORT")
            self.port = port_
        timeout = os.getenv("TB_TIMEOUT")

    def change_timeout(self, timeout: Union[int, float]):
        self.timeout = float(timeout)


class TradersWorkstation(Gateway):
    host = "localhost"
    port = 7497
    timeout = 5.0


class Btcjopaper(Gateway):
    host = config.gateway_hosts.get("btcjopaper", "btcjopaper.rockyahoo.com")
    port = 4002
    # port = config.gateway_hosts.get('port')
    account = "DF3987931"


class Btchfpaper(Gateway):
    host = config.gateway_hosts.get("btchfpaper", "btchfpaper.rockyahoo.com")
    port = 4002

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

    def __build_args__(self):
        host=self.gateway.host
        port=int(self.gateway.port)
        clientId=int(self.client_id)
        timeout=float(self.gateway.timeout)

        return tradingargs = {'host':host, 'port':port, 'clientId':clientId, 'timeout':timeout}

    def connect(self, client_id: Optional[int] = None):
        if client_id is None:
            self.client_id = Connection.client_id
            Connection.client_id += 1  # increment class variable
        connect_args = self.__build_args__()
        self.ib.connect(**connect_args)
        if self.ib.isConnected():
            logger.info(f"successful connection to {self.gateway.host}")
        return self.ib

    async def connect_async(self, client_id=None):
        if client_id is None:
            self.client_id = Connection.client_id
            Connection.client_id += 1

        connect_args = self.__build_args__()
        await self.ib.connectAsync(**connect_args)
        return self.ib

    def close(self):
        if self.ib.isConnected():
            self.ib.disconnect()


if __name__ == "__main__":
    conn = Connection(Btcjopaper())
    ib = conn.connect()
    print("connected")

    conn.close()
    print("disconnected")
