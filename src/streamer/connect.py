import logging
import ib_insync as ibs

# from bot.crypto import CryptoConnection
class Connection:
    """just a way to manage connection defaults"""

    tws = {  # local traders workstation connection
        "host": "127.0.0.1",
        "port": 7496,
    }
    gw_defaults = {
        "ib_gateway_port_live": 4001,
        "ib_gateway_port": 4002,
        "port": 4002,
        "timeout": 10.0,
    }
    btcjo = gw_defaults.copy()
    btcjo.update({"host": "btcjopaper.rockyahoo.com"})

    btchf = gw_defaults.copy()
    btchf.update({"host": "btchfpaper.rockyahoo.com"})

    client_id = 50

    def __init__(self):
        self.connection_dict = None
        self.ib = ibs.IB()

    def select(self, connection: dict):
        """use one of the class dictionaries: tws, btcjo or btchf"""
        self.connection_dict = connection

    def connect(self, client_id=None):
        if client_id is None:
            client_id = Connection.client_id
            Connection.client_id += 1
        self.ib.connect(
            host=self.connection_dict["host"],
            port=int(self.connection_dict["port"]),
            clientId=client_id,
            timeout=float(self.connection_dict["timeout"]),
        )
        return self.ib

    async def connect_async(self, client_id=None):
        if client_id is None:
            client_id = Connection.client_id
            Connection.client_id += 1
        await self.ib.connectAsync(
            host=self.connection_dict["host"],
            port=int(self.connection_dict["port"]),
            clientId=Connection.client_id,
            timeout=float(self.connection_dict["timeout"]),
        )
        return self.ib
