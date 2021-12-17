import logging
import ib_insync as ibs

# from bot.crypto import CryptoConnection


def connect(contract_type=None, client_id=11):
    # if contract_type == "crypto":
    #     return CryptoConnection()

    ib = ibs.IB()
    ib_gateway_port = 4002
    ib_gateway_port_live = 4001
    trader_workstation_port = 7496
    # ib.connect('127.0.0.1', ib_gateway_port_live, clientId=client_id)  # IB Gateway
    # ib.connect('127.0.0.1', ib_gateway_port, clientId=client_id)  # IB Gateway
    # logger=logging.getLogger().setLevel(logging.DEBUG)
    ib.connect(
        "127.0.0.1", trader_workstation_port, clientId=client_id, timeout=20
    )  # Trader Workstation
    return ib


connection_defaults = {
    "ib_gateway_port_live": 4001,
    "ib_gateway_port": 4002,
    "trader_workstation_port": 7496,
}

btcjo = connection_defaults.copy()
btcjo.update(
    {"host": "btcjo.rockyahoo.com", "port": connection_defaults["ib_gateway_port"]}
)

localhost = connection_defaults.copy()
localhost.update(
    {"host": "127.0.0.1", "port": connection_defaults["trader_workstation_port"]}
)


async def connect_async(config: dict = btcjo, contract_type=None, client_id=11):
    # if contract_type == "crypto":
    #     return CryptoConnection()

    ib = ibs.IB()
    ib_gateway_port = config["ib_gateway_port"]
    ib_gateway_port_live = config["ib_gateway_port_live"]
    trader_workstation_port = config["trader_workstation_port"]
    port = config["port"]
    # ib.connect('127.0.0.1', ib_gateway_port_live, clientId=client_id)  # IB Gateway
    # ib.connect('127.0.0.1', ib_gateway_port, clientId=client_id)  # IB Gateway
    # logger=logging.getLogger().setLevel(logging.DEBUG)
    print(f"connecting to {config['host']}")
    await ib.connectAsync(
        config["host"],
        port,  # was traders workstation port
        clientId=client_id,
        timeout=20,
    )  # Trader Workstation
    return ib
