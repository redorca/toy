import logging
import ib_insync as ibs

from bot.crypto import CryptoConnection

connection_defaults = {
    'ib_gateway_port_live': 4001,
    'ib_gateway_port': 4002,
    'trader_workstation_port': 7496,
    'default': 4002,
    'timeout': 10.0,
}

btchfpaper = connection_defaults.copy()
btchfpaper.update(  {
    'port': connection_defaults['default'],
    'host': 'btchfpaper.rockyahoo.com',
})

btcjo = connection_defaults.copy()
btcjo.update(    {
    'host': 'btcjo.rockyahoo.com',
    'port': connection_defaults['ib_gateway_port']
})

localhost = connection_defaults.copy()
localhost.update({
    'host': '127.0.0.1',
    'port': connection_defaults["trader_workstation_port"]
})

def connect(config:dict=btchfpaper, contract_type=None, client_id=11):
    if contract_type == 'crypto':
        return CryptoConnection()

    ib_gateway_port = config['port']
    ib_gateway_host = config['host']
    ib = ibs.IB()
    ib.connect(ib_gateway_host, ib_gateway_port, clientId=client_id, timeout=config['timeout'])
    return ib

async def connect_async(config:dict=btcjo, contract_type=None, client_id=11):
    if contract_type == 'crypto':
        return CryptoConnection()

    ib = ibs.IB()
    ib_gateway_port = config['port']
    ib_gateway_port_live = config['ib_gateway_port_live']
    trader_workstation_port = config['host']
    port = config["port"]
    print(f"connecting to {config['host']}")
    await ib.connectAsync(config['host'],
        ib_gateway_port, # was traders workstation port
        clientId=client_id,
        timeout=20
    )
    return ib
