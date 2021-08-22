# Toy Project Async

Purpose is to create an async module that communicates with IB and TA as fast as possible while maintaining reliability
and scalability

## To run

`make run ARGS=trade`

### Example output:
```bash
make run ARGS=trade
Installing dependencies from lock file

No dependencies to install or update
2021-08-21 23:11:28.476 [CRITICAL] main - <module>: docopt for args
2021-08-21 23:11:28.476 [INFO] main - run: Starting args...
2021-08-21 23:11:28.485 [WARNING] main - run_trading_async: run async trading...
2021-08-21 23:11:28.485 [INFO] client - connectAsync: Connecting to 127.0.0.1:7496 with clientId 23...
2021-08-21 23:11:28.487 [INFO] client - connectAsync: Connected
2021-08-21 23:11:28.493 [INFO] client - _onSocketHasData: Logged on to server version 152
2021-08-21 23:11:28.494 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:usfuture.nj
2021-08-21 23:11:28.494 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:usfarm.nj
2021-08-21 23:11:28.494 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:usfuture
2021-08-21 23:11:28.494 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:cafarm
2021-08-21 23:11:28.494 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:cashfarm
2021-08-21 23:11:28.495 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:usopt
2021-08-21 23:11:28.495 [INFO] wrapper - error: Warning 2104, reqId -1: Market data farm connection is OK:usfarm
2021-08-21 23:11:28.495 [INFO] wrapper - error: Warning 2106, reqId -1: HMDS data farm connection is OK:ushmds
2021-08-21 23:11:28.495 [INFO] wrapper - error: Warning 2158, reqId -1: Sec-def data farm connection is OK:secdefnj
2021-08-21 23:11:28.495 [INFO] client - connectAsync: API connection ready
2021-08-21 23:11:28.503 [INFO] wrapper - position: position: Position(account='DU1825086', contract=Stock(conId=344809106, symbol='MRNA', exchange='NASDAQ', currency='USD', localSymbol='MRNA', tradingClass='NMS'), position=820.0, avgCost=341.55066219512196)
2021-08-21 23:11:28.503 [INFO] wrapper - position: position: Position(account='DU1825082', contract=Stock(conId=344809106, symbol='MRNA', exchange='NASDAQ', currency='USD', localSymbol='MRNA', tradingClass='NMS'), position=818.0, avgCost=341.4439413202934)
2021-08-21 23:11:28.504 [INFO] wrapper - position: position: Position(account='DU1825083', contract=Stock(conId=344809106, symbol='MRNA', exchange='NASDAQ', currency='USD', localSymbol='MRNA', tradingClass='NMS'), position=826.0, avgCost=341.8672152542373)
2021-08-21 23:11:28.504 [INFO] wrapper - position: position: Position(account='DU1825084', contract=Stock(conId=344809106, symbol='MRNA', exchange='NASDAQ', currency='USD', localSymbol='MRNA', tradingClass='NMS'), position=824.0, avgCost=341.7622734223301)
....
2021-08-21 23:11:28.544 [INFO] wrapper - position: position: Position(account='DU1825086', contract=Future(conId=428520008, symbol='RTY', lastTradeDateOrContractMonth='20210917', multiplier='50', currency='USD', localSymbol='RTYU1', tradingClass='RTY'), position=3.0, avgCost=116829.09999999999)
2021-08-21 23:11:28.740 [INFO] ib - connectAsync: Synchronization complete
2021-08-21 23:11:28.740 [INFO] main - run_trading_async: check if mkt data
2021-08-21 23:11:28.896 [INFO] ib - disconnect: Disconnecting from 127.0.0.1:7496, 238 B sent in 11 messages, 70.1 kB received in 1217 messages, session time 411 ms.
2021-08-21 23:11:28.897 [INFO] client - disconnect: Disconnecting

```

## To build and run in docker:

`make docker_build`
`make docker_run`

# TO build a binary for any platform:

`make build`

