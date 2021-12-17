import connect

from ib_insync import ScannerSubscription


def onScanData(scanData):
    print(scanData[0])
    print(len(scanData))


async def main():
    ib = await connect.connect_async()

    sub = ScannerSubscription(
        instrument="FUT.US", locationCode="FUT.GLOBEX", scanCode="TOP_PERC_GAIN"
    )
    scanData = ib.reqScannerSubscription(sub)
    scanData.updateEvent += onScanData
    ib.sleep(60)
    ib.cancelScannerSubscription(scanData)
