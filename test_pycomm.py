from pycomm3 import LogixDriver

PLC_Address = "10.20.30.100"
PLC_tag = "Clear"
Robot_PLC = LogixDriver(PLC_Address)
Robot_PLC.open() #uploads the tag list

def read_single():
    with LogixDriver(PLC_Address) as plc:
        return plc.read('DINT1')

try:
    with Robot_PLC as plc:
        result = plc.read(PLC_tag)
        if result.error is None:
            print(f"Tag: {result.tag}")
            print(f"Value: {result.value}")
            print(f"Data type: {result.type}")
        else:
            print(f"Read failed: {result.error}")
except Exception as error:
    print(f"PLC Connection Failed: {error}")