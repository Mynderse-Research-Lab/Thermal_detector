from pycomm3 import LogixDriver

PLC_Address = "10.10.1.10" #update
PLC_tag = "UN01R15_uRobot_1_IO.IN.DO73_ClrOfFix1" #update
#Robot_PLC = LogixDriver(PLC_Address)
#Robot_PLC.open() #uploads the tag list

#def read_single():
#    with LogixDriver(PLC_Address) as plc:
#        return plc.read('DINT1')

try:
    with LogixDriver(PLC_Address) as plc:
        print(f"Connected to PLC at {PLC_Address}")
        result = plc.read(PLC_tag)
        if result.error is None:
            print(f"Tag: {result.tag}")
            print(f"Value: {result.value}")
            print(f"Data type: {result.type}")
            if result.value == True:
                print("Robot is CLEAR of thermal cam zone.")
            elif result.value == False:
                print("Robot is NOT clear of thermal cam zone.")
            else:
                print("Status Unavailable")
        else:
            print(f"Read failed: {result.error}")
except Exception as error:
    print(f"PLC Connection Failed: {error}")

