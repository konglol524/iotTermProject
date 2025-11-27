import smbus
import time

# I2C setup
# 0x23 is the standard address for BH1750. 
# If ADDR pin is connected to VCC, use 0x5C.
DEVICE_ADDRESS = 0x23 
POWER_ON = 0x01
RESET = 0x07
# Continuous High Resolution Mode
CONTINUOUS_HIGH_RES_MODE = 0x10

# Initialize I2C (Bus 1 is the default for GPIO 2 and 3)
bus = smbus.SMBus(1)

def convert_to_number(data):
    # Simple function to convert 2 bytes of data
    # into a decimal number
    return ((data[1] + (256 * data[0])) / 1.2)

def read_light():
    try:
        # Read data from I2C interface
        data = bus.read_i2c_block_data(DEVICE_ADDRESS, CONTINUOUS_HIGH_RES_MODE)
        return convert_to_number(data)
    except OSError as e:
        print(f"Error reading from sensor: {e}")
        print("Check wiring: SDA->GPIO2, SCL->GPIO3, VCC->3.3V, GND->GND")
        return None

print(f"Reading BH1750 Light Sensor at address {hex(DEVICE_ADDRESS)}...")
print("Press CTRL+C to stop")

try:
    while True:
        light_level = read_light()
        
        if light_level is not None:
            print(f"Light Level: {light_level:.2f} lux")
            
            # Simple logic examples
            if light_level < 10:
                print(" -> Too dark!")
            elif light_level > 500:
                print(" -> Very bright!")
                
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMeasurement stopped by user")