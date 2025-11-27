from gpiozero import MotionSensor
from time import sleep

# Connect the OUTPUT pin of the PIR sensor to GPIO 17
pir = MotionSensor(17)

print("PIR Module Test (CTRL+C to exit)")
print("Waiting for sensor to settle...")
pir.wait_for_no_motion() # Wait for the sensor to initialize
print("Ready")

try:
    # Loop constantly to check the status
    while True:
        if pir.value:
            print("Motion detected!")
        else:
            print("No motion")
        
        # Adjust sleep to change how fast it reads (e.g., 0.5 or 1 second)
        sleep(1)

except KeyboardInterrupt:
    print("Stopped")