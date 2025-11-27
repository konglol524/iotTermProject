from gpiozero import AngularServo
from time import sleep

# UPDATE THIS: The GPIO pin number (BCM) you connected the signal wire to
SERVO_PIN = 12

# SG90 usually operates with a pulse width between 0.5ms and 2.5ms
servo = AngularServo(SERVO_PIN, min_angle=-90, max_angle=90, 
                     min_pulse_width=0.0005, max_pulse_width=0.0025)

print("Starting Servo Test... Press Ctrl+C to stop")

try:
    while True:
        print("Moving to -90 degrees")
        servo.angle = -90
        sleep(1)
        
        print("Moving to 0 degrees")
        servo.angle = 0
        sleep(1)
        
        print("Moving to 90 degrees")
        servo.angle = 90
        sleep(1)
        
except KeyboardInterrupt:
    print("Stopping...")