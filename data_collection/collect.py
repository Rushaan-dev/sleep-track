# Original code adapted from: https://github.com/Umang-Bansal/BCI
# Modifications made:
# - Fixed negative value dropping (isdigit bug replaced with try/except)
# - Added immediate CSV flush to prevent data loss on crash
# - Added serial disconnect error handling
# - Extended recording duration to 7200 seconds (2 hours)
# - Renamed output file to eeg_data.csv

import serial
import csv
import time
import datetime

COM_PORT = 'COM5'
BAUD_RATE = 115200

# Open the serial connection
ser = serial.Serial(COM_PORT, BAUD_RATE)

# Create a CSV file to save the data
with open('eeg_data.csv', 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Set the maximum duration of data collection (in seconds)
    max_duration = 7200  # 2 hours
    start_time = time.time()
    
    print("Collecting data...")
    
    while time.time() - start_time < max_duration:
        try:
            # Read a line of data from Arduino
            data = ser.readline().decode("latin-1").strip()
            
            # Get current timestamp
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Split data
            values = data.split(',')
            
            # Fix 1 - handles negative values correctly
            try:
                val = int(values[0])
                csvwriter.writerow([current_time, val])
                csvfile.flush()  # Fix 2 - immediate flush
            except ValueError:
                pass
                
        except serial.SerialException:
            print("Serial connection lost. Stopping.")
            break

ser.close()
print("Data collection complete. Saved to eeg_data.csv")
