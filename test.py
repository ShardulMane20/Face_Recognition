import os

# Define the path for the attendance CSV file
attendance_file = r"D:\Projects\Python Object Detection\attendance.csv"

# Check if the file already exists
if not os.path.exists(attendance_file):
    # Create the file and write headers
    with open(attendance_file, 'w') as f:
        f.write("Name, Timestamp\n")  # Writing headers
    print(f"Attendance file created: {attendance_file}")
else:
    print(f"Attendance file already exists: {attendance_file}")
