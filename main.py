import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime

# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLOv8 Model...")
model = YOLO("yolov8n.pt")

# ==========================================
# SETTINGS
# ==========================================

MIN_DISTANCE = 80
HIGH_CROWD_THRESHOLD = 10

# ==========================================
# DETECTION FUNCTION
# ==========================================

def detect_people_and_distance(frame):

    results = model(frame, verbose=False)

    centers = []
    boxes = []

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            # PERSON CLASS
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                centers.append((center_x, center_y))
                boxes.append((x1, y1, x2, y2))

    violation_indexes = set()

    # ==========================================
    # DISTANCE CHECK
    # ==========================================

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):

            distance = np.linalg.norm(
                np.array(centers[i]) - np.array(centers[j])
            )

            if distance < MIN_DISTANCE:

                violation_indexes.add(i)
                violation_indexes.add(j)

                cv2.line(
                    frame,
                    centers[i],
                    centers[j],
                    (0, 0, 255),
                    2
                )

    # ==========================================
    # DRAW BOXES
    # ==========================================

    for idx, (x1, y1, x2, y2) in enumerate(boxes):

        if idx in violation_indexes:
            color = (0, 0, 255)
            label = "Too Close"
        else:
            color = (0, 255, 0)
            label = "Safe"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame, len(boxes), len(violation_indexes)

# ==========================================
# CHOOSE INPUT MODE
# ==========================================

print("\nChoose Mode:")
print("1 - Live Camera")
print("2 - Video File")

choice = input("Enter 1 or 2: ")

if choice == "1":

    cap = cv2.VideoCapture(0)

elif choice == "2":

    video_path = input("Enter full video path: ")
    cap = cv2.VideoCapture(video_path)

else:
    print("Invalid Choice")
    exit()

if not cap.isOpened():
    print("Error opening source")
    exit()

# ==========================================
# FULLSCREEN WINDOW
# ==========================================

window_name = "Generative AI Crowd Monitoring System"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ==========================================
# TRACKING VARIABLES
# ==========================================

max_people_detected = 0
total_violations = 0
high_crowd_frames = 0

# ==========================================
# DAILY TRACKING VARIABLES
# ==========================================

today_date = datetime.now().strftime("%Y-%m-%d")

session_people_total = 0
session_frames = 0

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1000, 600))

    output_frame, people_count, violations = detect_people_and_distance(frame)

    # ==========================================
    # UPDATE SESSION STATS
    # ==========================================

    session_people_total += people_count
    session_frames += 1

    # ==========================================
    # UPDATE GLOBAL STATS
    # ==========================================

    if people_count > max_people_detected:
        max_people_detected = people_count

    total_violations += violations

    # ==========================================
    # CROWD ALERT
    # ==========================================

    if people_count >= HIGH_CROWD_THRESHOLD:

        high_crowd_frames += 1

        cv2.putText(
            output_frame,
            "ALERT: CROWD INCREASING!",
            (300, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ==========================================
    # DISPLAY LIVE STATS
    # ==========================================

    cv2.putText(
        output_frame,
        f"People Detected: {people_count}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Violations: {violations}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(window_name, output_frame)

    # PRESS Q TO EXIT

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()
cv2.destroyAllWindows()

# ==========================================
# CALCULATE AVERAGE PEOPLE
# ==========================================

if session_frames > 0:
    average_people = round(session_people_total / session_frames, 2)
else:
    average_people = 0

# ==========================================
# SAVE REPORT TO EXCEL
# ==========================================

excel_file = "live_tracking_report.xlsx"

new_data = {
    "Date": [today_date],
    "Max People": [max_people_detected],
    "Total Violations": [total_violations],
    "Average People": [average_people],
    "Sessions": [1]
}

new_df = pd.DataFrame(new_data)

# ==========================================
# SAVE / UPDATE EXCEL FILE
# ==========================================

try:

    # IF FILE EXISTS
    if os.path.exists(excel_file):

        old_df = pd.read_excel(excel_file)

        # CHECK IF TODAY DATE EXISTS
        if today_date in old_df["Date"].astype(str).values:

            row_index = old_df[
                old_df["Date"].astype(str) == today_date
            ].index[0]

            # UPDATE MAX PEOPLE
            old_df.loc[row_index, "Max People"] = max(
                old_df.loc[row_index, "Max People"],
                max_people_detected
            )

            # UPDATE VIOLATIONS
            old_df.loc[row_index, "Total Violations"] += total_violations

            # UPDATE AVERAGE PEOPLE
            old_average = old_df.loc[row_index, "Average People"]

            old_df.loc[row_index, "Average People"] = round(
                (old_average + average_people) / 2,
                2
            )

            # UPDATE SESSION COUNT
            old_df.loc[row_index, "Sessions"] += 1

        else:

            # ADD NEW DATE ROW
            old_df = pd.concat([old_df, new_df], ignore_index=True)

        # SAVE UPDATED FILE
        old_df.to_excel(excel_file, index=False)

    else:

        # CREATE NEW FILE
        new_df.to_excel(excel_file, index=False)

    print("\nExcel report updated successfully!")

except PermissionError:

    print("\n===================================")
    print("ERROR: EXCEL FILE IS OPEN")
    print("===================================")
    print("Please close 'live_tracking_report.xlsx'")
    print("and run the program again.")
    print("===================================")

# ==========================================
# AI SUMMARY
# ==========================================

print("\n===================================")
print("FINAL AI CROWD ANALYSIS REPORT")
print("===================================\n")

print(f"Date                     : {today_date}")
print(f"Maximum People Detected  : {max_people_detected}")
print(f"Total Violations         : {total_violations}")
print(f"Average People Detected  : {average_people}")

# ==========================================
# AI-STYLE SUMMARY
# ==========================================

print("\nAI Summary:\n")

if max_people_detected >= HIGH_CROWD_THRESHOLD:

    print(
        "High crowd density was detected in the monitored area. "
        "Multiple unsafe proximity events occurred between individuals. "
        "The environment appears congested and requires crowd management attention."
    )

elif total_violations > 0:

    print(
        "Moderate crowd activity detected. "
        "Several people were found standing too close to each other. "
        "Safety monitoring is recommended for better crowd control."
    )

else:

    print(
        "Crowd activity remained under safe limits. "
        "No major unsafe proximity events were detected."
    )

# ==========================================
# SAFETY ALERTS
# ==========================================

print("\nSafety Alert:\n")

if max_people_detected >= HIGH_CROWD_THRESHOLD:

    print(
        "- Increase crowd monitoring\n"
        "- Maintain safe movement space\n"
        "- Reduce congestion in crowded areas\n"
        "- Deploy additional safety supervision if required"
    )

elif total_violations > 0:

    print(
        "- Encourage safe distancing\n"
        "- Monitor crowd gathering points\n"
        "- Use warning indicators in crowded zones"
    )

else:

    print(
        "- Crowd conditions appear safe\n"
        "- Continue regular monitoring"
    )

print("\n===================================")
print("SYSTEM ANALYSIS COMPLETED")
print("===================================")