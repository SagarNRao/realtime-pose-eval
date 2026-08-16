# Workout Form Correction System using MediaPipe and OpenCV
# This notebook detects user pose and compares it with ideal exercise forms

# Install required packages (run once)
# !pip install mediapipe opencv-python numpy

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define exercise reference poses (normalized keypoint positions)
# Format: {landmark_name: (x, y)} - normalized coordinates

EXERCISES = {
    "dumbbell_overhead_press": {
        "name": "Dumbbell Overhead Press",
        "keyframes": [
            # Starting position - dumbbells at shoulder level
            {
                "left_shoulder": (0.35, 0.35),
                "right_shoulder": (0.65, 0.35),
                "left_elbow": (0.30, 0.50),
                "right_elbow": (0.70, 0.50),
                "left_wrist": (0.28, 0.45),
                "right_wrist": (0.72, 0.45),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            },
            # Extended position - arms overhead
            {
                "left_shoulder": (0.35, 0.35),
                "right_shoulder": (0.65, 0.35),
                "left_elbow": (0.33, 0.20),
                "right_elbow": (0.67, 0.20),
                "left_wrist": (0.35, 0.10),
                "right_wrist": (0.65, 0.10),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            }
        ]
    },
    "dumbbell_curl": {
        "name": "Dumbbell Bicep Curl",
        "keyframes": [
            # Starting position - arms extended
            {
                "left_shoulder": (0.35, 0.35),
                "right_shoulder": (0.65, 0.35),
                "left_elbow": (0.32, 0.50),
                "right_elbow": (0.68, 0.50),
                "left_wrist": (0.30, 0.65),
                "right_wrist": (0.70, 0.65),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            },
            # Curled position - weights at shoulders
            {
                "left_shoulder": (0.35, 0.35),
                "right_shoulder": (0.65, 0.35),
                "left_elbow": (0.32, 0.50),
                "right_elbow": (0.68, 0.50),
                "left_wrist": (0.30, 0.35),
                "right_wrist": (0.70, 0.35),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            }
        ]
    },
    "overhead_tricep_extension": {
        "name": "Overhead Tricep Extension",
        "keyframes": [
            # Starting position - arms overhead
            {
                "left_shoulder": (0.35, 0.30),
                "right_shoulder": (0.65, 0.30),
                "left_elbow": (0.40, 0.20),
                "right_elbow": (0.60, 0.20),
                "left_wrist": (0.50, 0.10),
                "right_wrist": (0.50, 0.10),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            },
            # Bent position - elbows bent behind head
            {
                "left_shoulder": (0.35, 0.30),
                "right_shoulder": (0.65, 0.30),
                "left_elbow": (0.40, 0.20),
                "right_elbow": (0.60, 0.20),
                "left_wrist": (0.50, 0.35),
                "right_wrist": (0.50, 0.35),
                "left_hip": (0.40, 0.65),
                "right_hip": (0.60, 0.65),
            }
        ]
    }
}

# MediaPipe landmark indices
LANDMARK_MAP = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
}

# Connection pairs for drawing skeleton
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def normalize_pose(landmarks):
    """Normalize pose landmarks relative to hip center and shoulder width"""
    if not landmarks:
        return None
    
    # Get hip center
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    
    # Get shoulder width for scaling
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width = abs(right_shoulder.x - left_shoulder.x)
    
    if shoulder_width < 0.01:
        shoulder_width = 0.2
    
    # Normalize all landmarks
    normalized = {}
    for name, landmark_idx in LANDMARK_MAP.items():
        lm = landmarks[landmark_idx.value]
        norm_x = 0.5 + (lm.x - hip_center_x) / shoulder_width
        norm_y = (lm.y - hip_center_y) / shoulder_width + 0.5
        normalized[name] = (norm_x, norm_y)
    
    return normalized

def compare_poses(user_pose, reference_pose, threshold=0.15):
    """
    Compare user pose with reference pose
    Returns dict with body part: is_correct (bool)
    """
    if not user_pose:
        return {part: False for part in reference_pose.keys()}
    
    correctness = {}
    for part_name, ref_pos in reference_pose.items():
        if part_name in user_pose:
            user_pos = user_pose[part_name]
            distance = calculate_distance(user_pos, ref_pos)
            correctness[part_name] = distance < threshold
        else:
            correctness[part_name] = False
    
    return correctness

def draw_skeleton(frame, pose_dict, correctness, color_correct=(0, 255, 0), 
                  color_incorrect=(0, 0, 255), color_default=(255, 0, 0), radius=8):
    """Draw skeleton on frame with color coding based on correctness"""
    h, w = frame.shape[:2]
    
    # Draw connections
    for part1, part2 in CONNECTIONS:
        if part1 in pose_dict and part2 in pose_dict:
            pt1 = (int(pose_dict[part1][0] * w), int(pose_dict[part1][1] * h))
            pt2 = (int(pose_dict[part2][0] * w), int(pose_dict[part2][1] * h))
            
            # Determine line color based on both points
            if correctness:
                if correctness.get(part1, False) and correctness.get(part2, False):
                    color = color_correct
                else:
                    color = color_incorrect
            else:
                color = color_default
            
            cv2.line(frame, pt1, pt2, color, 3)
    
    # Draw joints
    for part_name, pos in pose_dict.items():
        pt = (int(pos[0] * w), int(pos[1] * h))
        
        if correctness:
            color = color_correct if correctness.get(part_name, False) else color_incorrect
        else:
            color = color_default
        
        cv2.circle(frame, pt, radius, color, -1)
        cv2.circle(frame, pt, radius, (255, 255, 255), 2)

def animate_reference_pose(exercise_data, frame_count, fps=30, cycle_duration=3.0):
    """Animate between keyframes of reference pose"""
    keyframes = exercise_data["keyframes"]
    num_keyframes = len(keyframes)
    
    # Calculate animation progress
    frames_per_cycle = int(fps * cycle_duration)
    cycle_position = (frame_count % frames_per_cycle) / frames_per_cycle
    
    # Determine which keyframes to interpolate between
    segment = cycle_position * (num_keyframes * 2 - 2)
    
    if segment < num_keyframes - 1:
        # Forward animation
        idx1 = int(segment)
        idx2 = idx1 + 1
        t = segment - idx1
    else:
        # Backward animation
        segment -= (num_keyframes - 1)
        idx1 = int(num_keyframes - 1 - segment)
        idx2 = idx1 - 1
        if idx2 < 0:
            idx2 = 0
        t = segment - int(segment)
    
    # Interpolate between keyframes
    kf1 = keyframes[idx1]
    kf2 = keyframes[idx2]
    
    interpolated = {}
    for part_name in kf1.keys():
        x1, y1 = kf1[part_name]
        x2, y2 = kf2[part_name]
        interpolated[part_name] = (
            x1 + (x2 - x1) * t,
            y1 + (y2 - y1) * t
        )
    
    return interpolated

def main():
    """Main function to run the workout form correction system"""
    # Select exercise
    print("Available Exercises:")
    exercise_list = list(EXERCISES.keys())
    for i, ex_key in enumerate(exercise_list):
        print(f"{i + 1}. {EXERCISES[ex_key]['name']}")
    
    choice = input("\nSelect exercise (1-3): ")
    try:
        exercise_key = exercise_list[int(choice) - 1]
        exercise_data = EXERCISES[exercise_key]
        print(f"\nStarting: {exercise_data['name']}")
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Dumbbell Overhead Press")
        exercise_key = "dumbbell_overhead_press"
        exercise_data = EXERCISES[exercise_key]
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_count = 0
    fps = 30
    
    print("\nInstructions:")
    print("- Match your pose (RED/GREEN) to the reference pose (BLUE)")
    print("- GREEN parts indicate correct form")
    print("- RED parts need adjustment")
    print("- Press 'q' to quit")
    print("- Press '1', '2', or '3' to switch exercises")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Create black canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get animated reference pose
        reference_pose = animate_reference_pose(exercise_data, frame_count, fps)
        
        # Draw reference pose in BLUE
        draw_skeleton(canvas, reference_pose, None, color_default=(255, 0, 0), radius=10)
        
        # Process user pose
        if results.pose_landmarks:
            user_pose = normalize_pose(results.pose_landmarks.landmark)
            
            if user_pose:
                # Compare poses
                correctness = compare_poses(user_pose, reference_pose, threshold=0.12)
                
                # Draw user pose with color coding (GREEN=correct, RED=incorrect)
                draw_skeleton(canvas, user_pose, correctness, 
                            color_correct=(0, 255, 0), color_incorrect=(0, 0, 255), radius=8)
                
                # Calculate overall accuracy
                correct_count = sum(correctness.values())
                total_count = len(correctness)
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
                
                # Display accuracy
                cv2.putText(canvas, f"Form Accuracy: {accuracy:.0f}%", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display exercise name
        cv2.putText(canvas, exercise_data['name'], 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display legend
        cv2.putText(canvas, "BLUE: Target | GREEN: Correct | RED: Adjust", 
                   (w - 520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Workout Form Correction', canvas)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            idx = int(chr(key)) - 1
            if idx < len(exercise_list):
                exercise_key = exercise_list[idx]
                exercise_data = EXERCISES[exercise_key]
                print(f"\nSwitched to: {exercise_data['name']}")
                frame_count = 0
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\nWorkout session ended!")

# Run the application
if __name__ == "__main__":
    main()