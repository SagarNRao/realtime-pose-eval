import cv2
import numpy as np
import mediapipe as mp
import time, os, glob

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
L = mp_pose.PoseLandmark

REF_DIR = "exercises"          # put squat.mp4, pushup.mp4, curl.mp4 ... here
CACHE_DIR = "reference_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

JOINTS = {  # a, b, c landmarks -> angle at b
    "L_ELBOW": (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
    "R_ELBOW": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    "L_KNEE":  (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
    "R_KNEE":  (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
    "L_HIP":   (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE),
    "R_HIP":   (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE),
}
SEGMENTS = [
    (L.LEFT_SHOULDER, L.LEFT_ELBOW, "L_ELBOW"), (L.LEFT_ELBOW, L.LEFT_WRIST, "L_ELBOW"),
    (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, "R_ELBOW"), (L.RIGHT_ELBOW, L.RIGHT_WRIST, "R_ELBOW"),
    (L.LEFT_HIP, L.LEFT_KNEE, "L_KNEE"), (L.LEFT_KNEE, L.LEFT_ANKLE, "L_KNEE"),
    (L.RIGHT_HIP, L.RIGHT_KNEE, "R_KNEE"), (L.RIGHT_KNEE, L.RIGHT_ANKLE, "R_KNEE"),
    (L.LEFT_SHOULDER, L.LEFT_HIP, "L_HIP"), (L.RIGHT_SHOULDER, L.RIGHT_HIP, "R_HIP"),
    (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, None), (L.LEFT_HIP, L.RIGHT_HIP, None),
]
ERROR_THRESH = 15  # degrees


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ang = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang


def normalize(lm):
    """Return 33x2 landmark array centered on mid-hip, scaled by torso length (similarity transform)."""
    pts = np.array([[p.x, p.y] for p in lm])
    mid_hip = (pts[L.LEFT_HIP.value] + pts[L.RIGHT_HIP.value]) / 2
    mid_sh = (pts[L.LEFT_SHOULDER.value] + pts[L.RIGHT_SHOULDER.value]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip)
    torso = torso if torso > 1e-6 else 1e-6
    return (pts - mid_hip) / torso, mid_hip, torso


def extract_reference(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(CACHE_DIR, name + ".npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5) as p:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = p.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                norm, _, _ = normalize(res.pose_landmarks.landmark)
                frames.append(norm)
    cap.release()
    arr = np.array(frames)
    if len(arr):
        np.save(cache_path, arr)
    return arr


def load_exercises():
    exercises = {}
    for path in sorted(glob.glob(os.path.join(REF_DIR, "*.*"))):
        name = os.path.splitext(os.path.basename(path))[0]
        arr = extract_reference(path)
        if len(arr):
            exercises[name] = arr
    return exercises


def ref_angles(norm_frame):
    return {k: calculate_angle(norm_frame[a.value], norm_frame[b.value], norm_frame[c.value])
            for k, (a, b, c) in JOINTS.items()}


exercises = load_exercises()
names = list(exercises.keys())
if not names:
    print(f"No reference videos found in ./{REF_DIR}/ (e.g. squat.mp4). Add some and rerun.")
    raise SystemExit

idx = 0
ref_frame_i = 0
cap = cv2.VideoCapture(0)
prev_t = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    ex_name = names[idx]
    ref_seq = exercises[ex_name]
    ref_norm = ref_seq[ref_frame_i % len(ref_seq)]
    ref_frame_i += 1
    r_angles = ref_angles(ref_norm)

    status_ok = True
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        u_norm, mid_hip, torso = normalize(lm)
        mid_hip_px = mid_hip * [w, h]

        # project reference skeleton into user's frame (same position/scale as user)
        ref_px = ref_norm * torso * [w, h] + mid_hip_px

        u_angles = {k: calculate_angle([lm[a.value].x*w, lm[a.value].y*h],
                                        [lm[b.value].x*w, lm[b.value].y*h],
                                        [lm[c.value].x*w, lm[c.value].y*h])
                    for k, (a, b, c) in JOINTS.items()}

        for p1, p2, jkey in SEGMENTS:
            bp1, bp2 = tuple(np.int32(ref_px[p1.value])), tuple(np.int32(ref_px[p2.value]))
            cv2.line(frame, bp1, bp2, (255, 120, 0), 2)  # blue ideal-form overlay

            up1 = tuple(np.int32([lm[p1.value].x*w, lm[p1.value].y*h]))
            up2 = tuple(np.int32([lm[p2.value].x*w, lm[p2.value].y*h]))
            if jkey is None:
                color = (255, 255, 255)
            else:
                diff = abs(u_angles[jkey] - r_angles[jkey])
                color = (0, 255, 0) if diff <= ERROR_THRESH else (0, 0, 255)
                if diff > ERROR_THRESH:
                    status_ok = False
            cv2.line(frame, up1, up2, color, 3)
            cv2.circle(frame, up1, 5, (0, 255, 255), -1)
            cv2.circle(frame, up2, 5, (0, 255, 255), -1)

        tag = "FORM MATCH" if status_ok else "INCORRECT FORM"
        cv2.putText(frame, tag, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if status_ok else (0, 0, 255), 3)

    curr_t = time.time()
    fps = 1 / (curr_t - prev_t) if curr_t != prev_t else 0
    prev_t = curr_t
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Exercise: {ex_name}  [n/p to switch]", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Workout Form Evaluator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        idx = (idx + 1) % len(names); ref_frame_i = 0
    elif key == ord('p'):
        idx = (idx - 1) % len(names); ref_frame_i = 0

cap.release()
cv2.destroyAllWindows()