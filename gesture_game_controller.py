import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui


# -----------------------------
# Configuration
# -----------------------------
CAMERA_INDEX = 0
WINDOW_NAME = "Gesture Game Controller"
CALIBRATION_SECONDS = 2.0
SMOOTHING_WINDOW = 5
POSE_DET_CONF = 0.5
POSE_TRACK_CONF = 0.5

# Thresholds are normalized because MediaPipe landmarks are normalized.
LEFT_RIGHT_THRESHOLD = 0.08
DUCK_THRESHOLD = 0.10
JUMP_MARGIN = 0.03

# Cooldowns in seconds
COOLDOWN_LEFT_RIGHT = 0.45
COOLDOWN_JUMP = 0.65
COOLDOWN_DUCK = 0.65
GLOBAL_MIN_GAP = 0.20

# Safety: move mouse to a corner to trigger pyautogui fail-safe.
pyautogui.FAILSAFE = True


# -----------------------------
# Helpers
# -----------------------------
class Smoother:
    def __init__(self, maxlen: int = 5):
        self.values = deque(maxlen=maxlen)

    def update(self, value: float) -> float:
        self.values.append(value)
        return sum(self.values) / len(self.values)


class CalibrationState:
    def __init__(self):
        self.nose_x_values = []
        self.nose_y_values = []
        self.start_time = None
        self.center_nose_x = None
        self.center_nose_y = None
        self.done = False

    def start(self):
        self.start_time = time.time()

    def update(self, nose_x: float, nose_y: float):
        self.nose_x_values.append(nose_x)
        self.nose_y_values.append(nose_y)

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def finalize(self):
        if not self.nose_x_values or not self.nose_y_values:
            return False
        self.center_nose_x = sum(self.nose_x_values) / len(self.nose_x_values)
        self.center_nose_y = sum(self.nose_y_values) / len(self.nose_y_values)
        self.done = True
        return True


class ActionController:
    def __init__(self):
        now = 0.0
        self.last_global = now
        self.last_left = now
        self.last_right = now
        self.last_jump = now
        self.last_duck = now
        self.current_action_text = "NONE"
        self.current_action_until = 0.0

    def can_trigger(self, action: str, now: float) -> bool:
        if now - self.last_global < GLOBAL_MIN_GAP:
            return False

        if action == "LEFT":
            return now - self.last_left >= COOLDOWN_LEFT_RIGHT
        if action == "RIGHT":
            return now - self.last_right >= COOLDOWN_LEFT_RIGHT
        if action == "JUMP":
            return now - self.last_jump >= COOLDOWN_JUMP
        if action == "DUCK":
            return now - self.last_duck >= COOLDOWN_DUCK
        return False

    def trigger(self, action: str):
        now = time.time()
        if not self.can_trigger(action, now):
            return False

        if action == "LEFT":
            pyautogui.press("left")
            self.last_left = now
        elif action == "RIGHT":
            pyautogui.press("right")
            self.last_right = now
        elif action == "JUMP":
            pyautogui.press("up")
            self.last_jump = now
        elif action == "DUCK":
            pyautogui.press("down")
            self.last_duck = now
        else:
            return False

        self.last_global = now
        self.current_action_text = action
        self.current_action_until = now + 0.35
        print(f"[ACTION] {action}")
        return True

    def get_overlay_action(self) -> str:
        if time.time() <= self.current_action_until:
            return self.current_action_text
        return "NONE"


# -----------------------------
# Pose utilities
# -----------------------------
def get_landmark_xy(landmarks, idx: int):
    lm = landmarks[idx]
    return lm.x, lm.y


def draw_text_block(frame, lines, x=20, y=30, line_gap=28, color=(0, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def detect_action(landmarks, baseline_x: float, baseline_y: float):
    mp_pose = mp.solutions.pose

    nose_x, nose_y = get_landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE.value)
    left_wrist_x, left_wrist_y = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)
    right_wrist_x, right_wrist_y = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
    left_shoulder_x, left_shoulder_y = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder_x, right_shoulder_y = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

    # Derived signals
    nose_dx = nose_x - baseline_x
    nose_dy = nose_y - baseline_y
    head_level = min(left_shoulder_y, right_shoulder_y) - JUMP_MARGIN

    # Priority matters. Jump/duck first to reduce accidental lane switches.
    both_wrists_above_head = left_wrist_y < head_level and right_wrist_y < head_level
    if both_wrists_above_head:
        return "JUMP", nose_x, nose_y, nose_dx, nose_dy

    crouched = nose_dy > DUCK_THRESHOLD
    if crouched:
        return "DUCK", nose_x, nose_y, nose_dx, nose_dy

    if nose_dx < -LEFT_RIGHT_THRESHOLD:
        return "LEFT", nose_x, nose_y, nose_dx, nose_dy

    if nose_dx > LEFT_RIGHT_THRESHOLD:
        return "RIGHT", nose_x, nose_y, nose_dx, nose_dy

    return "NONE", nose_x, nose_y, nose_dx, nose_dy


# -----------------------------
# Main app
# -----------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=POSE_DET_CONF,
        min_tracking_confidence=POSE_TRACK_CONF,
    )

    x_smoother = Smoother(maxlen=SMOOTHING_WINDOW)
    y_smoother = Smoother(maxlen=SMOOTHING_WINDOW)
    calibration = CalibrationState()
    calibration.start()
    controller = ActionController()

    print("Starting gesture controller...")
    print("Controls:")
    print("- Lean head left  -> LEFT")
    print("- Lean head right -> RIGHT")
    print("- Raise both hands above head -> JUMP")
    print("- Crouch / move head down -> DUCK")
    print("Press 'q' to quit.")
    print("Tip: open your game first, then click back into the game window after calibration if needed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        action_text = controller.get_overlay_action()
        nose_dx = 0.0
        nose_dy = 0.0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2),
            )

            nose_x_raw, nose_y_raw = get_landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE.value)
            nose_x = x_smoother.update(nose_x_raw)
            nose_y = y_smoother.update(nose_y_raw)

            if not calibration.done:
                calibration.update(nose_x, nose_y)
                remaining = max(0.0, CALIBRATION_SECONDS - calibration.elapsed())
                draw_text_block(
                    frame,
                    [
                        "CALIBRATING... stand naturally and look at the screen",
                        f"Remaining: {remaining:.1f}s",
                    ],
                    color=(0, 255, 255),
                )
                if calibration.elapsed() >= CALIBRATION_SECONDS:
                    success = calibration.finalize()
                    if success:
                        print(
                            f"Calibration complete. Baseline nose=({calibration.center_nose_x:.3f}, {calibration.center_nose_y:.3f})"
                        )
                    else:
                        calibration = CalibrationState()
                        calibration.start()
                        print("Calibration failed, retrying...")
            else:
                # Build a temporary landmarks structure substitute using smoothed nose only.
                # We keep the original landmarks for wrists/shoulders and override nose for stability.
                action, _, _, nose_dx, nose_dy = detect_action(
                    landmarks,
                    calibration.center_nose_x,
                    calibration.center_nose_y,
                )

                # Override overlay text even if cooldown blocks actual press so user sees intent.
                action_text = action

                if action != "NONE":
                    controller.trigger(action)

                # Draw baseline guide lines
                h, w, _ = frame.shape
                center_x_px = int(calibration.center_nose_x * w)
                center_y_px = int(calibration.center_nose_y * h)
                left_line = int((calibration.center_nose_x - LEFT_RIGHT_THRESHOLD) * w)
                right_line = int((calibration.center_nose_x + LEFT_RIGHT_THRESHOLD) * w)
                duck_line = int((calibration.center_nose_y + DUCK_THRESHOLD) * h)

                cv2.line(frame, (center_x_px, 0), (center_x_px, h), (255, 255, 0), 1)
                cv2.line(frame, (left_line, 0), (left_line, h), (0, 255, 255), 1)
                cv2.line(frame, (right_line, 0), (right_line, h), (0, 255, 255), 1)
                cv2.line(frame, (0, center_y_px), (w, center_y_px), (255, 255, 0), 1)
                cv2.line(frame, (0, duck_line), (w, duck_line), (0, 165, 255), 1)
        else:
            if not calibration.done:
                draw_text_block(
                    frame,
                    ["CALIBRATING... no person detected", "Stand clearly in frame"],
                    color=(0, 0, 255),
                )
            else:
                draw_text_block(frame, ["No pose detected"], color=(0, 0, 255))

        status_lines = [
            f"Action: {controller.get_overlay_action() if calibration.done else 'WAIT'}",
            f"dx: {nose_dx:+.3f}   dy: {nose_dy:+.3f}",
            "Lean left/right | Both hands up = jump | Crouch = duck",
            "Press q to quit",
        ]
        draw_text_block(frame, status_lines, x=20, y=100 if not calibration.done else 30)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            calibration = CalibrationState()
            calibration.start()
            x_smoother = Smoother(maxlen=SMOOTHING_WINDOW)
            y_smoother = Smoother(maxlen=SMOOTHING_WINDOW)
            print("Recalibrating...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
