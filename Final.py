import cv2
import numpy as np
import time
import os
import pyautogui
from pynput.keyboard import Controller, Key as PynputKey
import mediapipe as mp

# --------------------- Hand Tracking Classes ---------------------
class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def getHandedness(self, hand_index=0):
        if self.results.multi_handedness:
            return self.results.multi_handedness[hand_index].classification[0].label
        return None

# --------------------- Virtual Keyboard Classes ---------------------
class Key:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def drawKey(self, img, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        bg_rec = img[self.y:self.y + self.h, self.x:self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = res
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)[0]
        text_pos = (self.x + self.w // 2 - text_size[0] // 2, self.y + self.h // 2 + text_size[1] // 2)
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h

# --------------------- Gesture Functions ---------------------
def fingers_up(landmarks):
    fingers = []
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if landmarks[tip][2] < landmarks[tip - 2][2] else 0)
    return fingers

# --------------------- Initialization ---------------------
cap = cv2.VideoCapture(0)
tracker = HandTracker()
keyboard = Controller()

if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Keyboard layout setup
w, h = 80, 60
startX, startY = 40, 200
is_special = False
click_time = 0

numbers = list("1234567890")
letters = list("QWERTYUIOPASDFGHJKLZXCVBNM")
special_chars = list("!@#$%^&*()_+-=[]{}|;:',.<>?/\\")
control_keys = ["Space", "clr", "<--", "Enter", "Switch"]

keys = []

def generate_keys():
    global keys
    keys = []
    for i, n in enumerate(numbers):
        keys.append(Key(startX + i * w + i * 5, startY - h - 5, w, h, n))
    current_keys = letters if not is_special else special_chars
    for i, l in enumerate(current_keys):
        row = i // 10
        col = i % 10
        keys.append(Key(startX + col * w + col * 5, startY + row * (h + 5), w, h, l))
    keys += [
        Key(startX, startY + 3 * h + 15, 2 * w, h, "Switch"),
        Key(startX + 2 * w + 10, startY + 3 * h + 15, 3 * w, h, "Space"),
        Key(startX + 5 * w + 30, startY + 3 * h + 15, 2 * w, h, "<--"),
        Key(startX + 7 * w + 50, startY + 3 * h + 15, 2 * w, h, "clr"),
        Key(startX + 9 * w + 70, startY + 3 * h + 15, 2 * w, h, "Enter")
    ]

generate_keys()
textBox = Key(startX, startY - 2 * h - 10, 10 * w + 9 * 5, h, '')

last_action_time = {k: 0 for k in ["right_click", "left_click", "open_windows_search", "maximize_minimize", "switch_apps", "take_screenshot"]}
delay_thresholds = {"right_click": 3, "left_click": 3, "open_windows_search": 4, "maximize_minimize": 5, "switch_apps": 1, "take_screenshot": 8}

# --------------------- Main Loop ---------------------
ptime = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    frame = cv2.flip(frame, 1)
    tracker.findHands(frame)
    
    for i, _ in enumerate(tracker.results.multi_hand_landmarks or []):
        hand_type = tracker.getHandedness(i)
        lmList = tracker.getPosition(frame, i, draw=False)

        if not lmList:
            continue

        index_up, middle_up, ring_up, little_up = fingers_up(lmList)
        thumb_up = lmList[4][1] < lmList[3][1]

        if hand_type == "Right":
            # Gesture-based mouse/system control
            screen_x, screen_y = pyautogui.size()
            if index_up and not any([middle_up, ring_up, little_up]):
                x = int(lmList[8][1] / 960 * screen_x)
                y = int(lmList[8][2] / 540 * screen_y)
                pyautogui.moveTo(x, y)
            elif index_up and middle_up and not any([ring_up, little_up]):
                pyautogui.scroll(-10 if lmList[8][2] < lmList[12][2] else 10)
            elif index_up and middle_up and ring_up and not any([thumb_up, little_up]):
                if time.time() - last_action_time["right_click"] > delay_thresholds["right_click"]:
                    pyautogui.rightClick()
                    last_action_time["right_click"] = time.time()
            elif thumb_up and little_up and not any([index_up, middle_up, ring_up]):
                if time.time() - last_action_time["left_click"] > delay_thresholds["left_click"]:
                    pyautogui.click()
                    last_action_time["left_click"] = time.time()
            elif all([index_up, middle_up, ring_up, little_up, thumb_up]):
                if time.time() - last_action_time["open_windows_search"] > delay_thresholds["open_windows_search"]:
                    pyautogui.hotkey("win")
                    last_action_time["open_windows_search"] = time.time()
            elif little_up and not any([index_up, middle_up, ring_up, thumb_up]):
                if time.time() - last_action_time["maximize_minimize"] > delay_thresholds["maximize_minimize"]:
                    pyautogui.hotkey("win", "down")
                    last_action_time["maximize_minimize"] = time.time()
            elif all([index_up, middle_up, ring_up, little_up]) and not thumb_up:
                if time.time() - last_action_time["switch_apps"] > delay_thresholds["switch_apps"]:
                    pyautogui.hotkey("alt", "tab")  
                    last_action_time["switch_apps"] = time.time()
            elif thumb_up and not any([index_up, middle_up, ring_up, little_up]):
                if time.time() - last_action_time["take_screenshot"] > delay_thresholds["take_screenshot"]:
                    path = f"screenshots/screenshot_{int(time.time())}.png"
                    pyautogui.screenshot(path)
                    print(f"Screenshot saved: {path}")
                    last_action_time["take_screenshot"] = time.time()

        elif hand_type == "Left":
            # Virtual keyboard
            indexTip = lmList[8][1:3]
            thumbTip = lmList[4][1:3]
            for key in keys:
                key.drawKey(frame, alpha=0.1 if key.isOver(*indexTip) else 0.5)
                if key.isOver(*indexTip) and key.isOver(*thumbTip):
                    if time.time() - click_time > 1:
                        if key.text == "Space":
                            textBox.text += " "
                            keyboard.press(PynputKey.space)
                            keyboard.release(PynputKey.space)
                        elif key.text == "<--":
                            textBox.text = textBox.text[:-1]
                            keyboard.press(PynputKey.backspace)
                            keyboard.release(PynputKey.backspace)
                        elif key.text == "clr":
                            textBox.text = ""
                        elif key.text == "Enter":
                            textBox.text += "\n"
                            keyboard.press(PynputKey.enter)
                            keyboard.release(PynputKey.enter)
                        elif key.text == "Switch":
                            is_special = not is_special
                            generate_keys()
                        else:
                            textBox.text += key.text
                            keyboard.press(key.text)
                            keyboard.release(key.text)
                        click_time = time.time()

    textBox.drawKey(frame, (255, 255, 255), (0, 0, 0), alpha=0.3)
    fps = int(1 / (time.time() - ptime + 1e-5))
    ptime = time.time()
    cv2.putText(frame, f"{fps} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Gesture Control + Virtual Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
