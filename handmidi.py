import sys
import cv2
import numpy as np
import mediapipe as mp
import rtmidi
from threading import Thread, Lock

# Class that represents the video stream to be started in separate thread
# Inspired by : https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56
class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240):
        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started :
            print("Video stream already started.")
            return None

        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

        return self

    def update(self):
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
 
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()

# Normalize x to MIDI 0-127 range
# formula : normalized_x = (b - a) * ((x - min(x)) / (max(x) - min(x))) + a
def normalize_to_midi(val, minval, maxval):
    return max(min(int(127 * ((val - minval) / (maxval - minval))), 127), 0)

def main():
    # set video capture device index (0 is first one listed)
    device_index = 0

    # array of xyz coordinates (hand landmarks)
    points = [np.array([0, 0, 0]) for k in range(21)]

    count = 0

    # parse script arguments, set flags and values accordingly
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.lower() == "--deviceindex":
            try:
                device_index = int(args[i + 1])
            except:
                raise ValueError("For manual capture device index setting, use the --deviceindex argument followed by the desired capture device index.")

    # set openCV capture device
    vs = WebcamVideoStream(src=device_index).start()

    # initialize mediapipe hand scanning
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # get MIDI out ports, use first one found
    midi_out = rtmidi.MidiOut()
    available_ports = midi_out.get_ports()
    
    if available_ports:
        midi_out.open_port(0)
    else:
        midi_out.open_virtual_port("rtmidi Virtual MIDI Port")

    with midi_out:
        while True:
            img = vs.read()

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            # analyze hand landmarks
            if results.multi_hand_landmarks:
                # 21 landmarks total (0 is wrist)
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        # save coordinates
                        points[id] = np.array([lm.x, lm.y, lm.z])

                        # draw hand tracking on capture
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 3, (120, 100, 120), cv2.FILLED)

                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                        mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                        mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            # pinch
            pinch = np.linalg.norm(points[8] - points[4])
            # print(pinch)

            pinch_midi = [0XB0, 1, normalize_to_midi(pinch, 0.16, 0.04)]
            print(pinch_midi)

            # closed hand
            distances_to_wrist = [
                np.linalg.norm(points[8] - points[0]), # index
                np.linalg.norm(points[12] - points[0]), # middle
                np.linalg.norm(points[16] - points[0]), # ring
                np.linalg.norm(points[20] - points[0]) # pinky
            ]
            # print(distances_to_wrist)

            wrist_dist_avg = np.average(distances_to_wrist)
            # print(wrist_dist_avg)

            closed_hand_midi = [0XB0, 2, normalize_to_midi(wrist_dist_avg, 0.48, 0.14)]
            print(closed_hand_midi)
            
            # hand tilt
            # use landmarks at base of fingers for less impact of closed hand
            hand_tilt = np.average([points[2][2], points[5][2], points[9][2], points[13][2], points[17][2]])
            # print(hand_tilt)

            hand_tilt_midi = [0XB0, 3, normalize_to_midi(hand_tilt, -0.23, 0.2)]
            print(hand_tilt_midi)

            # send MIDI messages
            midi_out.send_message(pinch_midi)
            midi_out.send_message(closed_hand_midi)
            midi_out.send_message(hand_tilt_midi)

            cv2.imshow("HANDMIDI", img)
            if cv2.waitKey(1) == ord("q"):
                break

        vs.stop()
        cv2.destroyAllWindows()
        del midi_out

if __name__ == '__main__':
    main()

## CONTROLS : 
# pinch index and thumb : CC1
# close hand without thumb : CC2
# hand tilt : CC3