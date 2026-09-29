import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

mask_path = r"D:\Computer Vision\Parking System\parking\mask_1920_1080.png"
video_path = r"D:\Computer Vision\Parking System\parking\parking_1920_1080.mp4"
output_path = r"D:\Computer Vision\Parking System\parking\parking_output.mp4"

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

# --- VideoWriter setup ---
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# -------------------------

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]
previous_frame = None
frame_nmr = 0
ret = True
step = 90

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w, :])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in range(len(spots)) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_index in arr_:
            x1, y1, w, h = spots[spot_index]
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            status = empty_or_not(spot_crop)
            spots_status[spot_index] = status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_index, spot in enumerate(spots):
        x1, y1, w, h = spot
        status = spots_status[spot_index]
        if status is None:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (128, 128, 128), 2)
        elif status:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available Spots:{}/{}'.format(str(sum(spots_status)), str(len(spots_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)  # <-- save frame to output video

    cv2.namedWindow('Parking Video', cv2.WINDOW_NORMAL)
    cv2.imshow("Parking Video", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    frame_nmr += 1

cap.release()
out.release()  # <-- finalize the file
cv2.destroyAllWindows()
print(f"Output saved to: {output_path}")