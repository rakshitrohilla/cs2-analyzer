import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames, fps