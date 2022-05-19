import cv2
import os
import numpy as np
import argparse

SMALLEST_DIM = 256
IMAGE_CROP_SIZE = 224
FRAME_RATE = 25

# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def main(args):
    if not os.path.exists(args.video_path):
        print("This file does not exist")
        return

    # sample all video from video_path at specified frame rate (FRAME_RATE param)
    sample_video = load_video(args.video_path)

    # sample_video = np.expand_dims(sample_video, axis=0)

    video_name = args.video_path.split("/")[-1][:-4]
    npy_rgb_output = 'data/' + video_name + '_rgb.npy'
    np.save(npy_rgb_output, sample_video)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default="data/VID_NaderRun.mp4")

    args = parser.parse_args()

    main(args)
  