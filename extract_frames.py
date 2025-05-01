import cv2
import os

# change as per input video
video_number = 1

# set up video capture
cap = cv2.VideoCapture(f"video_data/video{video_number}/real_00{video_number}.mp4")
frame_rate = 1 # Save every frame
count = 0
output_dir = f"extracted_frames/video{video_number}/"

os.makedirs(output_dir, exist_ok=True)
# loop through video frames and save them
while True:
    ret, frame = cap.read()

    if not ret:
        break
    if int(cap.get(1)) % frame_rate == 0:
        # cv2.IMWRITE_JPEG_QUALITY ensures highest possible quality (100)
        # print status
        print(f"Extracting frame {count} from video {video_number}")
        cv2.imwrite(f"{output_dir}/frame_{count:03d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        count += 1
cap.release()

