import cv2
import os

video_number = 6
clip_number = 3
video_name = f"video{video_number}/{clip_number}"

print(video_name)

# set up video capture
cap = cv2.VideoCapture(f"video_data/{video_name}.mp4")
frame_rate = 1# Save every frame
count = 0
output_dir = f"extracted_frames/video_{video_number}_{clip_number}_every_{frame_rate}/"

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
        cv2.imwrite(f"{output_dir}/frame_{count*frame_rate:03d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        count += 1
cap.release()

