from high_res_stitcher import HighResStitcher

def main():
    # Initialize stitcher
    stitcher = HighResStitcher()

    # List of input video paths
    video_paths = ["video1.mp4", "video2.mp4"]

    # Function call order
    stitcher.load_videos(video_paths)
    stitcher.extract_key_frames(overlap_threshold=0.5)
    stitcher.detect_features()
    stitcher.match_features()
    stitcher.estimate_transforms()
    stitcher.warp_and_stitch()
    stitcher.blend_images()
    stitcher.save_output("output_high_res.png")
    # Optional: Launch viewer to inspect result
    stitcher.launch_viewer()

if __name__ == "__main__":
    main()
