import subprocess
import cv2
import os

def extract_frames_with_ffmpeg(input_video_path, output_dir, interval, fps=60):
    """
    Extract frames from a specific interval in the video using FFmpeg.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_frame = interval[0]
    end_frame = interval[1]
    # FFmpeg command to extract frames within the specific interval
    ffmpeg_command = [
        'D:/Program Files/ffmpeg/bin/ffmpeg.exe',
        '-i', input_video_path,            # Input video file
        '-vf', f'fps={fps},select=between(n\\,{start_frame}\\,{end_frame})',  # Extract frames from start_frame to end_frame
        '-vsync', '0',                     # Disable frame rate syncing
        os.path.join(output_dir, 'frame_%04d.png')  # Output frames
    ]
    
    subprocess.run(ffmpeg_command, check=True)

def create_cropped_video(input_video_path, output_video_path, interval, crop_coords=None, fps=60):
    """
    Create a cropped video from a specific interval of the video.
    """
    # Step 1: Extract frames from video using FFmpeg
    output_dir = 'extracted_frames'
    extract_frames_with_ffmpeg(input_video_path, output_dir, interval=interval, fps=60)
    
    # Step 2: Process each frame (crop and re-encode)
    frame_files = sorted(os.listdir(output_dir))
    frames = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(output_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if crop_coords is not None:
            # Crop the frame based on the provided coordinates (xmin, xmax, ymin, ymax)
            xmin, xmax, ymin, ymax = crop_coords
            frame = frame[ymin:ymax, xmin:xmax]
        
        frames.append(frame)
    
    # Step 3: Create the output video using the cropped frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    
    # Cleanup extracted frames
    for frame_file in frame_files:
        os.remove(os.path.join(output_dir, frame_file))
    os.rmdir(output_dir)
    
    print(f"Cropped video saved at: {output_video_path}")

if __name__ == '__main__':
    input_video_path = input("Enter video file to be cropped: ").strip()
    output_video_path = 'output_cropped_video.mp4'
    start_frame = 60  # Example start frame for the interval
    end_frame = 230    # Example end frame for the interval
    crop_coords = (100, 600, 50, 450)  # Example crop coordinates (xmin, xmax, ymin, ymax)
    
    create_cropped_video(input_video_path, output_video_path, interval=[start_frame, end_frame], crop_coords=None)
