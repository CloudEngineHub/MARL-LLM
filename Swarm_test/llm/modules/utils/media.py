"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

import os
import re
import cv2
import base64

import numpy as np


def generate_video_from_frames(frames_folder, video_path, fps=100):
    from modules.file import logger

    logger.log(f"Generating video from frames in {frames_folder}...")
    try:
        frame_files = sorted(
            [file for file in os.listdir(frames_folder) if re.search(r"\d+", file)],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
    except Exception as e:
        logger.log(f"Error reading frames: {e}", level="error")
        return

    if not frame_files:
        logger.log("No frames found", level="error")
        return
    frame_files = [os.path.join(frames_folder, file) for file in frame_files]

    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))

    cv2.destroyAllWindows()
    video.release()
    logger.log(f"Video generated: {video_path}", level="info")


def process_video(video_path, seconds_per_frame=2, start_time=0, end_time=None):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # 视频总时长，单位为秒

    # 如果没有指定 end_time，则默认使用视频的总时长
    if end_time is None:
        end_time = video_duration

    # Convert start_time and end_time to frames
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames)  # 确保 end_frame 不超过总帧数

    # Calculate the number of frames to skip
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = start_frame

    # Loop through the video and extract frames at the specified sampling rate
    while curr_frame < end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    # Ensure the last frame is included if end_frame is not captured by the loop
    if curr_frame != end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, end_frame - 1)
        success, frame = video.read()
        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    # Importing the logger
    from modules.file import logger

    logger.log(
        f"Extracted {len(base64Frames)} frames from {start_time}s to {end_time}s",
        level="info",
    )
    return base64Frames


def create_video_from_frames(base64Frames, output_path, fps=30):
    frames = []
    for base64_frame in base64Frames:
        frame_data = base64.b64decode(base64_frame)
        np_frame = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        frames.append(frame)
    if not frames:
        print("No frames to write to video")
        return
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify video codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    generate_video_from_frames(
        "/home/derrick/catkin_ws/src/code_llm/workspace/2024-06-24_06-02-10_搬运/data/frames/frame15",
        "output.mp4",
    )
