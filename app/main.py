import logging
import os
import queue
import threading
import re
import subprocess
from datetime import datetime, timezone
from fractions import Fraction

import av
import make87
from make87_messages.core.header_pb2 import Header
from make87_messages.file.simple_file_pb2 import RelativePathFile
from make87_messages.primitive.bool_pb2 import Bool
from make87_messages.video.any_pb2 import FrameAny

# Thread‑safe queue for files ready to be uploaded.
upload_queue: queue.Queue[str] = queue.Queue()


def get_new_filename(timestamp: datetime) -> str:
    """Generate a filename using a timestamp with millisecond and timezone details."""
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    return f"{timestamp:%Y%m%d_%H%M%S}_{timestamp.microsecond // 1000:03d}_{timestamp:%z}.mp4"


def run_ffmpeg(chunk_duration_sec: int, codec: str):
    """
    Start an FFmpeg process configured for segmentation.

    If frame_rate is provided, the '-r' option is inserted (before the input)
    so that ffmpeg assumes a constant frame rate.
    """
    # Map our codec to ffmpeg parameters.
    if codec == "h264":
        tag = "avc1"
    elif codec == "hevc":
        tag = "hvc1"
    elif codec == "av1":
        tag = "av01"
    else:
        raise ValueError("Unsupported codec: " + codec)

    ffmpeg_cmd = ["ffmpeg"]
    ffmpeg_cmd.extend(
        [
            "-f",
            "mpegts",
            "-i",
            "pipe:0",  # Read from stdin
            "-c:v",
            "copy",  # Copy codec without re-encoding
            "-movflags",
            "+faststart",
            "-tag:v",
            tag,  # Set the proper tag
            "-f",
            "segment",  # Use ffmpeg's built‑in segmenter
            "-segment_time",
            str(chunk_duration_sec),
            "-reset_timestamps",
            "1",  # Each segment starts at 0
            "-loglevel",
            "verbose",  # needs to stay to detect written segments and upload. Do not remove.
            "-progress",
            "pipe:1",  # Write progress info to stdout
            "output%03d.mp4",  # Output filename pattern (temporary name)
        ]
    )

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,  # Write raw data to ffmpeg
        stdout=subprocess.PIPE,  # Read progress/segmentation logs
        stderr=subprocess.STDOUT,
        bufsize=0,  # Unbuffered output
    )
    logging.info("Started ffmpeg process with command: " + " ".join(ffmpeg_cmd))
    return process


def ffmpeg_monitor_thread(process):
    """
    Monitor ffmpeg's output to detect when a segment has completed.
    When a segment ends, rename it using the current timestamp and enqueue it.
    """
    while True:
        output_line = process.stdout.readline()
        if not output_line:
            break  # ffmpeg process ended
        line = output_line.decode("utf-8", errors="ignore")
        # logging.error(f"FFmpeg Output: {line}")  # Log errors!
        # Detect segment completion (adjust the regex if your ffmpeg log format differs)
        if "ended" in line:
            match = re.search(r"segment:'(.+?\.mp4)'", line)
            if match:
                temp_filename = match.group(1)
                new_filename = get_new_filename(datetime.now())
                try:
                    os.rename(temp_filename, new_filename)
                    logging.info(f"Segment renamed from {temp_filename} to {new_filename}")
                    upload_queue.put(new_filename)
                except Exception as e:
                    logging.error(f"Error renaming file {temp_filename}: {e}")
    process.wait()


def recorder_worker_ffmpeg(chunk_duration_sec: int):
    """
    Recorder callback that buffers the first few frames to estimate a fixed frame rate,
    then starts an ffmpeg process with that rate. After that, incoming packets are fed
    directly to ffmpeg's stdin.
    """
    ffmpeg_process = None
    current_codec = None
    max_pts = float("-inf")

    pipe_container = None
    pipe_stream = None

    def record_frame(message: FrameAny):
        nonlocal ffmpeg_process, current_codec, max_pts, pipe_container, pipe_stream

        # Determine codec from the message.
        video_type = message.WhichOneof("data")
        if video_type == "h264":
            codec = "h264"
            submessage = message.h264
        elif video_type == "h265":
            codec = "hevc"
            submessage = message.h265
        elif video_type == "av1":
            codec = "av1"
            submessage = message.av1
        else:
            logging.warning("Unknown frame type received, skipping.")
            return

        if submessage.pts is None:
            logging.warning("Frame has no PTS, skipping.")
            return
        if submessage.pts <= max_pts:
            logging.warning("Frame has non-monotonic PTS, skipping.")
            return

        current_codec = codec

        if ffmpeg_process is None:
            ffmpeg_process = run_ffmpeg(chunk_duration_sec, codec=current_codec)
            threading.Thread(target=ffmpeg_monitor_thread, args=(ffmpeg_process,), daemon=True).start()
            pipe_container = av.open(ffmpeg_process.stdin, mode="w", format="mpegts")
            pipe_stream = pipe_container.add_stream(codec)

        pipe_packet = av.Packet(submessage.data)
        pipe_packet.pts = submessage.pts
        pipe_packet.dts = submessage.dts
        pipe_packet.duration = submessage.duration
        pipe_packet.is_keyframe = submessage.is_keyframe
        pipe_packet.stream = pipe_stream
        pipe_packet.stream.width = submessage.width
        pipe_packet.stream.height = submessage.height
        pipe_packet.stream.time_base = Fraction(submessage.time_base.num, submessage.time_base.den)

        pipe_container.mux_one(pipe_packet)

    return record_frame


def uploader_worker(path_prefix: str):
    """
    Continuously waits for completed segment filenames in the upload queue,
    reads their contents, and sends them using make87.request.
    """
    if path_prefix:
        path_prefix = path_prefix + "/"

    endpoint = make87.get_requester(
        name="FILE_TO_UPLOAD",
        requester_message_type=RelativePathFile,
        provider_message_type=Bool,
    )

    while True:
        filename = upload_queue.get()
        if filename is None:
            upload_queue.task_done()
            break

        header = Header()
        header.timestamp.GetCurrentTime()
        try:
            with open(filename, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")
            upload_queue.task_done()
            continue

        message = RelativePathFile(header=header, data=file_bytes, path=f"{path_prefix}{filename}")
        try:
            response = endpoint.request(message, timeout=60.0)
            if not response.value:
                logging.error(f"Failed to upload file: {filename}")
            else:
                logging.info(f"Uploaded file: {filename}")
        except Exception as e:
            logging.warning(f"Request failed for file {filename}: {e}")

        try:
            os.remove(filename)
        except Exception as e:
            logging.error(f"Error removing file {filename}: {e}")
        upload_queue.task_done()


def main():
    make87.initialize()

    # Start the uploader worker thread.
    path_prefix = make87.get_config_value("PATH_PREFIX", "", decode=lambda x: x.lstrip("/").rstrip("/"))
    uploader_thread = threading.Thread(target=uploader_worker, args=(path_prefix,), daemon=True)
    uploader_thread.start()

    # Retrieve chunk duration from configuration.
    chunk_duration_sec = make87.get_config_value("CHUNK_DURATION_SEC", "60", lambda x: int(x) if x.isdigit() else 60)

    # Subscribe to the video topic with our ffmpeg-based recorder callback.
    topic = make87.get_subscriber(name="VIDEO_DATA", message_type=FrameAny)
    topic.subscribe(recorder_worker_ffmpeg(chunk_duration_sec))

    make87.loop()


if __name__ == "__main__":
    main()
