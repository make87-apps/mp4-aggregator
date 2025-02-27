import logging
import os
import queue
import threading
import re
import subprocess
from datetime import datetime, timezone

import make87 as m87
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


def run_ffmpeg(chunk_duration_sec: int, codec: str, frame_rate: float = None):
    """
    Start an FFmpeg process configured for segmentation.

    If frame_rate is provided, the '-r' option is inserted (before the input)
    so that ffmpeg assumes a constant frame rate.
    """
    # Map our codec to ffmpeg parameters.
    if codec == "h264":
        input_format = "h264"
        tag = "avc1"
    elif codec in ("h265", "hevc"):
        input_format = "hevc"
        tag = "hvc1"
    elif codec == "av1":
        input_format = "av1"
        tag = "av01"
    else:
        raise ValueError("Unsupported codec: " + codec)

    ffmpeg_cmd = ["ffmpeg"]
    # If a fixed frame rate is estimated, inject it before the input format.
    if frame_rate is not None:
        ffmpeg_cmd.extend(["-r", f"{frame_rate:.2f}"])
    ffmpeg_cmd.extend(
        [
            "-f",
            input_format,  # Input format (e.g. h264, hevc, or av1)
            "-i",
            "pipe:0",  # Read from stdin
            "-c:v",
            "copy",  # Copy codec without re-encoding
            "-tag:v",
            tag,  # Set the proper tag
            "-f",
            "segment",  # Use ffmpeg's built‑in segmenter
            "-segment_time",
            str(chunk_duration_sec),
            "-reset_timestamps",
            "1",  # Each segment starts at 0
            "-loglevel",
            "verbose",  # Detailed logging for monitoring
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


def recorder_worker_ffmpeg(chunk_duration_sec: int, buffer_frames: int = 5):
    """
    Recorder callback that buffers the first few frames to estimate a fixed frame rate,
    then starts an ffmpeg process with that rate. After that, incoming packets are fed
    directly to ffmpeg's stdin.
    """
    ffmpeg_process = None
    codec_set = False
    current_codec = None
    fixed_frame_rate = None
    frame_buffer = []  # Buffer tuples of (frame_time, packet_data)

    def record_frame(message: FrameAny):
        nonlocal ffmpeg_process, codec_set, current_codec, fixed_frame_rate, frame_buffer

        # Determine codec from the message.
        video_type = message.WhichOneof("data")
        if video_type == "h264":
            codec = "h264"
            submessage = message.h264
        elif video_type == "h265":
            codec = "h265"
            submessage = message.h265
        elif video_type == "av1":
            codec = "av1"
            submessage = message.av1
        else:
            logging.warning("Unknown frame type received, skipping.")
            return

        # Get the frame timestamp.
        frame_time = submessage.header.timestamp.ToDatetime()

        # If we haven't computed a fixed frame rate, buffer the frame.
        if fixed_frame_rate is None:
            frame_buffer.append((frame_time, submessage.data))
            # Wait until we have enough frames to estimate the rate.
            if len(frame_buffer) < buffer_frames:
                return
            # Compute the average frame rate.
            start_time = frame_buffer[0][0]
            end_time = frame_buffer[-1][0]
            delta_sec = (end_time - start_time).total_seconds()
            if delta_sec <= 0:
                fixed_frame_rate = 30.0  # Default fallback
            else:
                fixed_frame_rate = (len(frame_buffer) - 1) / delta_sec
            logging.info(f"Computed fixed frame rate: {fixed_frame_rate:.2f} fps based on {len(frame_buffer)} frames.")

            # Start the ffmpeg process with the computed frame rate.
            current_codec = codec
            try:
                ffmpeg_process = run_ffmpeg(chunk_duration_sec, codec=current_codec, frame_rate=fixed_frame_rate)
            except Exception as e:
                logging.error("Error starting ffmpeg: " + str(e))
                return
            # Launch the monitor thread.
            threading.Thread(target=ffmpeg_monitor_thread, args=(ffmpeg_process,), daemon=True).start()
            codec_set = True

            # Send all buffered frames.
            for _, packet_data in frame_buffer:
                try:
                    ffmpeg_process.stdin.write(packet_data)
                    ffmpeg_process.stdin.flush()
                except Exception as e:
                    logging.error("Error writing buffered frame to ffmpeg stdin: " + str(e))
            frame_buffer.clear()
            return

        # If fixed frame rate is already set and ffmpeg is running, send current frame.
        if ffmpeg_process and ffmpeg_process.stdin:
            try:
                ffmpeg_process.stdin.write(submessage.data)
                ffmpeg_process.stdin.flush()
            except Exception as e:
                logging.error("Error writing to ffmpeg stdin: " + str(e))

    return record_frame


def uploader_worker():
    """
    Continuously waits for completed segment filenames in the upload queue,
    reads their contents, and sends them using m87.request.
    """
    endpoint = m87.get_requester(
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

        message = RelativePathFile(header=header, data=file_bytes, path=filename)
        try:
            response = endpoint.request(message, timeout=60.0)
        except Exception as e:
            logging.error(f"Request failed for file {filename}: {e}")
            upload_queue.task_done()
            continue

        if not response.value:
            logging.error(f"Failed to upload file: {filename}")
        else:
            logging.info(f"Uploaded file: {filename}")

        try:
            os.remove(filename)
        except Exception as e:
            logging.error(f"Error removing file {filename}: {e}")
        upload_queue.task_done()


def main():
    m87.initialize()

    # Start the uploader worker thread.
    uploader_thread = threading.Thread(target=uploader_worker, daemon=True)
    uploader_thread.start()

    # Retrieve chunk duration from configuration.
    chunk_duration_sec = m87.get_config_value("CHUNK_DURATION_SEC", 60, lambda x: int(x) if x.isdigit() else 60)

    # Subscribe to the video topic with our ffmpeg-based recorder callback.
    topic = m87.get_subscriber(name="VIDEO_DATA", message_type=FrameAny)
    topic.subscribe(recorder_worker_ffmpeg(chunk_duration_sec, buffer_frames=5))

    m87.loop()


if __name__ == "__main__":
    main()
