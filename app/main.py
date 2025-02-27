import logging
import os
import queue
import threading
import time
from datetime import datetime
from fractions import Fraction

import av
import make87 as m87
from make87_messages.core.header_pb2 import Header
from make87_messages.file.simple_file_pb2 import RelativePathFile
from make87_messages.primitive.bool_pb2 import Bool
from make87_messages.video.any_pb2 import FrameAny

# Create a thread‑safe queue for files ready to be uploaded.
upload_queue: queue.Queue[str] = queue.Queue()

# Time base for PTS/DTS
TIME_BASE: Fraction = Fraction(1, 90000)  # 90 kHz clock


def get_new_filename(timestamp: datetime) -> str:
    return f"{timestamp:%Y%m%d_%H%M%S}_{timestamp.microsecond // 1000:03d}_{timestamp:%z}.mp4"


def recorder_worker(chunk_duration_sec: int):
    """Handles writing incoming frames to MP4 files."""
    container: av.container.OutputContainer | None = None
    stream: av.video.VideoStream | None = None
    filename: str | None = None
    start_pts: int | None = None  # Store the PTS offset for each chunk
    chunk_duration_ticks: int = int(chunk_duration_sec / TIME_BASE)

    def record_frame(message: FrameAny):
        nonlocal container, stream, filename, start_pts

        # Identify codec and access correct field
        video_type = message.WhichOneof("data")
        if video_type == "h264":
            codec_name = "h264"
            codec_tag = "avc1"
            submessage = message.h264
        elif video_type == "h265":
            codec_name = "hevc"
            codec_tag = "hvc1"
            submessage = message.h265
        elif video_type == "av1":
            codec_name = "av1"
            codec_tag = "av01"
            submessage = message.av1
        else:
            print("Unknown frame type received, discarding.")
            return

        # Convert protobuf timestamp to PTS (90kHz time base)
        frame_time = submessage.header.timestamp.ToDatetime()
        current_pts = int(frame_time.timestamp() / TIME_BASE)

        # Wait for a keyframe to start a new file
        if container is None and not submessage.is_keyframe:
            return  # Ignore non-keyframes until a keyframe starts a new chunk

        # Start a new file if:
        # 1. No container exists (first chunk)
        # 2. The current chunk has exceeded the duration (in PTS)
        # 3. We have a keyframe and need to start a fresh segment
        if container is None or (start_pts is not None and (current_pts - start_pts) >= chunk_duration_ticks):
            if container:
                container.close()
                upload_queue.put(filename)  # Add completed file to upload queue

            # Create new filename and container
            filename = get_new_filename(frame_time)
            container = av.open(filename, mode="w")
            stream = container.add_stream(codec_name)
            stream.width = submessage.width
            stream.height = submessage.height
            stream.time_base = TIME_BASE
            stream.codec_context.codec_tag = codec_tag

            # Reset PTS base (first frame starts at PTS 0)
            start_pts = current_pts

        # Stop writing only if the next frame is a keyframe
        if submessage.is_keyframe and (current_pts - start_pts) >= chunk_duration_ticks:
            container.close()
            container = None  # Ensure we don't process further packets in a closed file
            upload_queue.put(filename)

            # Start a new file immediately with the current keyframe
            filename = get_new_filename(frame_time)
            container = av.open(filename, mode="w")
            stream = container.add_stream(codec_name)
            stream.time_base = TIME_BASE

            # Reset PTS base (this frame should start the new segment)
            start_pts = current_pts

        # Adjust PTS for this segment so the first frame is always PTS=0
        packet = av.Packet(submessage.data)
        packet.pts = current_pts - start_pts
        packet.dts = current_pts - start_pts

        container.mux(packet)

    return record_frame


def uploader_worker() -> None:
    """
    Continuously waits for filenames in the upload queue,
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


def main() -> None:
    m87.initialize()

    # Start the uploader worker thread.
    uploader_thread = threading.Thread(target=uploader_worker, daemon=True)
    uploader_thread.start()

    # In the configuration, set the chunk duration in seconds.
    chunk_duration_sec = m87.get_config_value("CHUNK_DURATION_SEC", 60, lambda x: int(x) if x.isdigit() else 60)

    topic = m87.get_subscriber(name="VIDEO_DATA", message_type=FrameAny)
    # Subscribe with recorder callback
    topic.subscribe(recorder_worker(chunk_duration_sec))

    m87.loop()


if __name__ == "__main__":
    main()
