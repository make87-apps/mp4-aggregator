import logging
import os
import queue
import threading
from datetime import datetime, timezone
from fractions import Fraction

import av
import make87 as m87
from make87_messages.core.header_pb2 import Header
from make87_messages.file.simple_file_pb2 import RelativePathFile
from make87_messages.primitive.bool_pb2 import Bool
from make87_messages.video.any_pb2 import FrameAny

# Create a thread‑safe queue for files ready to be uploaded.
upload_queue: queue.Queue[str] = queue.Queue()

# Fallback duration in 90kHz ticks (~33ms)
FALLBACK_DURATION_TICKS = 3_000
TIME_BASE = Fraction(1, 90_000)


def convert_h265_to_hevc(vps: bytes, sps: bytes, pps: bytes):
    """Convert H.265 VPS/SPS/PPS from Annex B to MP4 HEVC format."""

    def nalu_length_prefixed(nal):
        return len(nal).to_bytes(4, byteorder="big") + nal  # 4-byte length prefix

    return nalu_length_prefixed(vps) + nalu_length_prefixed(sps) + nalu_length_prefixed(pps)


class Mp4ChunkRecorder:
    def __init__(self, chunk_duration_sec: int):
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_index = 0
        self.first_chunk = True

        self.container = None
        self.stream = None
        self.chunk_start_time = None
        self.last_packet = None
        self.current_filename = None
        self.current_codec = None

    def start_new_chunk(self, timestamp: datetime, codec: str, width: int, height: int) -> None:
        """Opens a new mp4 chunk for the given codec and timestamp."""
        # If this is not the first chunk, increment the index.
        if not self.first_chunk:
            self.chunk_index += 1
        else:
            self.first_chunk = False

        self.current_codec = codec

        timestamp = timestamp.replace(tzinfo=timezone.utc)
        self.chunk_start_time = timestamp
        timestamp_local = timestamp.astimezone()
        self.current_filename = (
            timestamp_local.strftime("%Y%m%d_%H%M%S")
            + f"_{timestamp_local.microsecond // 1000:03d}"
            + "_"
            + timestamp_local.strftime("%z")
            + ".mp4"
        )
        logging.info(f"Opening new chunk: {self.current_filename} (codec: {codec})")
        self.container = av.open(self.current_filename, mode="w", format="mp4")

        if codec == "h264":
            stream_codec = "h264"
        elif codec == "h265":
            stream_codec = "hevc"
        elif codec == "av1":
            stream_codec = "av1"
        else:
            logging.error(f"Unsupported codec: {codec}")
            raise ValueError(f"Unsupported codec: {codec}")

        self.stream = self.container.add_stream(stream_codec)
        if width > 0 and height > 0:  # we assume any of the values being "0" means its not provided.
            self.stream.width = width
            self.stream.height = height
        self.stream.time_base = TIME_BASE
        self.last_packet = None

        self.stream.codec_context.extradata = convert_h265_to_hevc(
            vps=b"@\x01\x0c\x01\xff\xff\x01`\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x96\xac\t",
            sps=b"B\x01\x01\x01`\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x96\xa0\x01\xe0 \x02\x1c\x7f\x8a\xad;\xa2K\xb2",
            pps=b"D\x01\xc0r\xf0\x94\x1e\xf6H",
        )

    def flush_last_packet(self, final_pts: int | None = None) -> None:
        """Flushes the last packet if one exists."""
        if self.last_packet is not None:
            if final_pts is not None:
                dur = final_pts - self.last_packet.pts
                if dur <= 0:
                    dur = FALLBACK_DURATION_TICKS
            else:
                dur = FALLBACK_DURATION_TICKS
            self.last_packet.duration = dur
            self.last_packet.stream = self.stream
            self.container.mux_one(self.last_packet)
            self.last_packet = None

    def close_chunk(self) -> None:
        """Finalizes and closes the current chunk, then queues it for upload."""
        if self.container is not None:
            total_ticks = int(self.chunk_duration_sec * 90000)
            self.flush_last_packet(final_pts=total_ticks)
            self.container.close()
            logging.info(f"Closed chunk {self.chunk_index}")
            if self.current_filename:
                upload_queue.put(self.current_filename)
                logging.info(f"Queued file for upload: {self.current_filename}")
            # Reset chunk state
            self.container = None
            self.stream = None
            self.chunk_start_time = None
            self.last_packet = None
            self.current_filename = None
            self.current_codec = None

    def process_frame(self, message: FrameAny) -> None:
        """
        Processes incoming FrameAny messages. If the codec changes or the elapsed time
        meets/exceeds the chunk duration, the current chunk is closed and a new one is started.
        """
        # Determine which codec is used.
        if message.HasField("h264"):
            codec = "h264"
            frame_variant = message.h264
        elif message.HasField("h265"):
            codec = "h265"
            frame_variant = message.h265
        elif message.HasField("av1"):
            codec = "av1"
            frame_variant = message.av1
        else:
            logging.error("FrameAny message does not contain a supported codec.")
            return

        timestamp = message.header.timestamp.ToDatetime().replace(tzinfo=timezone.utc)
        width, height = frame_variant.width, frame_variant.height

        # If no active recording or if the codec has changed, start a new chunk.
        if self.container is None or self.current_codec != codec:
            if self.container is not None:
                self.close_chunk()
            self.start_new_chunk(timestamp, codec, width, height)

        elapsed_sec = (timestamp - self.chunk_start_time).total_seconds()
        if elapsed_sec >= self.chunk_duration_sec:
            self.flush_last_packet(final_pts=int(self.chunk_duration_sec * 90000))
            self.close_chunk()
            self.start_new_chunk(timestamp, codec, width, height)
            elapsed_sec = 0

        pts = int(elapsed_sec / TIME_BASE)

        # Create a packet from the frame bytes.
        packet = av.Packet(bytes(frame_variant.data))
        packet.pts = pts
        packet.dts = pts
        packet.is_keyframe = frame_variant.is_keyframe
        packet.time_base = TIME_BASE
        packet.stream = self.stream

        if self.last_packet is not None:
            duration = pts - self.last_packet.pts
            if duration <= 0:
                duration = FALLBACK_DURATION_TICKS
            self.last_packet.duration = duration
            self.last_packet.stream = self.stream
            self.container.mux_one(self.last_packet)

        self.last_packet = packet


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

    # Instantiate our recorder.
    recorder = Mp4ChunkRecorder(chunk_duration_sec)

    topic = m87.get_subscriber(name="VIDEO_DATA", message_type=FrameAny)
    # Subscribe a lambda that passes each message to our recorder.
    topic.subscribe(recorder.process_frame)
    m87.loop()


if __name__ == "__main__":
    main()
