import time

from make87_messages.core.header_pb2 import Header
from make87_messages.text.text_plain_pb2 import PlainText
import make87 as m87


def main():
    m87.initialize()
    topic = m87.get_publisher(name="HELLO_WORLD_MESSAGE", message_type=PlainText)

    while True:
        message = PlainText(header=m87.create_header(Header, entity_path="/"), body="Hello, World! 🐍")
        topic.publish(message)
        print(f"Published: {message}")
        time.sleep(1)


if __name__ == "__main__":
    main()
