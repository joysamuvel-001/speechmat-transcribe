import socket
from contextlib import closing

import server


def test_get_available_port_uses_a_different_port_when_preferred_is_busy():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        preferred = sock.getsockname()[1]

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as busy_sock:
            busy_sock.bind(("127.0.0.1", preferred))
            busy_sock.listen(1)

            chosen = server._get_available_port(preferred)

            assert chosen != preferred
            assert chosen > 0
