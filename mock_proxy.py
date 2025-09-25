import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}

        message = {
            "role": "assistant",
            "content": "print('proxy backend active')"
        }
        response = {"id": "mock", "choices": [{"message": message}]}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A003 - suppress default logging
        return


def main():
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
