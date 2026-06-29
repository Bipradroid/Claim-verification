import http.server
import socketserver
import webbrowser
import sys
import os

# Serve from the claims directory so index.html can find output.csv,
# user_history.csv, and all claim images via relative paths.
CLAIMS_DIR = "c:/Users/biswa/Downloads/claims/claims"
os.chdir(CLAIMS_DIR)

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

class MyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

print(f"Starting server on port {PORT}...")
print(f"To view the dashboard, open this link in your browser:")
print(f"👉 http://localhost:{PORT}/index.html 👈")
print("Press Ctrl+C to stop the server.")

try:
    with MyTCPServer(("", PORT), Handler) as httpd:
        # Try to open the browser automatically
        try:
            webbrowser.open(f"http://localhost:{PORT}/index.html")
        except Exception:
            pass
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nStopping server.")
    sys.exit(0)
except Exception as e:
    print(f"Error starting server: {e}")
