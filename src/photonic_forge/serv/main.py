"""Entrypoint for the web server."""

import uvicorn

def main():
    """Run the web server."""
    print("Starting PhotonicForge Web Prototype on http://localhost:8000")
    uvicorn.run("photonic_forge.serv.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
