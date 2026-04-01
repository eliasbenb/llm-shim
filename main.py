"""llm-shim server entry point."""

import uvicorn

from llm_shim.core.config import get_settings


def main() -> None:
    """Run the llm-shim API server."""
    settings = get_settings()
    server = settings.server
    uvicorn.run(
        "llm_shim:app",
        host=server.host,
        port=server.port,
        reload=server.reload,
        workers=server.workers,
        log_level=server.log_level,
    )


if __name__ == "__main__":
    main()
