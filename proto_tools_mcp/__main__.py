"""mcp_server/__main__.py

Entry point: python -m mcp_server."""

try:
    from mcp_server.server import main
except ImportError as exc:
    if "fastmcp" in str(exc):
        raise SystemExit(
            "MCP server requires fastmcp. Install with: pip install -e '.[mcp]'"
        ) from None
    raise

main()
