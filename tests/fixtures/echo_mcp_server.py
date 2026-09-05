"""供集成测试使用的最小 stdio MCP Server。"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def get_echo(text: str) -> str:
    """把输入原样回显，用来验证 MCP 通路。"""

    return f"echo:{text}"


if __name__ == "__main__":
    server.run(transport="stdio")
