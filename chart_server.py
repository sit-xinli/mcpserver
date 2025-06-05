# repl_server.py

from langchain_experimental.utilities import PythonREPL
from typing import Annotated, List
from mcp.server.fastmcp import FastMCP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import contextlib
import sys

# MCP Server のインスタンス作成
mcp = FastMCP("PythonREPL")

# Python REPL ユーティリティの初期化
repl = PythonREPL()

@mcp.tool()
def python_repl(
    dates: Annotated[List[str], "日付のリスト（例: ['2025-04-01', ...]）"],
    closes: Annotated[List[float], "終値のリスト（例: [141.43, ...]）"]
) -> str:
    """
    日付と終値のリストからmatplotlibでチャートを生成し、base64 PNG画像として返す。
    """
    try:
        from datetime import datetime
        import base64
        from io import BytesIO

        dt_list = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        plt.figure(figsize=(10, 6))
        plt.plot(dt_list, closes, marker='o')
        plt.title('Oracle Stock Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        return f"![chart](data:image/png;base64,{img_base64})"
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
