# from langchain_community.chat_models import ChatOCIGenAI
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

import asyncio
import os
import re

from IPython.display import display, Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime

#.envファイルから環境変数を読み込む
load_dotenv(override=True)

# OpenAIのエンドポイントとAPIキーの定義
OPENAI_URL="https://api.openai.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    # OpenAI GPT-4.1-nanoモデルの定義
    model = ChatOpenAI(
        model="gpt-4.1-nano",  # gpt-4.1-nanoの正式なモデル名
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        temperature=0.7,
        max_tokens=3000
    )

    # MCPサーバーの定義
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": ["server.py"],
                "transport": "stdio",
            },
            "chart": {
                "command": "python",
                "args": ["chart_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print("--- 利用可能なツール一覧 ---")
    print(tools)

    # エージェントの定義(LangGraphでReActエージェントを定義)
    agent = create_react_agent(model, tools)

    # 入力プロンプトの定義
    agent_response = await agent.ainvoke({
        "messages": (
            "2025年4月1日から2025年4月5日までのOracle（ORCL）の株の日次終値データを取得し、日付と終値のリストからmatplotlibでチャートを生成し、base64 PNG画像として返す"
        )
    })
    # 出力結果の表示
    print(">>> 株価データ（生データ）---")
    print(agent_response["messages"][-1].content)

    print("--- 株価データ（生データ）<<<")

    # agent_response["messages"] チャートデータ抽出して表示する
    match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=\s]+)', agent_response["messages"][-1].content, re.DOTALL)
    if match:
        image_data = match.group(1).replace('\n', '').replace('\r', '').replace(' ', '')
        from IPython.display import display, Image
        import base64
        display(Image(data=base64.b64decode(image_data)))
    else:
        print("画像データが見つかりませんでした。")

if __name__ == "__main__":
    asyncio.run(main())
