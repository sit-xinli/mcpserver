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
                "args": ["repl_server.py"],
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
            "2025年4月1日から2025年4月5日までのOracle（ORCL）の株の日次終値データを取得し、"
            "その日付と終値のリストを教えてください。"
        )
    })
    result = agent_response["messages"][-1].content
    # 出力結果の表示
    print(">>> 株価データ（生データ）---")
    print(agent_response["messages"][-1].content)

    print("--- 株価データ（生データ）<<<")

    # agent_response["messages"] から日付リスト・終値リストを抽出
    dates = None
    closes = None
    for msg in agent_response["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # 1. リスト形式での抽出
            match_dates = re.search(r'日付リスト[:：]\s*\[([\s\S]+?)\]', msg.content)
            match_closes = re.search(r'終値リスト[:：]\s*\[([\s\S]+?)\]', msg.content)
            if match_dates and match_closes:
                try:
                    dates = eval(f'[{match_dates.group(1)}]')
                    closes = eval(f'[{match_closes.group(1)}]')
                except Exception:
                    continue
                break
            # 2. テーブル形式での抽出
            match_table = re.search(r'日付.*?出来高\n([\s\S]+?)```', msg.content)
            if match_table:
                stock_data = match_table.group(1)
                dates = []
                closes = []
                for line in stock_data.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            date_str = parts[0]
                            close = float(parts[5])
                            dates.append(date_str)
                            closes.append(close)
                        except Exception:
                            continue
                if dates and closes:
                    break
    if dates and closes:
        # ツールリストからpython_replツールを探す
        python_repl_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "python_repl":
                python_repl_tool = tool
                break

        if python_repl_tool:
            result = await python_repl_tool.ainvoke({"dates": dates, "closes": closes})
            print("--- ---------------------------------------- ---")
            print(result)
            # base64画像を抽出して表示
            match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=\s]+)', result, re.DOTALL)
            if match:
                image_data = match.group(1).replace('\n', '').replace('\r', '').replace(' ', '')
                from IPython.display import display, Image
                import base64
                display(Image(data=base64.b64decode(image_data)))
            else:
                print("画像データが見つかりませんでした。")
        else:
            print("python_replツールが見つかりませんでした。")
    else:
        print("日付リストまたは終値リストが見つかりませんでした。")

if __name__ == "__main__":
    asyncio.run(main())
