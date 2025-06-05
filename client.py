# from langchain_community.chat_models import ChatOCIGenAI
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

import asyncio
import os
from dotenv import load_dotenv

#.envファイルから環境変数を読み込む
load_dotenv(override=True)

# OpenAIのエンドポイントとAPIキーの定義
OPENAI_URL="https://api.openai.com/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Oracle Cloudの認証情報定義（不要なら削除可）
# config = oci.config.from_file("~/.oci/config", "DEFAULT")

async def main():
    # OpenAI GPT-4.1-nanoモデルの定義
    model = ChatOpenAI(
        model="gpt-4.1-nano",  # gpt-4.1-nanoの正式なモデル名
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        temperature=0.7,
        max_tokens=500
    )

    # MCPサーバーの定義
    client = MultiServerMCPClient(
        {
            "stock": {
                "command": "python",
                "args": ["server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()

    # エージェントの定義(LangGraphでReActエージェントを定義)
    agent = create_react_agent(model, tools)

    # 入力プロンプトの定義
    agent_response = await agent.ainvoke({
        "messages": "2025年4月1から2025年4月5日までのOracleの株の日次の終値を取得してください。"
    })

    # 出力結果の表示
    print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())
