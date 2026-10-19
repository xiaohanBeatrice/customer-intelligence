"""测试 Claude API 是否配置成功。运行：python llm/hello.py"""
import anthropic
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件读取 ANTHROPIC_API_KEY

client = anthropic.Anthropic()  # 会自动使用环境变量里的 key

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=200,
    messages=[{"role": "user", "content": "用一句话介绍什么是客户分群。"}],
)

print(message.content[0].text)