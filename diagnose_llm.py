"""LLM 诊断脚本：验证 DeepSeek 是否真正被调用"""

from langchain_openai import ChatOpenAI
from src.config import config

print("=" * 60)
print("LLM 配置信息")
print("=" * 60)
print(f"  模型名称: {config.llm.model_name}")
print(f"  API 地址: {config.llm.base_url}")
print(f"  Temperature: {config.llm.temperature}")
print(f"  Max Tokens: {config.llm.max_tokens}")
print(f"  API Key 前5位: {config.llm.api_key[:5]}...")
print()

print("=" * 60)
print("开始调用 DeepSeek-V4 Pro")
print("=" * 60)

llm = ChatOpenAI(
    model=config.llm.model_name,
    api_key=config.llm.api_key,
    base_url=config.llm.base_url,
    temperature=0.7,
    max_tokens=500,
)

response = llm.invoke("请用一句话自我介绍，格式：我是[名字]，来自[机构]，专长是[领域]")
print(f"\n模型原始返回对象: {type(response)}")
print(f"模型返回内容: {response.content}")
print()

print("=" * 60)
print("调用成功！LLM 生成验证通过")
print("=" * 60)
