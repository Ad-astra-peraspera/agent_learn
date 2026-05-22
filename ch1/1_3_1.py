AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步解决问题。

# 可用工具：
- `get_weather(city:str)`: 查询指定城市的实时天气
- `get_attraction(city:str, weather: str)`:根据城市和天气搜索推荐的旅游景点

# 输出格式要求：
你的每次回复必须严格遵循以下格式，包含一对Thought和Action。
Thought:[你的思考过程和下一步计划]
Action:[你要执行的具体行动]

Action的格式必须是以下之一：
1.调用工具：function_name(arg_name = "arg_value")
2.结束任务：Finish[最终答案]

# 重要提示
-每次只输出一对Thought-Action
-Action必须在同一行，不要换行
-当收集到足够信息可以回答用户问题时，必须使用Action:Finish[最终答案]格式结束

请开始吧！
"""

# 工具部分————————————————————————————————
# 1.天气工具
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def get_weather(city: str) -> str:
    """
    通过调用wttr.in API 查询真实的天气信息
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url, timeout=10)
        # 检查状态码
        response.raise_for_status()
        # 解析返回的json数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}的当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误：查询天气时遇到网络错误 - {e}"
    
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误，解析天气数据失败"

# 2. 搜索并推荐旅游景点—

def get_attraction(city:str, weather:str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误，未配置tavily_api_key"
    tavily = TavilyClient(api_key = api_key)
    query = f"'{city}'在'{weather}'天气下最值得去的旅游经典推荐及理由"
    try:
        response = tavily.search(query = query, search_depth = "basic", include_answer = True)
        if response.get("answer"):
            return response["answer"]
        formatted_results = []
        for result in response.get('results', []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        if not formatted_results:
            return "抱歉没找到相关的推荐"
        return "根据搜索，为您找到一下信息：\n" + "\n".join(formatted_results)
    
    except Exception as e:
        return f"错误：执行Tavily搜索时遇到问题 - {e}"
        
# 所有工具
available_tools = {
    "get_weather" : get_weather,
    "get_attraction": get_attraction,
}

class OpenAICompatibleClient:
    '''
    一个用于调用兼容openai接口的llm服务客户端
    '''
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key = api_key, base_url = base_url)
    
    def generate(self, prompt : str, system_prompt : str ) -> str:
        print("正在调用大模型")
        try:
            messages = [
                {"role" : 'system', 'content':system_prompt},
                {'role' : 'user', 'content': prompt }
            ]
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages,
                stream = False
            )
            answer = response.choices[0].message.content
            print('大语言模型相应成功')
            return answer
        except Exception as e:
            print(f"调用llm时发生错误:{e}")
            return '错误，调用语言模型服务时出错'

# 执行行动循环——————————————————————————————————
ENV_FILE = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_FILE)


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"缺少环境变量 {name}。请在 {ENV_FILE} 中配置后重试。")


API_KEY = require_env_var("API_KEY")
BASE_URL = require_env_var("BASE_URL")
MODEL_ID = require_env_var("MODEL_ID")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

llm = OpenAICompatibleClient(
    model = MODEL_ID,
    api_key = API_KEY,
    base_url=BASE_URL
)
#-----------------------------

user_prompt = '你好，请帮我查询一下北京今天的天气，然后根据天气推荐一个合适的旅游景点'
prompt_history = [f'用户请求：{user_prompt}']

print(f'用户输入：{user_prompt}\n' + '='*40)

#-----------------------

for i in range(5):
    print(f'----循环{i+1}------')
    full_prompt = '\n'.join(prompt_history)

    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        llm_output,
        re.DOTALL,
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print('已截取多余 Thought-Action 对')
    print(f'模型输出：\n{llm_output}')
    prompt_history.append(llm_output)

    #-----------------------------
    action_match = re.search(r"Action:(.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = '错误，未解析到action字段。请确保你的回复严格遵循Thought:....Action:.....格式'
        observation_str = f'Observation :{observation}'
        print(f'{observation_str}\n' + '='*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()
    
    if action_str.startswith('Finish'):
        final_answer = re.match(r"Finish\[(.*)\]$", action_str, re.DOTALL).group(1)
        print(f'任务完成，最终答案：{final_answer}')
        break

    tool_name = re.search(r'(\w+)\(', action_str).group(1)
    args_str = re.search(r'\((.*)\)', action_str).group(1)
    kwargs = dict(
        re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str)
    )

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f'错误，未定义的工具:"{tool_name}"'
    
    observation_str = f'Observation:{observation}'
    print(f'{observation_str}\n'+'='*40)
    prompt_history.append(observation_str)
