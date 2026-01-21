import json
import re
import datetime
import traceback

def safe_parse_json(text: str) -> dict:
    """鲁棒的 JSON 解析
    引入容错解析逻辑（如 json_repair 库的思想或更强的正则
    """
    try:
        # 1. 尝试提取 Markdown 块
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
            clean_target = match.group(0) # 用于后续删除
        else:
            # 2. 尝试提取最外层 {}
            match = re.search(r"\{.*\}", text, re.DOTALL)
            clean_target = match
            if match:
                text = match.group(0)
                clean_target = text
        
        # 3. 简单的预处理（生产环境建议使用 json_repair 库）
        text = re.sub(r",\s*\}", "}", text) # 去除尾部逗号
        return json.loads(text), clean_target
    except Exception as e:
        return {},""

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
