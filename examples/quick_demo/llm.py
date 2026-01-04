import json
import requests
from typing import Tuple, Optional, Any, Dict

# 假设的全局变量（实际使用时需要初始化）
map = {
    "工作台": (28.83777255787819, -1.3764122022097915, -1.5598185730243177),
    "门口": (28.644476781664718, -2.902279232068204, -1.4588087271409134),
}

objects = {
    "杯子": {"x": 320, "y": 240, "depth": 0.8, "confidence": 0.95},
    "书": {"x": 400, "y": 180, "depth": 0.9, "confidence": 0.87},
    "手机": {"x": 280, "y": 300, "depth": 0.7, "confidence": 0.91}
}

# DeepSeek API配置（需要替换为实际的API密钥）
DEEPSEEK_API_KEY = "sk-a93a0c965ad5490ea147db69300fd565"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 假设的其他函数（需要根据实际硬件/仿真环境实现）
def detect() -> Dict[str, Any]:
    """执行物体检测，更新objects字典"""
    print("执行物体检测...")
    # 这里应该是实际的物体检测代码
    # 返回新的objects字典
    new_objects = {
        "杯子": {"x": 320, "y": 240, "depth": 0.8, "confidence": 0.95},
        "书": {"x": 400, "y": 180, "depth": 0.9, "confidence": 0.87},
        "手机": {"x": 280, "y": 300, "depth": 0.7, "confidence": 0.91}
    }
    return new_objects

def move_to_position(x: float, y: float, yaw: float) -> bool:
    """移动到指定位置"""
    print(f"移动到位置: x={x}, y={y}, yaw={yaw}")
    # 这里应该是实际的移动代码
    return True

def grasp_object(object_name: str) -> bool:
    """抓取指定物体"""
    print(f"抓取物体: {object_name}")
    # 这里应该是实际的抓取代码
    return True

def call_deepseek_api(user_instruction: str) -> Tuple[str, Optional[Any]]:
    """
    调用DeepSeek API对指令进行分类
    
    Args:
        user_instruction: 用户指令文本
        
    Returns:
        Tuple[str, Optional[Any]]: (分类结果, 第二返回值)
    """
    # 构建prompt让DeepSeek进行分类
    system_prompt = f"""你是一个机器人指令分类器。请将用户指令分类为以下之一：
        1. detect - 检测环境中的物体
        2. move - 移动到某个位置
        3. grasp - 抓取某个物体
        4. exit - 退出程序
        5. skip - 无法理解或不需要执行动作
        
        同时，根据根据环境中的信息分类返回相应的第二返回值：
        - detect, exit, skip: 返回None
        - move: 返回目的地名称（如"桌子"、"椅子"等）
        - grasp: 返回物体名称（如"杯子"、"书"等）
        我们目前move可以移动到的地点有{list(map.keys())}，grasp可以抓取的物体有{list(objects.keys())}。
        如果目的地或物体不在上述列表中，请寻找相似的地方，如果没有相似的，返回skip作为action
        
        请以JSON格式回复：{{"action": "分类", "target": "第二返回值"}}"""
    
    try:
        # 调用DeepSeek API
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction}
            ],
            "temperature": 0.1
        }
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        print("debug: DeepSeek API响应状态码:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("debug: DeepSeek API返回内容:", content)
            
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                action = parsed.get("action", "skip")
                target = parsed.get("target", None)
                
                # 根据action处理target
                if action == "move":
                    # 检查目的地是否在map中
                    if target in map:
                        return action, map[target]
                    else:
                        print(f"警告：未找到目的地 '{target}'，跳过此指令")
                        return "skip", None
                
                elif action == "grasp":
                    # 检查物体是否在objects中
                    if target in objects:
                        return action, target
                    else:
                        print(f"警告：未找到物体 '{target}'，跳过此指令")
                        return "skip", None
                
                else:
                    return action, None
            else:
                print("警告：API返回格式错误，跳过此指令")
                return "skip", None
                
        else:
            print(f"API调用失败: {response.status_code}")
            return "skip", None
            
    except Exception as e:
        print(f"调用DeepSeek API时出错: {e}")
        return "skip", None

def process_instruction(instruction: str) -> bool:
    """
    处理单个指令
    
    Args:
        instruction: 用户指令
        
    Returns:
        bool: 是否继续循环（False表示退出）
    """
    if not instruction.strip():
        return True
    
    # 调用DeepSeek API进行分类
    action, target = call_deepseek_api(instruction)
    
    print(f"分类结果: action={action}, target={target}")
    
    # 根据分类执行相应操作
    if action == "detect":
        global objects
        objects = detect()
        print("物体检测完成，objects已更新")
        
    elif action == "move":
        if target:
            x, y, yaw = target
            success = move_to_position(x, y, yaw)
            if success:
                print("移动完成")
            else:
                print("移动失败")
    
    elif action == "grasp":
        if target:
            success = grasp_object(target)
            if success:
                print(f"抓取 {target} 完成")
                # 抓取成功后从objects中移除该物体
                if target in objects:
                    del objects[target]
            else:
                print(f"抓取 {target} 失败")
    
    elif action == "exit":
        print("退出程序")
        return False
    
    elif action == "skip":
        print("跳过此指令")
    
    return True

def main():
    """主循环"""
    print("机器人控制系统启动")
    print("可用指令示例:")
    print("  - '看看周围有什么' (detect)")
    print("  - '移动到桌子那里' (move)")
    print("  - '抓取杯子' (grasp)")
    print("  - '退出' (exit)")
    print("  - '你好' (skip)")
    print("-" * 40)
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("请输入指令: ").strip()
            
            # 处理指令
            should_continue = process_instruction(user_input)
            
            if not should_continue:
                break
                
            print()  # 空行分隔
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理指令时出错: {e}")

# 简化版本：如果不使用API，可以使用基于关键词的分类
def simple_classify(instruction: str) -> Tuple[str, Optional[Any]]:
    """
    简化版本的指令分类器（不使用API）
    """
    instruction_lower = instruction.lower()
    
    # 检测关键词
    if any(word in instruction_lower for word in ["检测", "看看", "观察", "detect", "look"]):
        return "detect", None
    
    elif any(word in instruction_lower for word in ["移动", "去", "到", "move", "go"]):
        # 尝试提取目的地
        for location in map.keys():
            if location in instruction:
                return "move", map[location]
        return "skip", None
    
    elif any(word in instruction_lower for word in ["抓取", "拿", "取", "grasp", "pick"]):
        # 尝试提取物体名
        for obj in objects.keys():
            if obj in instruction:
                return "grasp", obj
        return "skip", None
    
    elif any(word in instruction_lower for word in ["退出", "结束", "exit", "quit"]):
        return "exit", None
    
    else:
        return "skip", None

# 如果不想使用API，可以用simple_classify替换call_deepseek_api
if __name__ == "__main__":
    # 初始化配置
    # 如果你没有DeepSeek API密钥，使用简化版本
    use_simple_classifier = False  # 设置为True使用简化分类器
    
    if use_simple_classifier:
        # 覆盖call_deepseek_api函数为简化版本
        call_deepseek_api = simple_classify
        print("使用简化分类器（关键词匹配）")
    
    main()