#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
import json
import requests
from typing import Tuple, Optional, Any, Dict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

project_root_parent = Path(
    __file__
).parent.parent.parent.parent  # robonix root
sys.path.insert(0, str(project_root_parent))

from robonix.uapi import get_runtime, set_runtime
from robonix.manager.log import logger, set_log_level
from robonix.uapi.runtime.action import EOS_TYPE_ActionResult

from robonix.skill import *

set_log_level("debug")

# global variables to store map and detected objects
map = {
    "工作台": (28.83777255787819, -1.3764122022097915, -1.5598185730243177),
    "门口": (28.644476781664718, -2.902279232068204, -1.4588087271409134),
}

objects = {
    "杯子": {"x": 320, "y": 240, "depth": 0.8, "confidence": 0.95},
    "书": {"x": 400, "y": 180, "depth": 0.9, "confidence": 0.87},
    "手机": {"x": 280, "y": 300, "depth": 0.7, "confidence": 0.91}
}

# DeepSeek API
DEEPSEEK_API_KEY = "sk-a93a0c965ad5490ea147db69300fd565"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def detect(runtime) -> Dict[str, Any]:
    """detect objects and update global objects dict"""
    print("detect object...")
    new_objects = {}
    runtime.configure_action("get_detection", ranger_path="/ranger")

    logger.info("Starting ranger test action...")
    thread = runtime.start_action("get_detection")
    result = runtime.wait_for_action("get_detection", timeout=30.0)

    logger.info(f"get detection completed with result: {result}")
    return result

def move_to_position(runtime, x: float, y: float, yaw: float) -> bool:
    """move to specified position"""
    print(f"move to position: x={x}, y={y}, yaw={yaw}")
    runtime.configure_action("move_ranger", ranger_path="/ranger", x=x, y=y, yaw=yaw)

    logger.info("Starting move_ranger action...")
    thread = runtime.start_action("move_ranger")
    result = runtime.wait_for_action("move_ranger", timeout=30.0)

    logger.info(f"Ranger test completed with result: {result}")
    logger.info("move_ranger completed successfully")
    return True

def grasp_object(runtime, object_name: str) -> bool:
    """grasp specified object"""
    print(f"grasp: {object_name}")

    runtime.configure_action("grasp", ranger_path="/ranger", object_name = object_name)
    thread = runtime.start_action("grasp")
    result = runtime.wait_for_action("grasp", timeout=30.0)
    return True

def call_deepseek_api(user_instruction: str) -> Tuple[str, Optional[Any]]:
    """
    call DeepSeek API to classify instruction
    
    Args:
        user_instruction: user instruction text
        
    Returns:
        Tuple[str, Optional[Any]]: (classification result, second return value)
    """
    # 构建prompt让DeepSeek进行分类
    global objects, map
    print(f"debug : map: {map} objects: {objects}")
    system_prompt = f"""You are a robot command classifier. Please classify user instructions into one of the following categories:
        1. detect - detect objects in the environment
        2. move - move to a location
        3. grasp - grasp an object
        4. exit - exit the program
        5. skip - unable to understand or no action needed

        Based on the environmental information, return the appropriate second return value:
        - detect, exit, skip: return None
        - move: return the destination name (e.g., "table", "chair")
        - grasp: return the object name (e.g., "cup", "book")
        Currently available destinations for move are: {list(map.keys())}
        Available objects for grasp and their information are: {objects}
        (Camera resolution: height: 480, width: 640)
        If the destination or object is not in the above lists, please find similar ones; if no similar item exists, return "skip" as the action.

        Please reply in JSON format: {{"action": "classification", "target": "second return value"}}"""
    try:
        # call DeepSeek API
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
        print("debug: DeepSeek API response status code:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("debug: DeepSeek API response content:", content)
            
            # parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                action = parsed.get("action", "skip")
                target = parsed.get("target", None)
                
                # handle move action
                if action == "move":
                    # check if target in map
                    if target in map:
                        return action, map[target]
                    else:
                        print(f"[warn]: target '{target}' not in map, skip this instruction")
                        return "skip", None
                
                elif action == "grasp":
                    # check if target in objects
                    if target in objects:
                        return action, target
                    else:
                        print(f"[warn]: object '{target}' not in objects, skip this instruction")
                        return "skip", None
                
                else:
                    return action, None
            else:
                print("[warn]: failed to parse JSON from DeepSeek response")
                return "skip", None
                
        else:
            print(f"[warn]: API call failed: {response.status_code}")
            return "skip", None
            
    except Exception as e:
        print(f"[warn]: error calling DeepSeek API: {e}")
        return "skip", None

def process_instruction(runtime, instruction: str) -> bool:
    """
    prosess single instruction
    
    Args:
        instruction: user instruction string
        
    Returns:
        bool: whether to continue loop (False means exit)
    """
    if not instruction.strip():
        return True
    
    # call DeepSeek API to classify instruction
    action, target = call_deepseek_api(instruction)
    
    print(f"[info]: action={action}, target={target}")
    
    # execute action based on classification
    if action == "detect":
        global objects
        objects = detect(runtime)
        print("object detection completed, updated objects:", objects)
        
    elif action == "move":
        if target:
            x, y, yaw = target
            success = move_to_position(runtime, x, y, yaw)
            if success:
                print("move successful")
            else:
                print("move failed")
    
    elif action == "grasp":
        if target:
            success = grasp_object(runtime, target)
            if success:
                print(f"grasp {target} successful")
            else:
                print(f"grasp {target} failed")
    
    elif action == "exit":
        print("exit program")
        return False
    
    elif action == "skip":
        print("skip this instruction")
    
    return True


def init_skill_providers(runtime):
    """Initialize skill providers for ranger demo"""
    from robonix.uapi.runtime.provider import SkillProvider

    # dump __all__ in robonix.skill to skills list
    try:
        from robonix.skill import __all__
        skills = __all__
    except ImportError:
        logger.warning("robonix.skill module not available")
        skills = []

    local_provider = SkillProvider(
        name="local_provider",
        IP="127.0.0.1",
        skills=skills,
    )

    runtime.registry.add_provider(local_provider)
    logger.info(f"Added skill providers: {runtime.registry}")

def create_ranger_entity_builder():
    """Create a ranger-specific entity graph builder"""
    def builder(runtime, **kwargs):
        from robonix.uapi.graph.entity import create_root_room, create_controllable_entity

        root_room = create_root_room()
        runtime.set_graph(root_room)

        ranger = create_controllable_entity("ranger")
        root_room.add_child(ranger)

        # vision skills
        ranger.bind_skill("cap_camera_rgb", cap_camera_rgb, "local_provider")
        ranger.bind_skill("cap_camera_dep_rgb", cap_camera_dep_rgb, "local_provider")
        ranger.bind_skill("cap_camera_info", cap_camera_info, "local_provider")
        ranger.bind_skill("cap_tf_transform", cap_tf_transform, "local_provider")
        
        # geasp skills
        ranger.bind_skill("skl_detect_objs", skl_detect_objs, "local_provider")
        ranger.bind_skill("skl_grasp_object", skl_grasp_object, "local_provider")

        # move skills
        ranger.bind_skill("cap_get_pose", get_pose, "local_provider")
        ranger.bind_skill("cap_set_goal", simple_set_goal, "local_provider")
        ranger.bind_skill("skl_move_to_ab_pos", move_to_ab_pos, "local_provider")

        logger.info("Ranger entity graph initialized:")
        logger.info(f"  root room: {root_room.get_absolute_path()}")
        logger.info(f"  ranger: {ranger.get_absolute_path()}")

    return builder


def main():
    parser = argparse.ArgumentParser(description="Grasp Demo")
    parser.add_argument(
        "--export-scene",
        type=str,
        help="Export scene information to JSON file",
    )
    args = parser.parse_args()

    logger.info("Starting grasp demo")

    runtime = get_runtime()
    runtime.register_entity_builder("ranger", create_ranger_entity_builder())
    init_skill_providers(runtime)
    runtime.build_entity_graph("ranger")
    set_runtime(runtime)
    runtime.print_entity_tree()
    if args.export_scene:
        scene_info = runtime.export_scene_info(args.export_scene)
        logger.info(f"Scene information exported to: {args.export_scene}")

    action_program_path = os.path.join(
        os.path.dirname(__file__), "quick_demo.action")
    logger.info(f"Loading action program from: {action_program_path}")

    try:
        action_names = runtime.load_action_program(action_program_path)
        logger.info(f"Loaded action functions: {action_names}")
        print("demo start:")
        print("available actions:")
        print("  - 'detect' (detect)")
        print("  - 'move' (move)")
        print("  - 'grasp' (grasp)")
        print("  - 'exit' (exit)")
        print("  - 'skip' (skip)")
        print("-" * 40)
        
        while True:
            try:
                # get user input
                user_input = input("please input action: ").strip()
                
                # process instruction
                should_continue = process_instruction(runtime, user_input)
                
                if not should_continue:
                    break
                    
                print()
                    
            except KeyboardInterrupt:
                print("\nprogram interrupted by user")
                break
            except Exception as e:
                print(f"error: {e}")

            
        logger.info("Demo completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
