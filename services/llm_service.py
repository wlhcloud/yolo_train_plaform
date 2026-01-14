import base64
import json
import requests
from typing import Dict, List, Optional, Tuple
from models import LLMConfig

class LLMService:
    """大模型服务类，用于调用OpenAI兼容的API进行图像识别"""
    
    def __init__(self):
        self.active_config = None
        self._load_active_config()
    
    def _load_active_config(self) -> Optional[LLMConfig]:
        """加载当前激活的LLM配置"""
        try:
            self.active_config = LLMConfig.query.filter_by(is_active=True).first()
            return self.active_config
        except Exception as e:
            print(f"加载LLM配置失败: {e}")
            return None
    
    def reload_config(self):
        """重新加载配置"""
        self._load_active_config()
    
    def _get_api_url(self) -> str:
        """构建完整的API URL"""
        if not self.active_config:
            raise Exception("未找到激活的LLM配置")
        
        base_url = self.active_config.base_url
        
        # 如果URL已经包含chat/completions，则直接使用
        if 'chat/completions' in base_url:
            return base_url
        
        # 如果URL以/v1结尾，则添加/chat/completions
        if base_url.endswith('/v1'):
            return base_url + '/chat/completions'
        
        # 如果URL以/v1/结尾，则添加chat/completions
        if base_url.endswith('/v1/'):
            return base_url + 'chat/completions'
        
        # 否则添加标准的v1/chat/completions路径
        if base_url.endswith('/'):
            return base_url + 'v1/chat/completions'
        else:
            return base_url + '/v1/chat/completions'
    
    def is_configured(self) -> bool:
        """检查是否已配置LLM"""
        return self.active_config is not None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"图片编码失败: {e}")
    
    def generate_detection_prompt(self, labels: List[str]) -> str:
        """生成目标检测提示词"""
        if not labels:
            return "你是一个专业的视觉内容分析助手，正在对图片进行特定目标检测。请分析，仅基于图像可见内容进行客观判断，不要推测或想象。只返回清晰可见的对象。以JSON格式响应: {\"objects\": []}。如果没有看到清晰的对象，返回空数组。"
        
        labels_str = ", ".join(labels)
        prompt = f"""你是一个专业的视觉内容分析助手，正在对图片进行特定目标检测。请分析，仅基于图像可见内容进行客观判断，不要推测或想象。

检测任务：在图像中检测和定位所有可见的目标对象，要求极高的精确度。

目标检测标签：{labels_str}

核心要求：
1. 仅基于图像中实际可见的内容进行判断
2. 不要推测、想象或假设任何不可见的内容
3. 只检测清晰可见、边界明确的目标对象
4. 提供精确的边界框坐标，紧密贴合目标对象
5. x,y坐标表示对象的中心点
6. width,height表示边界框的精确尺寸
7. 所有坐标必须归一化到0.0-1.0范围内
8. 使用高精度小数（至少6位小数）- 避免使用0.1, 0.2, 0.5等整数
9. 坐标应该真实且多样化，如0.215696, 0.321893, 0.457900等

坐标系统：
- x=0.0为左边缘，x=1.0为右边缘
- y=0.0为上边缘，y=1.0为下边缘
- width/height：相对于图像尺寸的比例

检测指导原则：
- 面部/口罩检测：仅包含面部区域（从额头到下巴，从耳朵到耳朵）- 不包含整个人体
- 口罩检测：专注于佩戴口罩的面部区域，不包含全身
- 人员检测：仅当特定检测\"person\"类别时，才包含从头部到可见身体部分
- 重要：检测\"已戴口罩\"或\"未戴口罩\"时，返回面部区域的边界框，不是整个人
- 确保边界框紧密但完整地围绕目标区域
- 仔细研究图像，提供与实际对象位置匹配的真实坐标
- 客观分析：只报告确实可见的内容，不要填补或想象缺失的部分

严格按照以下JSON格式返回结果：
{{
    \"objects\": [
        {{
            \"label\": \"object_name\",
            \"confidence\": 0.95,
            \"x\": 0.234567,
            \"y\": 0.456789,
            \"width\": 0.123456,
            \"height\": 0.234567
        }}
    ]
}}

关键要求：仔细分析图像，仅基于可见内容进行客观判断，返回高精度坐标，准确反映对象位置！"""
        return prompt
    
    def call_vision_api(self, image_path: str, prompt: str) -> Dict:
        """调用视觉API进行图像分析"""
        if not self.active_config:
            raise Exception("未找到激活的LLM配置")
        
        try:
            # 编码图片
            base64_image = self.encode_image_to_base64(image_path)
            
            # 构建请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.active_config.api_key}"
            }
            
            payload = {
                "model": self.active_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 32000,  # 进一步增加token限制以避免响应被截断
                "temperature": 0.0,
                "stream": False
            }
            
            # 构建完整的API URL
            api_url = self._get_api_url()
            
            # 发送请求（禁用代理）
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180,  # 延长超时时间到3分钟
                proxies={'http': None, 'https': None}
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            return result
            
        except requests.exceptions.Timeout:
            print(f"API请求超时，超时时间: 180秒")
            raise Exception("API请求超时")
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {e}")
            print(f"请求URL: {api_url}")
            raise Exception(f"网络请求失败: {e}")
        except Exception as e:
            print(f"API调用失败: {e}")
            print(f"请求URL: {api_url}")
            raise Exception(f"API调用失败: {e}")
    
    def parse_detection_result(self, api_response: Dict) -> List[Dict]:
        """解析API返回的检测结果"""
        try:
            # 获取API返回的内容
            if 'choices' not in api_response or not api_response['choices']:
                raise Exception("API返回格式错误：缺少choices字段")
            
            content = api_response['choices'][0]['message']['content']
            print(f"LLM返回的原始内容: {content}")
            
            # 尝试解析JSON
            detection_data = None
            try:
                detection_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                # 如果不是纯JSON，尝试提取JSON部分
                import re
                # 尝试提取代码块中的JSON
                json_match = re.search(r'```(?:json)?\s*({.*?)(?:```|$)', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        detection_data = json.loads(json_content)
                    except json.JSONDecodeError:
                        # 尝试修复不完整的JSON
                        detection_data = self._fix_incomplete_json(json_content)
                else:
                    # 尝试提取普通的JSON对象
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            detection_data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            # 尝试修复不完整的JSON
                            detection_data = self._fix_incomplete_json(json_match.group())
                    else:
                        # 尝试修复整个内容
                        detection_data = self._fix_incomplete_json(content)
            
            if not detection_data:
                return []
            
            # 处理不同的响应格式
            detections = []
            
            # 格式1: {"objects": [{"label": "", "confidence": 0.0, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}]} - 完整对象格式
            if 'objects' in detection_data:
                for obj in detection_data['objects']:
                    if isinstance(obj, str):
                        # 简单字符串格式
                        detections.append({
                            'label': obj,
                            'confidence': 0.8,
                            'x': 0.1,
                            'y': 0.1,
                            'width': 0.2,
                            'height': 0.2
                        })
                    elif isinstance(obj, dict) and 'label' in obj:
                        # 完整对象格式
                        if all(key in obj for key in ['x', 'y', 'width', 'height']):
                            # 保持原始格式，不转换为bbox
                            detections.append({
                                'label': obj['label'],
                                'confidence': float(obj.get('confidence', 0.8)),
                                'x': float(obj['x']),
                                'y': float(obj['y']),
                                'width': float(obj['width']),
                                'height': float(obj['height'])
                            })
                        else:
                            # 缺少坐标信息，使用默认值
                            detections.append({
                                'label': obj['label'],
                                'confidence': float(obj.get('confidence', 0.8)),
                                'x': 0.1,
                                'y': 0.1,
                                'width': 0.2,
                                'height': 0.2
                            })
            
            # 格式2: {"detections": [{"label": "", "confidence": 0.0, "bbox": []}]} - 完整检测格式
            elif 'detections' in detection_data:
                for detection in detection_data['detections']:
                    if all(key in detection for key in ['label', 'confidence', 'bbox']):
                        # 验证bbox格式
                        bbox = detection['bbox']
                        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                            detections.append({
                                'label': detection['label'],
                                'confidence': float(detection['confidence']),
                                'bbox': [float(x) for x in bbox]
                            })
            
            print(f"解析得到的检测结果: {detections}")
            return detections
            
        except Exception as e:
            print(f"解析检测结果时出错: {e}")
            raise Exception(f"解析检测结果失败: {e}")
    
    def _fix_incomplete_json(self, json_str: str) -> Dict:
        """修复不完整的JSON字符串
        
        Args:
            json_str: 不完整的JSON字符串
            
        Returns:
            Dict: 修复后的JSON对象
        """
        try:
            # 移除多余的空白字符
            json_str = json_str.strip()
            
            # 如果JSON以{开始但没有}结束，尝试添加缺失的括号
            if json_str.startswith('{') and not json_str.endswith('}'):
                # 计算需要的右括号数量
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                missing_braces = open_braces - close_braces
                
                # 添加缺失的右括号
                json_str += '}' * missing_braces
            
            # 如果JSON以[开始但没有]结束，尝试添加缺失的方括号
            if json_str.startswith('[') and not json_str.endswith(']'):
                # 计算需要的右方括号数量
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')
                missing_brackets = open_brackets - close_brackets
                
                # 添加缺失的右方括号
                json_str += ']' * missing_brackets
            
            # 尝试解析修复后的JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 如果仍然失败，尝试更激进的修复
                # 查找最后一个完整的对象或数组
                import re
                
                # 尝试找到objects数组
                objects_match = re.search(r'"objects"\s*:\s*\[(.*)', json_str, re.DOTALL)
                if objects_match:
                    objects_content = objects_match.group(1)
                    # 尝试修复objects数组
                    if not objects_content.strip().endswith(']'):
                        # 移除末尾的逗号和空白
                        objects_content = objects_content.rstrip(',\s\n')
                        
                        # 处理不完整的最后一个对象
                        if objects_content:
                            # 查找所有完整的对象
                            complete_objects = []
                            current_pos = 0
                            brace_count = 0
                            start_pos = -1
                            
                            for i, char in enumerate(objects_content):
                                if char == '{':
                                    if brace_count == 0:
                                        start_pos = i
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0 and start_pos != -1:
                                        # 找到一个完整的对象
                                        obj_str = objects_content[start_pos:i+1]
                                        complete_objects.append(obj_str)
                                        start_pos = -1
                            
                            # 重新构建objects数组
                            if complete_objects:
                                objects_content = ','.join(complete_objects)
                        
                        objects_content += ']'
                    
                    try:
                        return {
                            "objects": json.loads('[' + objects_content)
                        }
                    except json.JSONDecodeError as e:
                        print(f"修复objects数组失败: {e}")
                        # 如果仍然失败，尝试只解析完整的对象
                        try:
                            # 查找第一个完整的对象作为示例
                            first_obj_match = re.search(r'\{[^{}]*"label"[^{}]*"confidence"[^{}]*"x"[^{}]*"y"[^{}]*"width"[^{}]*"height"[^{}]*\}', objects_content)
                            if first_obj_match:
                                sample_obj = json.loads(first_obj_match.group())
                                return {"objects": [sample_obj]}
                        except:
                            pass
                        return {"objects": []}
                
                # 尝试找到detections数组
                detections_match = re.search(r'"detections"\s*:\s*\[(.*)', json_str, re.DOTALL)
                if detections_match:
                    detections_content = detections_match.group(1)
                    # 尝试修复detections数组
                    if not detections_content.strip().endswith(']'):
                        detections_content = detections_content.rstrip(',\s') + ']'
                    
                    return {
                        "detections": json.loads('[' + detections_content)
                    }
                
                # 如果所有修复尝试都失败，返回空结果
                print(f"无法修复JSON: {json_str}")
                return {"objects": []}
                
        except Exception as e:
            print(f"修复JSON时出错: {e}")
            return {"objects": []}
    
    def detect_objects(self, image_path: str, labels: List[str]) -> Tuple[bool, List[Dict], str]:
        """检测图片中的物体
        
        Args:
            image_path: 图片路径
            labels: 要检测的标签列表
            
        Returns:
            Tuple[bool, List[Dict], str]: (是否成功, 检测结果列表, 错误信息)
        """
        try:
            if not self.is_configured():
                return False, [], "未配置LLM或配置无效"
            
            # 生成提示词
            prompt = self.generate_detection_prompt(labels)
            
            # 调用API
            api_response = self.call_vision_api(image_path, prompt)
            
            # 解析结果
            detections = self.parse_detection_result(api_response)
            
            return True, detections, ""
            
        except Exception as e:
            return False, [], str(e)
    
    def detect_objects_from_base64(self, image_data: str, labels: List[str]) -> List[Dict]:
        """从base64图片数据检测物体
        
        Args:
            image_data: base64编码的图片数据（包含data:image/jpeg;base64,前缀）
            labels: 要检测的标签列表
            
        Returns:
            List[Dict]: 检测结果列表
        """
        try:
            if not self.is_configured():
                raise Exception("未配置LLM或配置无效")
            
            # 提取base64数据（去掉data:image/jpeg;base64,前缀）
            if 'base64,' in image_data:
                base64_image = image_data.split('base64,')[1]
            else:
                base64_image = image_data
            
            # 生成提示词
            # 如果labels是字符串列表，直接使用；如果是对象列表，提取name属性
            if labels and hasattr(labels[0], 'name'):
                label_names = [label.name for label in labels]
            else:
                label_names = labels
            prompt = self.generate_detection_prompt(label_names)
            
            # 构建请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.active_config.api_key}"
            }
            
            payload = {
                "model": self.active_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 32000,
                "temperature": 0.1,
                "stream": False
            }
            
            # 构建完整的API URL
            api_url = self._get_api_url()
            
            # 发送请求（禁用代理）
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
                proxies={'http': None, 'https': None}
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # 解析结果
            detections = self.parse_detection_result(result)
            
            # 检测结果已经是正确格式，直接返回
            return detections
            
        except Exception as e:
            raise Exception(f"检测失败: {e}")
    
    def convert_relative_to_absolute(self, detections: List[Dict], image_width: int, image_height: int) -> List[Dict]:
        """将相对坐标转换为绝对坐标
        
        Args:
            detections: 检测结果列表（相对坐标）
            image_width: 图片宽度
            image_height: 图片高度
            
        Returns:
            List[Dict]: 转换后的检测结果（绝对坐标）
        """
        absolute_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            # 相对坐标转绝对坐标
            x = bbox[0] * image_width
            y = bbox[1] * image_height
            width = bbox[2] * image_width
            height = bbox[3] * image_height
            
            absolute_detections.append({
                'label': detection['label'],
                'confidence': detection['confidence'],
                'bbox': [x, y, width, height]
            })
        
        return absolute_detections
    
    def test_connection(self) -> Tuple[bool, str]:
        """测试LLM连接
        
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            if not self.active_config:
                return False, "未找到激活的LLM配置"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.active_config.api_key}"
            }
            
            # 发送简单的测试请求
            payload = {
                "model": self.active_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a test message."
                    }
                ],
                "max_tokens": 10
            }
            
            # 构建完整的API URL
            api_url = self._get_api_url()
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=10,
                proxies={'http': None, 'https': None}
            )
            
            if response.status_code == 200:
                return True, "连接测试成功"
            else:
                return False, f"连接测试失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return False, f"连接测试失败: {e}"