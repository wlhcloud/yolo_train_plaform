import base64
import requests
import json
import re
from models import LLMConfig
from urllib.parse import urljoin


class LLMClient:
    """多模态大模型客户端"""
    
    def __init__(self, config_id):
        """
        初始化LLM客户端
        
        Args:
            config_id (int): LLM配置ID
        """
        self.config = LLMConfig.query.get_or_404(config_id)
    
    def encode_image(self, image_path):
        """
        将图片编码为base64格式
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            str: base64编码的图片
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _get_api_url(self):
        """
        获取正确的API URL
        
        Returns:
            str: 完整的API URL
        """
        # 如果URL已经包含chat/completions，则直接使用
        if 'chat/completions' in self.config.api_url:
            return self.config.api_url
        
        # 如果URL以/v1结尾，则添加/chat/completions
        if self.config.api_url.endswith('/v1'):
            return self.config.api_url + '/chat/completions'
        
        # 如果URL以/v1/结尾，则添加chat/completions
        if self.config.api_url.endswith('/v1/'):
            return self.config.api_url + 'chat/completions'
        
        # 否则添加标准的v1/chat/completions路径
        if self.config.api_url.endswith('/'):
            return self.config.api_url + 'v1/chat/completions'
        else:
            return self.config.api_url + '/v1/chat/completions'
    
    def send_example_to_llm(self, example_image_path, example_annotations):
        """
        发送示例给大模型，让其学习标注模式
        
        Args:
            example_image_path (str): 示例图片路径
            example_annotations (list): 示例标注数据
            
        Returns:
            dict: 大模型响应
        """
        # 构建提示词
        prompt = f"""
        请记住以下标注内容的模式。这是一张图片和它对应的标注框：
        
        标注信息：
        {json.dumps(example_annotations, indent=2, ensure_ascii=False)}
        
        请记住这些标注的模式、类别和位置关系，我将在后续请求中要求你对新图片进行相似的标注。
        """
        
        # 编码图片
        base64_image = self.encode_image(example_image_path)
        
        # 构建请求
        headers = {
            "Content-Type": "application/json"
        }
        
        # 如果有API密钥则添加认证头
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model_name or "gpt-4-vision-preview",
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
            "max_tokens": 300
        }
        
        # 获取正确的API URL
        api_url = self._get_api_url()
        
        # 发送请求
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # 抛出HTTP错误
        return response.json()
    
    def annotate_image_with_prompt(self, image_path, prompt_text):
        """
        使用自然语言提示词对图片进行标注
        
        Args:
            image_path (str): 图片路径
            prompt_text (str): 自然语言提示词
            
        Returns:
            dict: 大模型响应
        """
        # 构建更明确的提示词，强制要求返回JSON格式
        full_prompt = f"""
你是一个专业的图像目标检测系统。请对这张图片进行详细的目标检测，识别出所有符合以下描述的目标：

"{prompt_text}"

请严格按照以下要求进行标注：
1. 对于每个识别到的目标，提供准确的类别标签（例如：car, person, dog, tree等，使用英文标签）
2. 提供精确的边界框坐标（x_min, y_min, x_max, y_max，这些值需要归一化到0-1之间）
   - x_min: 边界框左边缘的x坐标（归一化值）
   - y_min: 边界框上边缘的y坐标（归一化值）
   - x_max: 边界框右边缘的x坐标（归一化值）
   - y_max: 边界框下边缘的y坐标（归一化值）
3. 确保边界框紧密贴合目标边缘
4. 不要遗漏图片中的任何相关目标

特别说明：
- 如果用户输入的是中文提示词（如"车"），请识别所有类型的车辆（包括car, truck, bus, ambulance等）
- 如果用户输入的是"人"，请识别所有与人相关的对象（包括person, face, head, body等）
- 尽可能识别出所有符合描述的对象

输出格式要求：
请**严格**以以下JSON格式返回结果，**不要添加任何额外的解释或文本**：
```json
{{
    "objects": [
        {{
            "label": "car",
            "bbox": {{
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.3,
                "y_max": 0.4
            }}
        }}
    ]
}}
```

特别注意：
- 如果没有识别到任何目标，请返回 {{"objects": []}}
- **必须**返回有效的JSON格式，不要使用任何其他格式
- 坐标值必须是0到1之间的浮点数
- 标签应使用英文单词
- **不要**添加任何解释性文字，只返回JSON

严格按照以上格式返回结果！
        """
        
        # 编码图片
        base64_image = self.encode_image(image_path)
        
        # 构建请求
        headers = {
            "Content-Type": "application/json"
        }
        
        # 如果有API密钥则添加认证头
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model_name or "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
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
            "max_tokens": 3000,  # 增加max_tokens以支持更详细的响应
            "temperature": 0.0,   # 降低随机性，提高一致性
            "stream": False      # 确保完整接收响应
        }
        
        # 获取正确的API URL
        api_url = self._get_api_url()
        
        # 发送请求
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)  # 延长超时时间到3分钟
        response.raise_for_status()  # 抛出HTTP错误
        
        # 获取响应文本
        response_text = response.text
        print(f"LLM原始响应文本: {response_text}")
        
        # 尝试解析JSON响应
        try:
            result = response.json()
            return result
        except json.JSONDecodeError:
            pass  # 继续尝试其他方法
        
        # 如果直接解析失败，尝试从响应中提取JSON
        extracted_json = self._extract_json_from_response(response_text)
        if extracted_json:
            return extracted_json
        
        # 如果所有方法都失败，返回原始响应和错误信息
        return {
            "raw_response": response_text,
            "status_code": response.status_code,
            "error": "Response is not valid JSON and no JSON could be extracted"
        }
    
    def _extract_json_from_response(self, response_text):
        """
        从响应文本中提取JSON
        
        Args:
            response_text (str): 响应文本
            
        Returns:
            dict: 提取的JSON对象，如果无法提取则返回None
        """
        # 首先尝试直接解析整个响应
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 尝试查找代码块中的JSON
        code_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
        code_block_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        
        for match in code_block_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 尝试从OpenAI格式响应中提取内容
        try:
            response_data = json.loads(response_text)
            if isinstance(response_data, dict) and 'choices' in response_data:
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    # 尝试解析内容为JSON
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        pass
                    
                    # 尝试从内容中提取JSON代码块
                    code_block_matches = re.findall(code_block_pattern, content, re.DOTALL)
                    for match in code_block_matches:
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
                    
                    # 尝试查找花括号包围的JSON对象
                    json_pattern = r'\{(?:[^{}]|(?R))*\}'
                    plain_matches = re.findall(json_pattern, content, re.DOTALL)
                    
                    for match in plain_matches:
                        try:
                            parsed_json = json.loads(match)
                            if isinstance(parsed_json, dict):
                                return parsed_json
                        except json.JSONDecodeError:
                            continue
        except json.JSONDecodeError:
            pass
        
        # 尝试查找花括号包围的JSON对象
        json_pattern = r'\{(?:[^{}]|(?R))*\}'
        plain_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in plain_matches:
            try:
                parsed_json = json.loads(match)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                continue
        
        # 如果没找到有效的JSON对象，返回None
        return None
    
    def annotate_next_image(self, image_path):
        """
        对下一张图片进行自动标注（基于之前学习的示例）
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            dict: 大模型响应
        """
        # 构建提示词
        prompt = """
你是一个专业的图像目标检测系统。基于之前学习的标注模式，请对这张新图片进行目标检测。

请严格按照以下要求进行标注：
1. 对于每个识别到的目标，提供准确的类别标签
2. 提供精确的边界框坐标（x_min, y_min, x_max, y_max，这些值需要归一化到0-1之间）
3. 确保边界框紧密贴合目标边缘
4. 不要遗漏图片中的任何相关目标

输出格式要求：
请**严格**以以下JSON格式返回结果，**不要添加任何额外的解释或文本**：
```json
{
    "objects": [
        {
            "label": "目标类别",
            "bbox": {
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.3,
                "y_max": 0.4
            }
        }
    ]
}
```

特别注意：
- 如果没有识别到任何目标，请返回 {"objects": []}
- **必须**返回有效的JSON格式，不要使用任何其他格式
- 坐标值必须是0到1之间的浮点数
- 标签应使用英文单词
- **不要**添加任何解释性文字，只返回JSON

严格按照以上格式返回结果！
        """
        
        # 编码图片
        base64_image = self.encode_image(image_path)
        
        # 构建请求
        headers = {
            "Content-Type": "application/json"
        }
        
        # 如果有API密钥则添加认证头
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model_name or "gpt-4-vision-preview",
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
            "max_tokens": 3000,  # 增加max_tokens以支持更详细的响应
            "temperature": 0.0,   # 降低随机性，提高一致性
            "stream": False      # 确保完整接收响应
        }
        
        # 获取正确的API URL
        api_url = self._get_api_url()
        
        # 发送请求
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)  # 延长超时时间到3分钟
        response.raise_for_status()  # 抛出HTTP错误
        
        # 获取响应文本
        response_text = response.text
        print(f"LLM原始响应文本: {response_text}")
        
        # 尝试解析JSON响应
        try:
            result = response.json()
            return result
        except json.JSONDecodeError:
            pass  # 继续尝试其他方法
        
        # 如果直接解析失败，尝试从响应中提取JSON
        extracted_json = self._extract_json_from_response(response_text)
        if extracted_json:
            return extracted_json
        
        # 如果所有方法都失败，返回原始响应和错误信息
        return {
            "raw_response": response_text,
            "status_code": response.status_code,
            "error": "Response is not valid JSON and no JSON could be extracted"
        }