#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查项目1第22张图片的标注数据
对比手动标注与AI检测的坐标差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models import db, Image, Annotation, Project
from services.llm_service import LLMService
import json
import base64

def get_image22_info():
    """获取项目1第22张图片的详细信息"""
    app = create_app()
    
    with app.app_context():
        # 查找项目1
        project = Project.query.filter_by(id=1).first()
        if not project:
            print("❌ 项目1不存在")
            return None
            
        print(f"📁 项目信息: {project.name}")
        
        # 查找第22张图片（按ID排序）
        images = Image.query.filter_by(project_id=1).order_by(Image.id).all()
        
        if len(images) < 22:
            print(f"❌ 项目1只有{len(images)}张图片，没有第22张")
            return None
            
        image22 = images[21]  # 第22张图片（索引21）
        
        print(f"\n🖼️ 第22张图片信息:")
        print(f"   ID: {image22.id}")
        print(f"   文件名: {image22.original_filename}")
        print(f"   路径: {image22.path}")
        print(f"   尺寸: {image22.width} x {image22.height}")
        
        # 获取标注数据
        annotations = Annotation.query.filter_by(image_id=image22.id).all()
        
        print(f"\n📍 当前标注数据 ({len(annotations)}个):")
        manual_annotations = []
        for i, ann in enumerate(annotations, 1):
            print(f"   {i}. 标签ID: {ann.label_id}, 坐标: ({ann.x:.6f}, {ann.y:.6f}), 尺寸: {ann.width:.6f} x {ann.height:.6f}")
            manual_annotations.append({
                'label_id': ann.label_id,
                'x': ann.x,
                'y': ann.y,
                'width': ann.width,
                'height': ann.height
            })
            
        return {
            'image': image22,
            'annotations': manual_annotations,
            'project': project
        }

def test_ai_detection(image_info, app):
    """测试AI检测并对比结果"""
    if not image_info:
        return
        
    with app.app_context():
        image = image_info['image']
        manual_annotations = image_info['annotations']
        
        # 构建图片完整路径
        image_path = os.path.join('/Users/boyan/yolotrain/static', image.path)
        
        if not os.path.exists(image_path):
            print(f"❌ 图片文件不存在: {image_path}")
            return
            
        print(f"\n🤖 开始AI检测...")
        print(f"   图片路径: {image_path}")
        
        try:
            # 读取图片并转换为base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # 使用LLM服务进行检测
            llm_service = LLMService()
            
            # 重新获取项目标签（在应用上下文中）
            project = Project.query.filter_by(id=1).first()
            label_names = [label.name for label in project.labels]
            
            print(f"   使用标签: {label_names}")
            
            # 调用AI检测
            ai_result = llm_service.detect_objects_from_base64(base64_data, label_names)
            
            print(f"\n🎯 AI检测结果:")
            if ai_result and isinstance(ai_result, list):
                ai_detections = ai_result
                print(f"   检测到 {len(ai_detections)} 个对象")
                
                for i, detection in enumerate(ai_detections, 1):
                    print(f"   {i}. 标签: {detection.get('label', 'N/A')}, 置信度: {detection.get('confidence', 0):.4f}")
                    print(f"      坐标: ({detection.get('x', 0):.6f}, {detection.get('y', 0):.6f}), 尺寸: {detection.get('width', 0):.6f} x {detection.get('height', 0):.6f}")
                    
                # 对比分析
                print(f"\n📊 坐标对比分析:")
                print(f"   手动标注: {len(manual_annotations)} 个")
                print(f"   AI检测: {len(ai_detections)} 个")
                
                if len(manual_annotations) == len(ai_detections):
                    print(f"\n   详细对比:")
                    for i in range(len(manual_annotations)):
                        manual = manual_annotations[i]
                        ai = ai_detections[i] if i < len(ai_detections) else None
                        
                        if ai:
                            x_diff = abs(manual['x'] - ai.get('x', 0))
                            y_diff = abs(manual['y'] - ai.get('y', 0))
                            w_diff = abs(manual['width'] - ai.get('width', 0))
                            h_diff = abs(manual['height'] - ai.get('height', 0))
                            
                            print(f"   对象{i+1}:")
                            print(f"     X坐标差异: {x_diff:.6f} ({x_diff*100:.2f}%)")
                            print(f"     Y坐标差异: {y_diff:.6f} ({y_diff*100:.2f}%)")
                            print(f"     宽度差异: {w_diff:.6f} ({w_diff*100:.2f}%)")
                            print(f"     高度差异: {h_diff:.6f} ({h_diff*100:.2f}%)")
                else:
                    print(f"   ⚠️ 检测数量不匹配，无法进行一对一对比")
                    
            elif ai_result and isinstance(ai_result, dict) and 'detections' in ai_result:
                ai_detections = ai_result['detections']
                print(f"   检测到 {len(ai_detections)} 个对象 (字典格式)")
                # 处理字典格式的逻辑...
            else:
                print(f"   ❌ AI检测失败或无结果")
                print(f"   原始返回: {ai_result}")
                
        except Exception as e:
            print(f"❌ AI检测出错: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("🔍 检查项目1第22张图片的标注数据\n")
    
    # 创建应用实例
    app = create_app()
    
    # 获取图片信息
    image_info = get_image22_info()
    
    if image_info:
        # 测试AI检测
        test_ai_detection(image_info, app)
    
    print("\n✅ 检查完成")

if __name__ == '__main__':
    main()