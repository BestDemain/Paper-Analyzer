#!/usr/bin/env python3
"""
Markdown报告生成器

将论文分析结果转换为Markdown格式的报告
"""

import re

def format_latex_equations(text: str) -> str:
    """将LaTeX格式的公式转换为Markdown支持的格式
    
    Args:
        text: 包含LaTeX公式的文本
        
    Returns:
        转换后的文本
    """
    # 处理行间公式 \[ ... \] -> $$ ... $$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    
    # 处理行内公式 \( ... \) -> $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # 处理其他常见的LaTeX公式格式 $$ ... $$ (已经是正确格式，但可能有空格问题)
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'$$\1$$', text, flags=re.DOTALL)
    
    # 处理其他常见的LaTeX公式格式 $ ... $ (已经是正确格式，但可能有空格问题)
    text = re.sub(r'\$\s*(.*?)\s*\$', r'$\1$', text, flags=re.DOTALL)
    
    return text


def generate_markdown_report(analysis_result: dict) -> str:
    """将论文分析结果转换为Markdown格式的报告
    
    Args:
        analysis_result: 论文分析结果字典
        
    Returns:
        Markdown格式的报告内容
    """
    markdown_content = []
    
    # 添加标题
    markdown_content.append("# 论文深度解读报告\n")
    
    # 如果是综合分析结果
    if "comprehensive_analysis" in analysis_result:
        content = format_latex_equations(analysis_result["comprehensive_analysis"])
        markdown_content.append(content)
        return "\n".join(markdown_content)
    
    # 如果是分步骤分析结果
    if "analysis_results" in analysis_result:
        # 获取提取的图像列表（如果有）
        extracted_images = analysis_result.get("extracted_images", [])
        
        for section, content in analysis_result["analysis_results"].items():
            # 添加二级标题
            markdown_content.append(f"\n## {section}\n")
            
            # 处理内容中可能的JSON格式
            try:
                # 尝试解析JSON
                import json
                section_content = json.loads(content)
                
                # 如果是字典，格式化输出
                if isinstance(section_content, dict):
                    for key, value in section_content.items():
                        markdown_content.append(f"### {key}\n")
                        # 处理公式
                        formatted_value = format_latex_equations(str(value))
                        markdown_content.append(f"{formatted_value}\n")
                else:
                    # 处理公式
                    formatted_content = format_latex_equations(content)
                    markdown_content.append(formatted_content)
            except:
                # 如果不是JSON格式，直接添加内容，但处理其中的公式
                formatted_content = format_latex_equations(content)
                markdown_content.append(formatted_content)
            
            # 获取章节与图像的关联信息（如果有）
            section_images = analysis_result.get("section_images", {})
            
            # 确定当前章节的相关图像
            current_section_images = []
            if section in section_images:
                # 如果有明确关联的图像，使用关联的图像
                current_section_images = section_images[section]
            elif (section == "实验结果" or "图" in section or "表" in section) and extracted_images:
                # 对于实验结果或包含图表的部分，如果没有明确关联，使用所有图像
                current_section_images = extracted_images
            
            # 如果有相关图像，添加到markdown中
            if current_section_images:
                # 添加图像部分标题
                markdown_content.append("\n### 相关图像\n")
                markdown_content.append("以下是论文中的相关图像：\n")
                
                # 添加图像引用
                for i, img_path in enumerate(current_section_images):
                    # 获取图像文件名
                    import os
                    img_filename = os.path.basename(img_path)
                    # 获取相对路径（假设图像在images目录下）
                    img_rel_path = f"images/{img_filename}"
                    
                    # 尝试获取图像标题（如果有）
                    img_caption = ""
                    if "image_captions" in analysis_result and img_path in analysis_result["image_captions"]:
                        img_caption = analysis_result["image_captions"][img_path]
                    else:
                        img_caption = img_filename
                    
                    # 添加图像引用
                    markdown_content.append(f"![图{i+1}]({img_rel_path})\n")
                    markdown_content.append(f"*图{i+1}: {img_caption}*\n\n")
    
    return "\n".join(markdown_content)