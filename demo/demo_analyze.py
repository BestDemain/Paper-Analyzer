#!/usr/bin/env python3
"""
论文深度解读演示脚本

本脚本演示如何使用PaperAnalyzer类对论文进行深度解读
只需运行此脚本，无需额外参数，即可完成论文分析
"""

import os
import sys
import time

# 添加Analyzer目录到系统路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Analyzer'))

from paper_analyzer import PaperAnalyzer

def main():
    # 设置基本路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    
    # 设置PDF文件路径 - 使用示例PDF
    pdf_path = os.path.join(PROJECT_DIR, "Data/DDPM.pdf")
    if not os.path.exists(pdf_path):
        print(f"错误：找不到示例PDF文件: {pdf_path}")
        print("请确保Data目录中存在DDPM.pdf文件")
        return
    
    # 获取文件名（不含扩展名）用于输出目录
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 设置输出目录
    output_dir = os.path.join(PROJECT_DIR, f"Result/{file_name}_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n================= 论文深度解读演示 =================\n")
    print(f"分析论文: {pdf_path}")
    print(f"结果将保存至: {output_dir}")
    
    # 请在此处替换为您的智谱AI API密钥
    # 您可以通过环境变量设置API密钥，或直接在此处填写
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    
    if not api_key:
        print("\n警告：未设置智谱AI API密钥！")
        print("请通过以下方式之一设置API密钥：")
        print("1. 设置环境变量ZHIPU_API_KEY")
        print("2. 在脚本中直接修改api_key变量的值")
        return
    
    try:
        # 初始化论文分析器
        print("\n初始化论文分析器...")
        analyzer = PaperAnalyzer(
            api_key=api_key,
            model="glm-4-flash",  # 使用glm-4-flash模型
            multimodal_model="glm-4v-flash",  # 使用glm-4v-flash多模态模型
            enable_formula_extraction=True,  # 启用公式提取
            enable_image_extraction=True  # 启用图像提取
        )
        
        # 开始计时
        start_time = time.time()
        
        # 分析论文
        print("\n开始分析论文...")
        result, markdown_content = analyzer.analyze_full_paper(
            pdf_path=pdf_path,
            output_dir=output_dir,
            use_comprehensive_prompt=False,  # 使用分步骤分析
            use_sections=True,  # 使用章节提取功能
            enable_multimodal=True  # 启用多模态分析
        )
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print("\n================= 分析完成 =================\n")
        print(f"分析耗时: {elapsed_time:.2f}秒")
        print(f"分析结果已保存至: {output_dir}")
        print("\n生成的文件包括:")
        print(f"- {output_dir}/analysis_plan.json (解读计划)")
        print(f"- {output_dir}/analysis_result.json (完整分析结果)")
        print(f"- {output_dir}/analysis_result.md (Markdown格式报告)")
        print(f"- {output_dir}/paper_sections.json (论文章节信息)")
        print(f"- {output_dir}/images/ (提取的图像文件)")
        
        print("\n您可以打开Markdown报告查看完整的论文解读结果。")
        
    except Exception as e:
        print(f"\n分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()