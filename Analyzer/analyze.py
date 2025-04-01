#!/usr/bin/env python3
"""
论文深入解读示例脚本

本脚本展示如何使用PaperAnalyzer类对论文进行深入解读，支持按章节分析
"""

import os
import json
import argparse
import time
from paper_analyzer import PaperAnalyzer

def main():
    parser = argparse.ArgumentParser(description="使用智谱AI进行论文深入解读示例")
    parser.add_argument("--api_key", type=str, default=None, help="智谱AI的API密钥，如果不提供则使用环境变量ZHIPU_API_KEY")
    parser.add_argument("--model", type=str, default="glm-4-flash", help="使用的模型名称，默认为glm-4-flash")
    parser.add_argument("--pdf", type=str, default=None, help="论文文件路径")
    parser.add_argument("--output", type=str, default=None, help="分析结果输出路径")
    parser.add_argument("--mode", type=str, choices=["comprehensive", "step_by_step", "both"], default="step_by_step", 
                        help="分析模式：comprehensive(综合分析)、step_by_step(逐步分析)或both(两种方式都执行)")
    parser.add_argument("--use_sections", default=True, action="store_true", help="是否使用章节提取功能进行分析，可提高分析精度")
    
    args = parser.parse_args()
    
    # 如果命令行没有提供API密钥，则尝试从环境变量获取
    api_key = args.api_key or os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("请提供智谱AI的API密钥，可以通过--api_key参数或ZHIPU_API_KEY环境变量设置")
    
    # 初始化论文分析器
    analyzer = PaperAnalyzer(api_key=api_key, model=args.model)
    
    # 确定PDF文件路径
    pdf_path = args.pdf or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Data/DDPM.pdf")
    file_name, _ = os.path.splitext(os.path.basename(pdf_path))
    print(f"开始分析论文: {pdf_path}")
    print(f"分析模式: {args.mode}")
    
    # 根据模式执行不同的分析方法
    if args.mode == "comprehensive" or args.mode == "both":
        print("\n=== 使用综合提示词一次性分析论文 ===")
        start_time = time.time()
        
        # 使用综合提示词一次性分析
        output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Result/"+file_name+"_comprehensive")
        result, markdown_content = analyzer.analyze_full_paper(pdf_path, output_dir, use_comprehensive_prompt=True)
        
        elapsed_time = time.time() - start_time
        print(f"综合分析完成，耗时: {elapsed_time:.2f}秒")
        print(f"分析结果已保存至: {output_dir}")
        
        # 打印部分分析结果
        if "comprehensive_analysis" in result:
            print("\n综合分析结果预览:")
            preview = result["comprehensive_analysis"][:500] + "..." if len(result["comprehensive_analysis"]) > 500 else result["comprehensive_analysis"]
            print(preview)
    
    if args.mode == "step_by_step" or args.mode == "both":
        start_time = time.time()
        
        # 使用分步骤详细分析
        output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Result/"+file_name+"_step_by_step")
        result, markdown_content = analyzer.analyze_full_paper(pdf_path, output_dir, use_comprehensive_prompt=False, use_sections=args.use_sections)
        
        elapsed_time = time.time() - start_time
        print("\n================= 分析完成 =================\n")
        print(f"分步骤分析完成，耗时: {elapsed_time:.2f}秒")
        print(f"分析结果已保存至: {output_dir}")
        
        # # 打印分析计划
        # if "analysis_plan" in result:
        #     print("\n分析计划:")
        #     if "raw_plan" in result["analysis_plan"]:
        #         print(result["analysis_plan"]["raw_plan"][:300] + "...")
        #     else:
        #         for step, details in list(result["analysis_plan"].items())[:3]:  # 只显示前3个步骤
        #             if isinstance(details, dict) and "instruction" in details:
        #                 print(f"- {step}: {details['instruction'][:100]}...")
        #             elif isinstance(details, str):
        #                 print(f"- {step}: {details[:100]}...")
        
        # # 打印部分分析结果
        # if "analysis_results" in result:
        #     print("\n部分分析结果:")
        #     for section, content in list(result["analysis_results"].items())[:2]:  # 只显示前2个部分
        #         print(f"\n== {section} ==\n")
        #         preview = content[:300] + "..." if len(content) > 300 else content
        #         print(preview)
    
    # 如果生成了章节信息文件，提示用户
    if args.mode == "step_by_step" or args.mode == "both":
        step_by_step_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Result/"+file_name+"_step_by_step")
        sections_path = os.path.join(step_by_step_dir if args.output is None else args.output, "paper_sections.json")
    
    print("\n论文分析完成！查看完整分析结果请打开输出文件。")

if __name__ == "__main__":
    main()