#!/usr/bin/env python3
"""
论文翻译演示脚本

本脚本演示如何使用pdf2zh工具对PDF论文进行翻译
只需运行此脚本，无需额外参数，即可完成PDF翻译
"""

import os
import sys
import time

# 添加PDFMathTranslate目录到系统路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../PDFMathTranslate'))

# 导入必要的模块
from pdf2zh.high_level import translate, download_remote_fonts
from pdf2zh.doclayout import OnnxModel, ModelInstance
from pdf2zh.config import ConfigManager

def main():
    # 设置基本路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    
    # 设置PDF文件路径 - 使用示例PDF
    pdf_path = os.path.join(PROJECT_DIR, "Data/one_page_example.pdf")
    if not os.path.exists(pdf_path):
        print(f"错误：找不到示例PDF文件: {pdf_path}")
        print("请确保Data目录中存在one_page_example.pdf文件")
        return
    
    # 设置输出目录
    output_dir = os.path.join(PROJECT_DIR, "Result/example")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n================= PDF论文翻译演示 =================\n")
    print(f"翻译论文: {pdf_path}")
    print(f"结果将保存至: {output_dir}")
    
    # 检查翻译服务API密钥
    # 默认使用智谱AI服务，可以通过环境变量设置API密钥
    service = os.environ.get("PDF_TRANSLATE_SERVICE", "zhipu")
    api_key_env_name = "ZHIPU_API_KEY"
    
    # 根据不同的服务设置对应的环境变量名
    if service.lower() == "openai":
        api_key_env_name = "OPENAI_API_KEY"
    elif service.lower() == "azure":
        api_key_env_name = "AZURE_API_KEY"
    elif service.lower() == "google":
        api_key_env_name = "GOOGLE_API_KEY"
    
    api_key = os.environ.get(api_key_env_name, "")
    
    if not api_key:
        print(f"\n警告：未设置{service}服务的API密钥！")
        print(f"请设置环境变量{api_key_env_name}")
        print("如果您想使用其他翻译服务，请设置环境变量PDF_TRANSLATE_SERVICE")
        print("支持的服务包括：zhipu, openai, azure, google等")
        return
    
    try:
        print("\n初始化翻译模型...")
        # 加载可用的模型
        model = OnnxModel.load_available()
        
        # 开始计时
        start_time = time.time()
        
        # 设置翻译参数
        params = {
            "files": [pdf_path],
            "output": output_dir,
            "service": service,
            "lang_in": "en",  # 源语言：英文
            "lang_out": "zh",  # 目标语言：中文
            "thread": 4,  # 线程数
            "compatible": True,  # 提高兼容性
            "skip_subset_fonts": False,  # 不跳过字体子集化
            "ignore_cache": False,  # 使用缓存
        }
        
        # 开始翻译
        print("\n开始翻译论文...")
        translate(model=model, **params)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print("\n================= 翻译完成 =================\n")
        print(f"翻译耗时: {elapsed_time:.2f}秒")
        print(f"翻译结果已保存至: {output_dir}")
        print("\n生成的文件包括:")
        print(f"- {output_dir}/example-mono.pdf (单语言翻译PDF)")
        print(f"- {output_dir}/example-dual.pdf (双语言对照PDF)")
        
        print("\n您可以打开生成的PDF文件查看翻译结果。")
        
    except Exception as e:
        print(f"\n翻译过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()