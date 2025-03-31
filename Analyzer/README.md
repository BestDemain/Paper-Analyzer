# 论文深入解读工具

这是一个使用智谱AI大模型对学术论文进行深入解读的工具。该工具可以自动分析PDF格式的学术论文，提取关键信息，并生成详细的解读报告。

## 功能特点

- 自动提取论文文本内容
- 生成详细的论文分析计划
- 分析论文的各个关键部分（摘要、研究背景、方法、创新点、实验结果等）
- 输出结构化的解读报告（支持JSON格式和Markdown格式）
- 支持生成美观的Markdown格式分析报告，便于阅读和分享

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 使用示例脚本

```bash
python analyze_example.py --api_key "your_zhipu_api_key" --output "analysis_result.json" [--markdown]
```

或者设置环境变量：

```bash
set ZHIPU_API_KEY=your_zhipu_api_key
python analyze_example.py --output "analysis_result.json"
```

### 参数说明

- `--api_key`: 智谱AI的API密钥（如果不提供，将尝试从环境变量ZHIPU_API_KEY获取）
- `--model`: 使用的模型名称，默认为glm-4-flash
- `--output`: 分析结果输出路径，默认为analysis_result.json
- `--markdown`: 可选参数，添加此参数将生成Markdown格式的解读报告

### 在自己的代码中使用

```python
from paper_analyzer import PaperAnalyzer

# 初始化分析器
analyzer = PaperAnalyzer(api_key="your_zhipu_api_key", model="glm-4-flash")

# 分析论文
result = analyzer.analyze_full_paper("path/to/your/paper.pdf", "output.json")

# 使用分析结果
print(result["analysis_results"]["摘要解读"])
```

## 分析结果格式

工具支持两种输出格式：

### JSON格式

分析结果以JSON格式保存，包含以下主要部分：

- `paper_path`: 论文文件路径
- `analysis_plan`: 论文分析计划
- `analysis_results`: 各部分的分析结果
  - 基本信息
  - 摘要解读
  - 研究背景
  - 研究方法
  - 创新点
  - 实验结果
  - 结论
  - 等等

### Markdown格式

当使用 `--markdown`参数时，工具会额外生成一个Markdown格式的解读报告，文件名为输出文件名去掉.json后缀并添加.md后缀。Markdown报告具有以下特点：

- 结构清晰，便于阅读和分享
- 自动处理JSON内容的格式化
- 支持多级标题和章节组织
- 保持与JSON输出相同的分析深度和内容质量

## 注意事项

- 需要有效的智谱AI API密钥
- 处理大型论文时可能需要较长时间
- 分析结果的质量取决于智谱AI模型的能力
