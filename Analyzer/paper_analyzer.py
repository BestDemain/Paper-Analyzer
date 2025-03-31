import os
import json
import fitz  # PyMuPDF
import logging
import re
import base64
from typing import Dict, List, Any, Optional, Tuple
from zhipuai import ZhipuAI
from prompt_templates import *
from markdown_generator import generate_markdown_report
from paper_section_extractor import PaperSectionExtractor
from formula_extractor import FormulaExtractor
from image_extractor import ImageExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperAnalyzer:
    """论文深度解读工具，使用智谱AI进行论文内容分析"""
    
    def __init__(self, api_key: str, model: str = "glm-4-flash", enable_formula_extraction: bool = True, enable_image_extraction: bool = True, multimodal_model: str = "glm-4v-flash"):
        """初始化论文分析器
        
        Args:
            api_key: 智谱AI的API密钥
            model: 使用的模型名称，默认为glm-4-flash
            enable_formula_extraction: 是否启用公式提取功能，默认为True
            enable_image_extraction: 是否启用图像提取功能，默认为True
            multimodal_model: 多模态模型名称，默认为glm-4v-flash
        """
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.multimodal_model = multimodal_model
        self.max_tokens = 50000
        self.section_extractor = PaperSectionExtractor(enable_formula_extraction=enable_formula_extraction)
        self.formula_extractor = FormulaExtractor() if enable_formula_extraction else None
        self.enable_formula_extraction = enable_formula_extraction
        self.enable_image_extraction = enable_image_extraction
        self.image_extractor = ImageExtractor() if enable_image_extraction else None
        self.extracted_images = []  # 存储提取的图像路径
        print("\n================= 准备工作 =================\n")
        logger.info(f"初始化PaperAnalyzer，使用模型: {model}，多模态模型: {multimodal_model}，公式提取功能: {'启用' if enable_formula_extraction else '禁用'}，图像提取功能: {'启用' if enable_image_extraction else '禁用'}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本内容，如果启用了公式提取功能，则保留数学公式
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容，如果启用了公式提取功能，则包含公式标记
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 如果启用了公式提取功能，使用FormulaExtractor提取带公式的文本
        if self.enable_formula_extraction and self.formula_extractor:
            try:
                logger.info(f"使用公式提取功能从{pdf_path}提取文本")
                text = self.formula_extractor.extract_text_with_formulas(pdf_path)
                logger.info(f"成功从{pdf_path}提取带公式的文本，共识别{len(self.formula_extractor.formulas)}个公式")
                return text
            except Exception as formula_error:
                logger.warning(f"使用公式提取功能失败: {str(formula_error)}，将使用普通文本提取方法")
                # 如果公式提取失败，回退到普通文本提取方法
        
        # 使用PyMuPDF提取文本
        try:
            doc = None
            try:
                doc = fitz.open(pdf_path)
                text = ""
                
                # 获取页数并记录日志
                page_count = len(doc)
                logger.info(f"PDF文件{pdf_path}共有{page_count}页")
                
                # 逐页提取文本
                for page_num in range(page_count):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        text += page_text
                    except Exception as page_error:
                        logger.warning(f"提取第{page_num+1}页时出错: {str(page_error)}，跳过此页")
                        continue
                
                logger.info(f"成功从{pdf_path}提取文本")
                return text
            except Exception as e:
                logger.error(f"使用PyMuPDF提取PDF文本时出错: {str(e)}")
                if doc:
                    try:
                        doc.close()
                    except:
                        pass
                raise
            finally:
                # 确保文档被关闭
                if doc:
                    try:
                        doc.close()
                    except:
                        pass
        except Exception as mupdf_error:
            # 如果PyMuPDF失败，尝试使用备选方法
            logger.warning(f"PyMuPDF提取失败，尝试使用备选方法: {str(mupdf_error)}")
            
            # 尝试使用PyPDF2作为备选方法
            try:
                import PyPDF2
                logger.info(f"尝试使用PyPDF2提取PDF文本: {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    page_count = len(reader.pages)
                    logger.info(f"PDF文件{pdf_path}共有{page_count}页(PyPDF2)")
                    
                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            text += page_text
                        except Exception as page_error:
                            logger.warning(f"PyPDF2提取第{i+1}页时出错: {str(page_error)}，跳过此页")
                            continue
                    
                    if text.strip():
                        logger.info(f"成功使用PyPDF2从{pdf_path}提取文本")
                        return text
                    else:
                        logger.warning("PyPDF2提取的文本为空")
                        raise ValueError("提取的文本为空")
            except Exception as pypdf_error:
                logger.error(f"备选方法PyPDF2也失败: {str(pypdf_error)}")
                # 返回一个简单的错误提示文本，以便分析过程可以继续
                return f"[无法提取PDF文本，请检查PDF文件格式是否正确。错误信息: {str(mupdf_error)}, PyPDF2错误: {str(pypdf_error)}]"
    
    def generate_analysis_plan(self, paper_text: str) -> Dict[str, Any]:
        """生成论文分析计划
        
        Args:
            paper_text: 论文文本内容
            
        Returns:
            分析计划，包含各个步骤
        """
        system_prompt = ANALYSIS_PLAN_PROMPT.substitute()
        
        try:
            # 截取论文前10000个字符作为输入，避免超出token限制
            truncated_text = paper_text
            
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,  # 使用较低的温度以获得更确定性的输出
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""以下是一篇学术论文的内容，请用中文制定一份详细深入的解读计划。请重点关注需要解读的方向和任务，而不是现在解读。
                     其中应强调对该论文提出的方法的解读任务。
                     如果论文涉及公式的推导，在计划中规划对公式推导的再现和解读任务。
                     请规划对论文中关键图表的解读任务。
                     请以json格式输出字典，键为解读任务（中文字符串），值为具体步骤（中文字符串），如：{"{"}"name of task 1": "detailed instruction", ...{"}"}，不要输出任何多余内容：\n
                     {truncated_text}"""}
                ]
            )
            
            response_text = completion.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            print("\n================= 解读计划 =================\n", response_text)
            
            # 尝试解析JSON响应
            try:
                plan = json.loads(response_text)
                logger.info("成功生成论文分析计划")
                return plan
            except json.JSONDecodeError:
                # 如果返回的不是有效JSON，则将文本作为字符串返回
                logger.warning("返回的分析计划不是有效JSON格式，返回原始文本")
                return {"raw_plan": response_text}
                
        except Exception as e:
            logger.error(f"生成分析计划时出错: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """从PDF中提取图像并保存到指定目录
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 图像输出目录，默认为None（使用相对路径'./image'）
            
        Returns:
            提取的图像文件路径列表
        """
        if not self.enable_image_extraction or not self.image_extractor:
            logger.warning("图像提取功能未启用")
            return []
            
        try:
            # 使用ImageExtractor提取图像
            self.extracted_images = self.image_extractor.extract_images_from_pdf(pdf_path, output_dir)
            logger.info(f"从{pdf_path}提取了{len(self.extracted_images)}个图像")
            return self.extracted_images
        except Exception as e:
            logger.error(f"提取PDF图像时出错: {str(e)}")
            return []
    
    def analyze_paper_section(self, paper_text: str, section: str, instruction: str, related_images: List[str] = None) -> str:
        """分析论文的特定部分
        
        Args:
            paper_text: 论文文本内容
            section: 要分析的部分名称
            instruction: 分析指导说明
            related_images: 与该部分相关的图像路径列表，默认为None
            
        Returns:
            分析结果
        """
        try:
            # 对于公式推导任务，使用分两部分处理的方法，不使用图片分析
            if section == "再现公式推导" or "公式" in section or "推导" in section:
                logger.info(f"执行公式推导任务，将分两部分处理，不包含图片分析")
                return self.analyze_formula_in_two_parts(paper_text, section, instruction)
            
            # 对于实验结果部分，如果有相关图像，使用多模态分析
            if (section == "实验结果" or "图" in section or "表" in section) and related_images and self.enable_image_extraction:
                logger.info(f"执行多模态分析任务，包含{len(related_images)}张图像")
                return self.analyze_with_images(paper_text, section, instruction, related_images)
            
            # 对于其他任务，截取论文前self.max_tokens个字符作为输入，避免超出token限制
            truncated_text = paper_text[:self.max_tokens] + "..." if len(paper_text) > self.max_tokens else paper_text
            
            # 处理文本中的公式标记
            has_formulas = False
            formula_instruction = ""
            if self.enable_formula_extraction and self.formula_extractor and re.search(r"\{v\d+\}", truncated_text):
                has_formulas = True
                # 将公式标记替换为LaTeX格式
                truncated_text = self.formula_extractor.replace_formula_markers(truncated_text)
                formula_instruction = "\n注意：文本中包含数学公式，以$$公式$$格式标记。请在分析中特别关注这些公式的含义和作用。"
                logger.info(f"文本中包含数学公式，已替换为LaTeX格式")
            
            # 根据部分名称选择合适的提示词模板
            prompt_template = None
            if section == "基本信息":
                prompt_template = BASIC_INFO_PROMPT
            elif section == "摘要解读":
                prompt_template = ABSTRACT_ANALYSIS_PROMPT
            elif section == "研究背景":
                prompt_template = BACKGROUND_ANALYSIS_PROMPT
            elif section == "研究方法":
                prompt_template = METHODOLOGY_ANALYSIS_PROMPT
            elif section == "创新点":
                prompt_template = INNOVATION_ANALYSIS_PROMPT
            elif section == "实验结果":
                prompt_template = RESULTS_ANALYSIS_PROMPT
            elif section == "关键图表解读" or section == "图表解读" or "图表" in section:
                prompt_template = CHART_ANALYSIS_PROMPT
            elif section == "结论":
                prompt_template = CONCLUSION_ANALYSIS_PROMPT
            elif section == "局限性":
                prompt_template = LIMITATIONS_ANALYSIS_PROMPT
            elif section == "相关工作":
                prompt_template = RELATED_WORK_ANALYSIS_PROMPT
            elif section == "综合评价":
                prompt_template = OVERALL_EVALUATION_PROMPT
            else:
                # 如果没有匹配的模板，使用自定义指令
                system_prompt = f"""你是一位专业的学术论文分析专家。请用中文对以下学术论文的{section}部分进行深入分析。
                {instruction}{formula_instruction}
                请提供详细、专业且有见解的分析。尽量列出并解读论文中的公式（使用LATEX格式：$$公式$$）。"""
            
            # 如果有匹配的模板，使用模板生成提示词
            if prompt_template:
                system_prompt = prompt_template.substitute()
                # 如果有公式，添加公式处理指令
                if has_formulas:
                    system_prompt += formula_instruction
            
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"以下是论文内容：\n\n{truncated_text}"}
                ]
            )
            
            # 获取大模型返回的内容
            response_content = completion.choices[0].message.content
            
            # 处理大模型返回内容中的公式
            try:
                from markdown_generator import format_latex_equations
                # 使用format_latex_equations函数处理公式
                response_content = format_latex_equations(response_content)
                logger.info(f"已处理大模型返回内容中的公式")
            except Exception as format_error:
                logger.warning(f"处理大模型返回内容中的公式时出错: {str(format_error)}")
            
            return response_content
            
        except Exception as e:
            logger.error(f"分析论文{section}部分时出错: {str(e)}")
            raise
            
    def analyze_with_images(self, paper_text: str, section: str, instruction: str, image_paths: List[str]) -> str:
        """使用多模态模型分析论文内容和图像
        
        Args:
            paper_text: 论文文本内容
            section: 要分析的部分名称
            instruction: 分析指导说明
            image_paths: 图像文件路径列表
            
        Returns:
            多模态分析结果
        """
        try:
            # 截取论文前self.max_tokens个字符作为输入，避免超出token限制
            truncated_text = paper_text[:self.max_tokens] + "..." if len(paper_text) > self.max_tokens else paper_text
            
            # 处理文本中的公式标记
            if self.enable_formula_extraction and self.formula_extractor and re.search(r"\{v\d+\}", truncated_text):
                # 将公式标记替换为LaTeX格式
                truncated_text = self.formula_extractor.replace_formula_markers(truncated_text)
                logger.info(f"文本中包含数学公式，已替换为LaTeX格式")
            
            # 构建多模态消息
            messages = []
            
            # 系统消息
            system_prompt = f"""你是一位专业的学术论文分析专家，擅长分析学术论文中的图表和实验结果。
            请用中文对以下学术论文的{section}部分进行深入分析，特别关注论文中的图表内容。
            {instruction}
            请提供详细、专业且有见解的分析，包括：
            1. 图表的主要内容和目的
            2. 图表中展示的关键数据和趋势
            3. 图表如何支持论文的论点和结论
            4. 图表的创新点和局限性
            
            请确保分析全面、专业，并与论文文本内容相结合。"""
            
            messages.append({"role": "system", "content": system_prompt})
            
            # 用户消息（包含文本和图像）
            content = []
            
            # 添加文本内容
            content.append({
                "type": "text",
                "text": f"以下是论文内容：\n\n{truncated_text}\n\n请分析这篇论文的{section}部分，特别关注下面的图表内容。"
            })
            
            # 添加图像内容（最多处理5张图像，避免超出token限制）
            for i, image_path in enumerate(image_paths[:5]):
                try:
                    # 获取图像的base64编码
                    img_base64 = self.image_extractor.get_image_base64(image_path)
                    if img_base64:
                        # 获取图像标题（如果有）
                        caption = self.image_extractor.get_image_caption(image_path) or f"图{i+1}"
                        
                        # 添加图像到内容中
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        })
                        
                        # 在图像后添加标题说明
                        content.append({
                            "type": "text",
                            "text": f"图片说明: {caption}"
                        })
                        
                        logger.info(f"添加图像到多模态分析: {image_path}")
                except Exception as img_error:
                    logger.warning(f"处理图像{image_path}时出错: {str(img_error)}")
            
            messages.append({"role": "user", "content": content})
            
            # 调用多模态模型
            logger.info(f"调用多模态模型{self.multimodal_model}进行分析")
            completion = self.client.chat.completions.create(
                model=self.multimodal_model,
                temperature=0.5,
                messages=messages
            )
            
            # 获取多模态模型返回的内容
            response_content = completion.choices[0].message.content
            
            # 处理返回内容中的公式
            try:
                from markdown_generator import format_latex_equations
                # 使用format_latex_equations函数处理公式
                response_content = format_latex_equations(response_content)
                logger.info(f"已处理多模态分析返回内容中的公式")
            except Exception as format_error:
                logger.warning(f"处理多模态分析返回内容中的公式时出错: {str(format_error)}")
            
            # 在分析结果中添加图片引用
            if image_paths:
                # 添加图片部分
                image_section = "\n\n### 相关图像\n\n以下是论文中的相关图像：\n\n"
                
                # 添加图片引用
                for i, image_path in enumerate(image_paths[:5]):
                    try:
                        # 获取图像文件名
                        img_filename = os.path.basename(image_path)
                        # 获取图像标题（如果有）
                        caption = self.image_extractor.get_image_caption(image_path) or f"图{i+1}"
                        
                        # 添加图像引用（使用相对路径，假设图像在images目录下）
                        img_rel_path = f"images/{img_filename}"
                        image_section += f"![图{i+1}]({img_rel_path})\n\n*图{i+1}: {caption}*\n\n"
                        
                        logger.info(f"在多模态分析结果中添加图像引用: {image_path}")
                    except Exception as img_error:
                        logger.warning(f"在多模态分析结果中添加图像引用时出错: {str(img_error)}")
                
                # 将图片部分添加到分析结果中
                response_content += image_section
            
            return response_content
            
        except Exception as e:
            logger.error(f"多模态分析时出错: {str(e)}")
            # 如果多模态分析失败，回退到普通文本分析
            logger.info("回退到普通文本分析")
            return self.analyze_paper_section(paper_text, section, instruction)
            
    def analyze_formula_in_two_parts(self, paper_text: str, section: str, instruction: str) -> str:
        """将公式解读任务分为两部分进行，以避免因公式占用过多token而导致输出不完整
        
        Args:
            paper_text: 论文文本内容
            section: 要分析的部分名称
            instruction: 分析指导说明
            
        Returns:
            两部分合并后的分析结果
        """
        try:
            # 使用完整的论文文本，不应用max_tokens限制
            logger.info(f"执行公式推导任务第一部分：公式背景、意义和符号解释")
            truncated_text = paper_text
            
            # 处理文本中的公式标记
            has_formulas = False
            formula_instruction = ""
            if self.enable_formula_extraction and self.formula_extractor and re.search(r"\{v\d+\}", truncated_text):
                has_formulas = True
                # 将公式标记替换为LaTeX格式
                truncated_text = self.formula_extractor.replace_formula_markers(truncated_text)
                formula_instruction = "\n注意：文本中包含数学公式，以$$公式$$格式标记。请在分析中特别关注这些公式的含义和作用。"
                logger.info(f"文本中包含数学公式，已替换为LaTeX格式")
            
            # 第一部分：公式背景、意义和符号解释
            system_prompt_part1 = FOMULATION_ANALYSIS_PART1_PROMPT.substitute()
            if has_formulas:
                system_prompt_part1 += formula_instruction
                
            completion_part1 = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": system_prompt_part1},
                    {"role": "user", "content": f"以下是论文内容：\n\n{truncated_text}"}
                ]
            )
            
            # 获取第一部分的内容
            response_content_part1 = completion_part1.choices[0].message.content
            logger.info(f"完成公式推导任务第一部分")
            
            # 第二部分：公式推导过程和应用实例
            logger.info(f"执行公式推导任务第二部分：公式推导过程和应用实例")
            system_prompt_part2 = FOMULATION_ANALYSIS_PART2_PROMPT.substitute()
            if has_formulas:
                system_prompt_part2 += formula_instruction
                
            completion_part2 = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": system_prompt_part2},
                    {"role": "user", "content": f"以下是论文内容：\n\n{truncated_text}"}
                ]
            )
            
            # 获取第二部分的内容
            response_content_part2 = completion_part2.choices[0].message.content
            logger.info(f"完成公式推导任务第二部分")
            
            # 合并两部分内容
            combined_content = f"## 公式解读 - 第一部分：公式背景、意义和符号解释\n\n{response_content_part1}\n\n## 公式解读 - 第二部分：公式推导过程和应用实例\n\n{response_content_part2}"
            
            # 处理合并内容中的公式
            try:
                from markdown_generator import format_latex_equations
                # 使用format_latex_equations函数处理公式
                combined_content = format_latex_equations(combined_content)
                logger.info(f"已处理合并内容中的公式")
            except Exception as format_error:
                logger.warning(f"处理合并内容中的公式时出错: {str(format_error)}")
            
            return combined_content
            
        except Exception as e:
            logger.error(f"分两部分分析公式时出错: {str(e)}")
            raise
    
    def determine_relevant_sections(self, sections: Dict[str, str], section_name: str, instruction: str) -> tuple[str, list]:
        """根据分析任务确定需要使用的相关章节
        
        Args:
            sections: 章节字典
            section_name: 要分析的部分名称
            instruction: 分析指导说明
            
        Returns:
            相关章节的文本内容，如果启用了公式提取功能，则包含公式标记
        """
        # 根据不同的分析任务，确定需要使用的章节
        if section_name == "基本信息":
            # 基本信息通常在标题、摘要和引言部分
            relevant_title = []
            relevant_text = ""
            for key in ["标题", "摘要", "Abstract", "Introduction", "引言", "1."]:
                if key in sections:
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "摘要解读":
            # 摘要解读主要使用摘要部分
            for key in ["摘要", "Abstract"]:
                if key in sections:
                    return sections[key], [key]
            
        elif section_name == "研究背景" or section_name == "相关工作":
            # 研究背景和相关工作通常在引言和相关工作章节
            relevant_title = []
            relevant_text = ""
            for key in sections.keys():
                if any(term in key for term in ["Introduction", "Related Work", "Background", "引言", "相关工作", "研究背景", "1.", "2."]):
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "研究方法":
            # 研究方法通常在方法章节
            relevant_title = []
            relevant_text = ""
            for key in sections.keys():
                if any(term in key for term in ["Method", "Methodology", "Approach", "方法", "3.", "4."]):
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "创新点":
            # 创新点可能分散在多个章节，需要综合分析
            # 使用摘要、引言和方法部分
            relevant_title = []
            relevant_text = ""
            for key in sections.keys():
                if any(term in key for term in ["Abstract", "Introduction", "Method", "Contribution", "摘要", "引言", "方法", "贡献"]):
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "实验结果":
            # 实验结果通常在结果和实验章节
            relevant_title = []
            relevant_text = ""
            for key in sections.keys():
                if any(term in key for term in ["Experiment", "Result", "Evaluation", "实验", "结果", "评估", "5.", "6."]):
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "结论" or section_name == "局限性" or section_name == "综合评价":
            # 结论、局限性和综合评价通常在结论和讨论章节
            relevant_title = []
            relevant_text = ""
            for key in sections.keys():
                if any(term in key for term in ["Conclusion", "Discussion", "Future Work", "Limitation", "结论", "讨论", "未来工作", "局限性", "7.", "8."]):
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
            # 对于综合评价，还需要添加摘要部分
            if section_name == "综合评价" and "摘要" in sections:
                relevant_title.append("摘要")
                relevant_text = sections["摘要"] + "\n\n" + relevant_text
            return relevant_text[:self.max_tokens], relevant_title  # 限制长度
            
        elif section_name == "再现公式推导" or section_name == "公式推导再现与解读" or section_name == "公式解读":
            # 公式推导通常在方法章节和理论章节
            relevant_title = []
            relevant_text = ""
            
            # 首先优先选择包含方法和理论的章节
            method_sections = []
            for key in sections.keys():
                if any(term in key for term in ["Method", "Methodology", "Approach", "Theory", "Theorem", "Proof", "方法", "理论", "证明", "3.", "4."]):
                    method_sections.append(key)
                    
            # 如果启用了公式提取功能，检查这些章节中是否包含公式标记
            if self.enable_formula_extraction and self.formula_extractor:
                formula_sections = []
                for key in method_sections:
                    if re.search(r"\{v\d+\}", sections[key]):
                        formula_sections.append(key)
                
                # 如果找到包含公式的章节，优先使用这些章节
                if formula_sections:
                    logger.info(f"找到包含公式的章节: {formula_sections}")
                    for key in formula_sections:
                        relevant_title.append(key)
                        relevant_text += sections[key] + "\n\n"
                    # 对于公式推导任务，不限制文本长度
                    return relevant_text, relevant_title  # 不限制长度
            
            # 如果没有找到包含公式的章节或未启用公式提取，使用方法章节
            if method_sections:
                for key in method_sections:
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
                # 对于公式推导任务，不限制文本长度
                return relevant_text, relevant_title  # 不限制长度
                
            # 如果没有找到方法章节，尝试使用包含数学符号的章节
            math_related_sections = []
            for key in sections.keys():
                if any(term in key.lower() for term in ["math", "equation", "formula", "数学", "公式", "推导"]):
                    math_related_sections.append(key)
                    
            if math_related_sections:
                for key in math_related_sections:
                    relevant_title.append(key)
                    relevant_text += sections[key] + "\n\n"
                # 对于公式推导任务，不限制文本长度
                return relevant_text, relevant_title  # 不限制长度
        
        # 如果无法确定相关章节，使用LLM来判断
        try:
            # 使用简单提示词让LLM判断需要哪些章节
            logger.info(f"使用LLM判断相关章节")
            section_titles = list(sections.keys())
            prompt = f"""请根据以下任务，从给定的论文章节列表中选择最相关的章节（可以选择多个，不要选择Refference）：
            任务：{section_name} - {instruction}
            
            论文章节列表：
            {', '.join(section_titles)}
            
            请直接列出相关章节的名称，用逗号分隔。"""
            
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "你是一个帮助分析学术论文的助手。请根据任务选择最相关的论文章节。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = completion.choices[0].message.content
            selected_sections = [s.strip() for s in response.split(',')]
            
            # 获取选定章节的内容
            relevant_title = []
            relevant_text = ""
            for section_title in selected_sections:
                if section_title in sections:
                    relevant_title.append(section_title)
                    relevant_text += sections[section_title] + "\n\n"
                else:
                    # 尝试模糊匹配
                    for key in sections.keys():
                        if section_title.lower() in key.lower():
                            relevant_title.append(key)
                            relevant_text += sections[key] + "\n\n"
                            break
            
            if relevant_text:
                return relevant_text[:self.max_tokens], relevant_title  # 限制长度
        except Exception as e:
            logger.warning(f"使用LLM判断相关章节时出错: {str(e)}")
        
        # 如果上述方法都失败，返回整个论文的前self.max_tokens个字符
        all_text = "\n\n".join(sections.values())
        return all_text[:self.max_tokens], ["整个论文的前self.max_tokens个字符"]

    def analyze_full_paper(self, pdf_path: str, output_dir: Optional[str] = None, use_comprehensive_prompt: bool = False, use_sections: bool = True, enable_multimodal: bool = True) -> Tuple[Dict[str, Any], Optional[str]]:
        """对整篇论文进行全面分析
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 分析结果输出路径，默认为None（不保存）
            use_comprehensive_prompt: 是否使用综合提示词一次性分析论文，默认为False
            use_sections: 是否使用章节分析，默认为True
            enable_multimodal: 是否启用多模态分析，默认为True
            
        Returns:
            完整的论文分析结果
        """
        # 提取论文章节
        logger.info(f"开始提取论文章节: {pdf_path}")
        paper_sections = self.section_extractor.extract_sections_from_pdf(pdf_path)
        
        # 如果启用了图像提取功能，提取论文中的图像
        images_dir = None
        if self.enable_image_extraction and enable_multimodal and output_dir:
            images_dir = os.path.join(output_dir, "images")
            logger.info(f"开始提取论文图像: {pdf_path}")
            self.extract_images_from_pdf(pdf_path, images_dir)
        
        # 如果没有成功提取章节或者不使用章节分析，则使用传统方法提取整个文本
        if not paper_sections or len(paper_sections) <= 1 or not use_sections:
            if not use_sections:
                logger.info("未启用章节分析功能，使用传统方法提取整个文本")
            else:
                logger.warning("未能成功提取论文章节，使用传统方法提取整个文本")
            paper_text = self.extract_text_from_pdf(pdf_path)
        else:
            logger.info(f"成功提取论文章节，共{len(paper_sections)}个章节")
            # 将所有章节内容合并为一个文本，用于综合分析或生成分析计划
            paper_text = "\n\n".join(paper_sections.values())
            
            # 如果启用了公式提取功能，检查章节提取器中是否有公式
            if self.enable_formula_extraction and hasattr(self.section_extractor, 'formulas') and self.section_extractor.formulas:
                logger.info(f"章节提取过程中识别到{len(self.section_extractor.formulas)}个公式")
                # 将章节提取器中的公式列表复制到公式提取器中，以便后续处理
                if self.formula_extractor:
                    self.formula_extractor.formulas = self.section_extractor.formulas.copy()
        
        # 初始化输出路径
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plan_path = os.path.join(output_dir, "analysis_plan.json")
            json_path = os.path.join(output_dir, "analysis_result.json")
            markdown_path = os.path.join(output_dir, "analysis_result.md")
            sections_path = os.path.join(output_dir, "paper_sections.json")
            
            # 如果启用了图像提取功能，确保图像目录存在
            if self.enable_image_extraction and enable_multimodal:
                if not images_dir:
                    images_dir = os.path.join(output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
            
            # 保存提取的章节
            if paper_sections and len(paper_sections) > 1:
                with open(sections_path, 'w', encoding='utf-8') as f:
                    # 只保存章节标题，不保存内容（内容可能很大）
                    section_titles = {k: f"[章节内容长度: {len(v)}字符。{v}]" for k, v in paper_sections.items()}
                    json.dump(section_titles, f, ensure_ascii=False, indent=2)
                logger.info(f"章节信息已保存至: {sections_path}")
        
        # 初始化结果字典
        result = {
            "paper_path": pdf_path,
            "analysis_results": {},
            "extracted_images": self.extracted_images,  # 添加提取的图像路径列表
            "image_captions": self.image_extractor.image_captions if self.image_extractor else {}  # 添加图像标题信息
        }
        
        # 如果使用综合提示词，则一次性分析整篇论文
        if use_comprehensive_prompt:
            logger.info("使用综合提示词一次性分析论文")
            try:
                # 截取论文前self.max_tokens个字符作为输入，避免超出token限制
                truncated_text = paper_text[:self.max_tokens] + "..." if len(paper_text) > self.max_tokens else paper_text
                
                # 处理文本中的公式标记
                formula_instruction = ""
                if self.enable_formula_extraction and self.formula_extractor and re.search(r"\{v\d+\}", truncated_text):
                    # 将公式标记替换为LaTeX格式
                    truncated_text = self.formula_extractor.replace_formula_markers(truncated_text)
                    formula_instruction = "\n注意：文本中包含数学公式，以$$公式$$格式标记。请在分析中特别关注这些公式的含义和作用。"
                    logger.info(f"文本中包含数学公式，已替换为LaTeX格式")
                
                system_prompt = COMPREHENSIVE_ANALYSIS_PROMPT.substitute()
                if formula_instruction:
                    system_prompt += formula_instruction
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.5,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"以下是论文内容：\n\n{truncated_text}"}
                    ]
                )
                
                # 获取大模型返回的内容
                response_content = completion.choices[0].message.content
                
                # 处理大模型返回内容中的公式
                try:
                    from markdown_generator import format_latex_equations
                    # 使用format_latex_equations函数处理公式
                    response_content = format_latex_equations(response_content)
                    logger.info(f"已处理大模型返回内容中的公式")
                except Exception as format_error:
                    logger.warning(f"处理大模型返回内容中的公式时出错: {str(format_error)}")
                
                result["comprehensive_analysis"] = response_content
                
                # 保存分析结果
                if json_path:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"分析结果已保存至: {json_path}")

                # 生成Markdown报告
                markdown_content = None
                markdown_content = generate_markdown_report(result)
                if markdown_path:
                    with open(markdown_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    logger.info(f"Markdown报告已保存至: {markdown_path}")
                
                return result, markdown_content
                
            except Exception as e:
                logger.error(f"使用综合提示词分析论文时出错: {str(e)}")
                logger.info("切换到分步分析方法")
        
        # 生成分析计划
        analysis_plan = self.generate_analysis_plan(paper_text)  
        result["analysis_plan"] = analysis_plan
        # 将解读计划存储为txt文件
        if plan_path:
            with open(plan_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_plan, f, ensure_ascii=False, indent=2)
            logger.info(f"解读计划已保存至: {plan_path}")
        
        # 如果分析计划是原始文本格式，则直接进行基本分析
        if "raw_plan" in analysis_plan:
            logger.info("使用默认分析步骤")
            sections = [
                {"name": "基本信息", "instruction": "提取论文的标题、作者、发表年份、期刊/会议等基本信息。"},
                {"name": "摘要解读", "instruction": "解读论文摘要，扩展其中的关键信息，解释专业术语。"},
                {"name": "研究背景", "instruction": "分析研究背景与意义，解释为什么这项研究很重要。"},
                {"name": "研究方法", "instruction": "详细解释研究方法，包括技术路线、算法、模型等。"},
                {"name": "创新点", "instruction": "识别并分析论文的关键创新点，解释其重要性。"},
                {"name": "实验结果", "instruction": "分析实验设计与结果，解释数据和图表的含义。"},
                {"name": "结论", "instruction": "总结论文的主要结论与贡献。"},
                {"name": "局限性", "instruction": "分析研究的局限性和未来研究方向。"},
                {"name": "综合评价", "instruction": "对论文进行综合评价，包括学术贡献和潜在影响。"},
                {"name": "再现公式推导", "instruction": "如果论文中涉及公式推导，请将其重新推导并解释。"}
            ]
        else:
            # 根据分析计划构建分析步骤
            sections = []
            for key, value in analysis_plan.items():
                sections.append({"name": key, "instruction": value})
        
        # 执行每个部分的分析
        print("\n================= 深度分析 =================\n")
        for section in sections:
            print(f"\n================= {section['name']} =================\n")
            # logger.info(f"分析论文{section['name']}部分")
            
            # 如果成功提取了章节，则根据分析任务确定需要使用的章节
            if paper_sections and len(paper_sections) > 1:
                # 根据分析任务确定需要使用的相关章节
                relevant_text, relevant_title = self.determine_relevant_sections(paper_sections, section['name'], section['instruction'])
                logger.info(f"为{section['name']}分析任务选择了相关章节：{relevant_title}，文本长度: {len(relevant_text)}字符")
            else:
                # 如果没有成功提取章节，则使用整个文本
                relevant_text = paper_text
            
            # 检查相关章节中是否包含公式标记
            if self.enable_formula_extraction and re.search(r"\{v\d+\}", relevant_text):
                logger.info(f"在{section['name']}相关章节中发现公式标记")
                # 如果章节提取器有公式处理功能，使用章节提取器替换公式标记
                if hasattr(self.section_extractor, 'replace_formula_markers') and self.section_extractor.formulas:
                    logger.info(f"使用章节提取器替换公式标记")
                    relevant_text = self.section_extractor.replace_formula_markers(relevant_text)
                # 如果章节提取器没有公式处理功能或没有提取到公式，但公式提取器可用，则使用公式提取器
                elif self.formula_extractor:
                    logger.info(f"使用公式提取器替换公式标记")
                    relevant_text = self.formula_extractor.replace_formula_markers(relevant_text)
            
            # 确定是否需要使用多模态分析
            related_images = []
            # 对于公式推导任务，不使用图片分析，避免token过多
            if self.enable_image_extraction and enable_multimodal and self.extracted_images and \
               (section["name"] == "实验结果" or "图" in section["name"] or "表" in section["name"]) and \
               not (section["name"] == "再现公式推导" or "公式" in section["name"] or "推导" in section["name"]):
                # 对于实验结果部分（非公式推导任务），使用所有提取的图像
                related_images = self.extracted_images
                logger.info(f"为{section['name']}分析任务准备了{len(related_images)}张相关图像")
                
                # 将当前章节与相关图像的关联信息保存到结果字典中
                if "section_images" not in result:
                    result["section_images"] = {}
                result["section_images"][section["name"]] = related_images
            
            # 分析相关章节，如果有相关图像则使用多模态分析
            analysis = self.analyze_paper_section(relevant_text, section["name"], section["instruction"], related_images)
            result["analysis_results"][section["name"]] = analysis
        
        # 生成Markdown报告
        markdown_content = None
        markdown_content = generate_markdown_report(result)
        if markdown_path:
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Markdown报告已保存至: {markdown_path}")
        
        # 保存JSON分析结果
        if json_path:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存至: {json_path}")
        
        return result, markdown_content


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用智谱AI进行论文深度解读")
    parser.add_argument("--pdf", type=str, required=True, help="PDF论文文件路径")
    parser.add_argument("--api_key", type=str, required=True, help="智谱AI的API密钥")
    parser.add_argument("--model", type=str, default="glm-4-flash", help="使用的模型名称，默认为glm-4-flash")
    parser.add_argument("--multimodal_model", type=str, default="glm-4v-flash", help="多模态模型名称，默认为glm-4v-flash")
    parser.add_argument("--output", type=str, default="../Result", help="分析结果输出路径")
    parser.add_argument("--disable_formula", action="store_true", help="禁用公式提取功能")
    parser.add_argument("--disable_image", action="store_true", help="禁用图像提取功能")
    parser.add_argument("--disable_multimodal", action="store_true", help="禁用多模态分析功能")
    
    args = parser.parse_args()
    
    analyzer = PaperAnalyzer(
        api_key=args.api_key, 
        model=args.model,
        multimodal_model=args.multimodal_model,
        enable_formula_extraction=not args.disable_formula,
        enable_image_extraction=not args.disable_image
    )
    analyzer.analyze_full_paper(
        args.pdf, 
        args.output,
        enable_multimodal=not args.disable_multimodal
    )
    print(f"论文分析完成，结果已保存至: {args.output}")


if __name__ == "__main__":
    main()