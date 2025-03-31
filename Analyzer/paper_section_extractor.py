#!/usr/bin/env python3
"""
改进后的论文章节提取模块

本模块用于从PDF文件中提取文本并识别章节结构，
支持跨行章节标题格式（例如：
    3
    MapCoder
将自动合并为“3 MapCoder”），
同时放宽候选标题的长度限制，提升拆分细致程度，
保证拆分顺序与原文一致，对于无编号的标题自动添加“第X章”前缀。
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
import unicodedata

import json
import os
from formula_extractor import FormulaExtractor  # 导入公式提取器

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperSectionExtractor:
    """改进后的论文章节提取器，支持公式识别和处理"""

    def __init__(self, enable_formula_extraction: bool = True):
        # 各种常见的章节标题正则表达式
        self.section_patterns = [
            r'^\s*第\s*\d+\s*章\s+.*$',                      # 例如：第1章 引言
            r'^\s*Chapter\s+\d+\s*[:.-]\s+.*$',               # 例如：Chapter 1: Introduction
            r'^\s*\d+\.\s+([A-Z][\w\s]{3,30})\s*$',           # 例如：1. Introduction
            r'^\s*\d+\s+[A-Za-z][\w\s]{2,50}\s*$',            # 例如：3 MapCoder（数字后无标点）
            r'^\s*[IVXivx]+\.\s+([A-Z][\w\s]{3,30})\s*$',      # 例如：I. Introduction
            r'^\s*([A-Z][A-Z\s]{2,30})\s*$',                  # 例如：INTRODUCTION（全部大写）
            r'^\s*(Introduction|Abstract|Related\s+Work|Background|Methodology|Method|Experiments?|Results|Discussion|Conclusion|References|Appendix)\s*$',
            r'^\s*\d+\.\s+([\u4e00-\u9fa5]{2,20})\s*$',       # 例如：1. 引言
            r'^\s*([\u4e00-\u9fa5]{2,15})\s*$'                # 例如：引言
        ]
        self.min_section_length = 100  # 章节内容最小长度阈值

        self.non_section_keywords = [
            'et al', 'figure', 'table', 'algorithm', 'et.', 'al.',
            'i.e.', 'e.g.', 'vs.', 'etc.', 'fig.', 'tab.', 'eq.',
            'pp.', 'vol.', 'no.', 'p.', 'cf.', 'viz.', 'resp.'
        ]
        
        # 公式提取相关
        self.enable_formula_extraction = enable_formula_extraction
        self.formula_extractor = FormulaExtractor() if enable_formula_extraction else None
        self.formulas = []  # 存储提取的公式

    def extract_sections_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        从PDF文件中提取文本并识别章节结构。
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            章节字典，键为章节标题（统一加上自动编号），值为对应章节内容
        """
        if not pdf_path:
            raise ValueError("PDF文件路径不能为空")
        
        text, _ = self._extract_text_with_pages(pdf_path)
        sections = self._identify_sections(text)
        
        if not sections or len(sections) <= 1:
            logger.warning("未能识别到明确的章节结构，将整个文本作为一个章节")
            title = self._extract_title(text[:1000])
            return {"标题": title or "未识别标题", "全文": text.strip()}
        
        return sections

    def _extract_text_with_pages(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        从PDF中提取文本，同时返回每页的文本列表。
        如果启用了公式提取功能，则会保留数学公式。
        """
        # 如果启用了公式提取功能，使用FormulaExtractor提取带公式的文本
        if self.enable_formula_extraction and self.formula_extractor:
            try:
                logger.info(f"使用公式提取功能从{pdf_path}提取文本")
                text = self.formula_extractor.extract_text_with_formulas(pdf_path)
                # 保存提取的公式列表，以便后续使用
                self.formulas = self.formula_extractor.formulas
                logger.info(f"成功从{pdf_path}提取带公式的文本，共识别{len(self.formulas)}个公式")
                # 由于使用了公式提取器，我们需要模拟page_texts列表
                page_texts = [text]  # 简化处理，将整个文本作为一页
                return text, page_texts
            except Exception as formula_error:
                logger.warning(f"使用公式提取功能失败: {str(formula_error)}，将使用普通文本提取方法")
                # 如果公式提取失败，回退到普通文本提取方法
        
        # 使用普通方法提取文本
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            page_texts = []
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    page_texts.append(page_text)
                    full_text.append(page_text)
                except Exception as e:
                    logger.warning(f"提取第{page_num+1}页时出错: {e}")
                    page_texts.append("")
            doc.close()
            combined_text = "\n".join(full_text)
            logger.info(f"成功从 {pdf_path} 提取文本")
            return combined_text, page_texts
        except Exception as e:
            logger.error(f"提取PDF文本时出错: {e}")
            raise

    def _extract_title(self, text: str) -> Optional[str]:
        """
        尝试从文本中提取论文标题（检查前几行并过滤页眉等信息）。
        """
        lines = text.splitlines()
        for line in lines[:10]:
            line = line.strip()
            if 10 < len(line) < 200:
                if line.isupper() or (line[0].isupper() and not line.isupper()):
                    if not any(x in line.lower() for x in ['page', 'copyright', 'journal', 'conference', 'proceedings']):
                        return line
        return None

    def _extract_abstract_with_index(self, text: str) -> Tuple[Optional[str], int, int]:
        """
        尝试提取摘要部分，同时返回摘要在全文中的起止位置。
        """
        abstract_patterns = [
            r'(?i)\bAbstract\b[\s\n]+([\s\S]+?)(?=\n\s*\d+\.\s+|\n\s*[IVX]+\.\s+|\n\s*Introduction|\n\s*INTRODUCTION)',
            r'(?i)\b摘要\b[\s\n]+([\s\S]+?)(?=\n\s*\d+\.\s+|\n\s*[一二三四五六七八九十]+[、\.]\s+|\n\s*引言|\n\s*绪论)'
        ]
        for pattern in abstract_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(), match.start(), match.end()
        return None, -1, -1

    def _extract_abstract(self, text: str) -> Optional[str]:
        """
        简化版摘要提取，返回摘要文本。
        """
        abstract, _, _ = self._extract_abstract_with_index(text)
        return abstract

    def _clean_section_title(self, title: str) -> str:
        """
        清洗候选章节标题，去掉前导的数字、中文“第X章”、英文“Chapter X:”等部分。
        """
        # 去掉中文数字章节标记，例如“第3章 ”或“第3章:”
        title = re.sub(r'^(第\s*\d+\s*章\s*[:：-]*\s*)', '', title)
        # 去掉英文数字章节标记，例如“Chapter 3: ”或“3. ”或“3 ”
        title = re.sub(r'^(Chapter\s+\d+\s*[:.-]*\s*|\d+\.\s*|\d+\s+)', '', title)
        return title.strip()

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """
        识别文本中的章节结构，支持跨行章节标题（如“3”换行“MapCoder”），
        并根据候选行的位置拆分章节内容。
        
        Args:
            text: 整个文档的文本
            
        Returns:
            章节字典，键为统一添加编号后的章节标题，值为章节内容
        """
        lines = text.splitlines()
        candidate_list = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            candidate_text = None

            # 若当前行仅包含数字，则尝试合并下一行（若存在且非空）
            if re.match(r'^\d+\s*$', stripped):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    candidate_text = stripped + " " + lines[j].strip()
                    candidate_index = i
                    i = j + 1  # 跳过合并过的行
                else:
                    candidate_text = stripped
                    candidate_index = i
                    i += 1
            else:
                candidate_text = stripped
                candidate_index = i
                i += 1

            if not candidate_text:
                continue
            # 长度要求：3～200字符
            if len(candidate_text) < 3 or len(candidate_text) > 200:
                continue
            # 如果全大写且过短则跳过
            if candidate_text.isupper() and len(candidate_text) < 5:
                continue
            if any(keyword.lower() in candidate_text.lower() for keyword in self.non_section_keywords):
                continue

            # 检查是否匹配任一章节标题模式
            for pattern in self.section_patterns:
                if re.match(pattern, candidate_text):
                    candidate_list.append((candidate_index, candidate_text))
                    break

        if not candidate_list:
            return {}

        # 提取论文标题和摘要（若存在）
        sections = {}
        title_extracted = self._extract_title(text[:1000])
        if title_extracted:
            sections["标题"] = title_extracted
        abstract_extracted, abs_start, abs_end = self._extract_abstract_with_index(text)
        if abstract_extracted:
            sections["摘要"] = abstract_extracted

        # 过滤掉候选中与摘要重复的（如“Abstract”或“摘要”）
        filtered_candidates = []
        for idx, cand in candidate_list:
            if abstract_extracted and cand.lower() in ["abstract", "摘要"]:
                continue
            filtered_candidates.append((idx, cand))
        filtered_candidates.sort(key=lambda x: x[0])

        # 根据候选行在全文中的位置拆分章节内容
        section_boundaries = []
        for j, (start_idx, cand_text) in enumerate(filtered_candidates):
            end_idx = filtered_candidates[j+1][0] if j+1 < len(filtered_candidates) else len(lines)
            content = "\n".join(lines[start_idx+1:end_idx]).strip()

            if len(content) < self.min_section_length:
                continue
            section_boundaries.append((start_idx, cand_text, content))

        if not section_boundaries:
            return sections

        # 重新统一自动编号（忽略原有编号），并清洗标题
        new_sections = {}
        for chapter_idx, (_, cand_text, content) in enumerate(section_boundaries, start=1):
            clean_title = self._clean_section_title(cand_text)
            new_key = f"第{chapter_idx}章 {clean_title}"
            new_sections[new_key] = content

        # 保持“标题”和“摘要”在最前面
        final_sections = {}
        if "标题" in sections:
            final_sections["标题"] = sections["标题"]
        if "摘要" in sections:
            final_sections["摘要"] = sections["摘要"]
        final_sections.update(new_sections)

        return final_sections

    def get_section_by_name(self, sections: Dict[str, str], section_name: str) -> Optional[str]:
        """
        根据章节名称获取章节内容，支持模糊匹配。
        """
        if section_name in sections:
            return sections[section_name]
        for title, content in sections.items():
            if section_name.lower() in title.lower():
                return content
        return None

    def get_relevant_sections(self, sections: Dict[str, str], query: str) -> Dict[str, str]:
        """
        根据查询关键词获取相关章节。
        """
        relevant_sections = {}
        query_terms = query.lower().split()
        for title, content in sections.items():
            title_relevance = sum(1 for term in query_terms if term in title.lower())
            content_relevance = sum(1 for term in query_terms if term in content.lower()[:500])
            if title_relevance > 0 or content_relevance > 0:
                relevant_sections[title] = content
        if not relevant_sections:
            if "标题" in sections:
                relevant_sections["标题"] = sections["标题"]
            if "摘要" in sections:
                relevant_sections["摘要"] = sections["摘要"]
        return relevant_sections
        
    def get_formula(self, index: int) -> Optional[str]:
        """
        获取指定索引的公式
        
        Args:
            index: 公式索引
            
        Returns:
            公式文本，如果索引无效则返回None
        """
        if self.enable_formula_extraction and 0 <= index < len(self.formulas):
            return self.formulas[index]
        return None
    
    def replace_formula_markers(self, text: str) -> str:
        """
        将文本中的公式标记替换为LaTeX格式的公式
        
        Args:
            text: 带有公式标记的文本
            
        Returns:
            替换后的文本
        """
        if not self.enable_formula_extraction or not self.formulas:
            return text
            
        def replace_match(match):
            formula_index = int(match.group(1))
            formula = self.get_formula(formula_index)
            if formula:
                return f"$${formula}$$"
            return match.group(0)
        
        # 替换所有形如{vN}的公式标记
        return re.sub(r'\{v(\d+)\}', replace_match, text)


if __name__ == "__main__":
    # 示例：处理指定PDF文件，并将章节信息保存为JSON文件
    project_root = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(project_root, "../Data/example.pdf")  # 请确保example.pdf在同一目录或调整路径
    print(f"正在处理PDF文件: {pdf_path}")

    extractor = PaperSectionExtractor()
    try:
        sections = extractor.extract_sections_from_pdf(pdf_path)
        print(f"\n成功提取到 {len(sections)} 个章节:")
        for i, (title, _) in enumerate(sections.items(), 1):
            print(f"{i}. {title}")

        print("\n各章节内容长度:")
        for title, content in sections.items():
            print(f"{title}: {len(content)} 字符")

        output_dir = os.path.join(project_root, "../Result/test_section_extractor")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "paper_sections.json")
        with open(output_file, "w", encoding="utf-8") as f:
            section_info = {title: f"内容长度: {len(content)} 字符: {content}" for title, content in sections.items()}
            json.dump(section_info, f, ensure_ascii=False, indent=4)
        print(f"\n章节信息已保存到: {output_file}")
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
