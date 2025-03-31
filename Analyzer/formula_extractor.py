import re
import unicodedata
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FormulaExtractor:
    """用于从PDF中提取文本并保留数学公式的工具类"""
    
    def __init__(self, vfont: str = None, vchar: str = None):
        """初始化公式提取器
        
        Args:
            vfont: 公式字体的正则表达式模式，默认为None（使用内置规则）
            vchar: 公式字符的正则表达式模式，默认为None（使用内置规则）
        """
        self.vfont = vfont
        self.vchar = vchar
        self.formulas = []  # 存储提取的公式
        
    def vflag(self, font: str, char: str) -> bool:
        """判断字符是否属于公式
        
        Args:
            font: 字体名称
            char: 字符
            
        Returns:
            是否属于公式
        """
        if isinstance(font, bytes):
            try:
                font = font.decode('utf-8')
            except UnicodeDecodeError:
                font = ""
        font = font.split("+")[-1]  # 字体名截断
        
        # 基于字体名规则的判定
        if self.vfont:
            if re.match(self.vfont, font):
                return True
        else:
            if re.match(
                # latex 字体
                r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                font,
            ):
                return True
        
        # 基于字符集规则的判定
        if self.vchar:
            if re.match(self.vchar, char):
                return True
        else:
            if (
                char
                and char != " "  # 非空格
                and (
                    unicodedata.category(char[0])
                    in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]  # 文字修饰符、数学符号、分隔符号
                    or ord(char[0]) in range(0x370, 0x400)  # 希腊字母
                )
            ):
                return True
        return False
    
    def extract_text_with_formulas(self, pdf_path: str) -> str:
        """从PDF中提取文本，保留数学公式
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            带有公式标记的文本
        """
        if not pdf_path:
            raise ValueError("PDF文件路径不能为空")
            
        try:
            doc = fitz.open(pdf_path)
            result_text = ""
            self.formulas = []  # 重置公式列表
            formula_index = 0
            
            # 逐页处理
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 获取页面上的文本块
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                
                # 处理每个文本块
                for block in blocks:
                    if block["type"] == 0:  # 文本块
                        block_text = ""
                        in_formula = False
                        formula_text = ""
                        
                        # 处理每个文本行
                        for line in block["lines"]:
                            line_text = ""
                            
                            # 处理每个跨度（span）
                            for span in line["spans"]:
                                font_name = span["font"]
                                text = span["text"]
                                size = span["size"]
                                flags = span["flags"]  # 字体标志（粗体、斜体等）
                                
                                # 检查是否是公式
                                is_formula = False
                                
                                # 检查字体名称是否符合公式字体特征
                                if self.vflag(font_name, text):
                                    is_formula = True
                                
                                # 检查是否是斜体（通常用于数学符号）
                                if flags & 2:  # 斜体标志
                                    is_formula = True
                                    
                                # 检查文本中是否包含数学符号
                                for char in text:
                                    if self.vflag("", char):
                                        is_formula = True
                                        break
                                
                                if is_formula:
                                    # 如果之前不在公式中，开始新公式
                                    if not in_formula:
                                        in_formula = True
                                        formula_text = text
                                    else:
                                        # 继续当前公式
                                        formula_text += text
                                else:
                                    # 如果之前在公式中，结束公式并添加标记
                                    if in_formula:
                                        self.formulas.append(formula_text)
                                        line_text += f"{{v{formula_index}}}"
                                        formula_index += 1
                                        in_formula = False
                                        formula_text = ""
                                    
                                    # 添加普通文本
                                    line_text += text
                            
                            # 如果行结束时仍在公式中，结束公式
                            if in_formula:
                                self.formulas.append(formula_text)
                                line_text += f"{{v{formula_index}}}"
                                formula_index += 1
                                in_formula = False
                                formula_text = ""
                                
                            block_text += line_text + "\n"
                        
                        page_text += block_text
                
                result_text += page_text + "\n\n"
            
            doc.close()
            return result_text
            
        except Exception as e:
            logger.error(f"提取带公式的文本时出错: {str(e)}")
            raise
    
    def get_formula(self, index: int) -> Optional[str]:
        """获取指定索引的公式
        
        Args:
            index: 公式索引
            
        Returns:
            公式文本，如果索引无效则返回None
        """
        if 0 <= index < len(self.formulas):
            return self.formulas[index]
        return None
    
    def replace_formula_markers(self, text: str) -> str:
        """将文本中的公式标记替换为实际公式
        
        Args:
            text: 带有公式标记的文本
            
        Returns:
            替换后的文本
        """
        def replace_match(match):
            formula_index = int(match.group(1))
            formula = self.get_formula(formula_index)
            if formula:
                return f"$${formula}$$"
            return match.group(0)
        
        # 替换所有{vN}格式的公式标记
        return re.sub(r"\{v(\d+)\}", replace_match, text)