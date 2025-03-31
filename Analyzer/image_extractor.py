import os
import logging
import base64
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageExtractor:
    """用于从PDF中提取图像的工具类"""
    
    def __init__(self, output_dir: str = None):
        """初始化图像提取器
        
        Args:
            output_dir: 图像输出目录，默认为None（使用相对路径'./image'）
        """
        self.output_dir = output_dir if output_dir else './image'
        self.images = []  # 存储提取的图像路径
        self.image_captions = {}  # 存储图像标题，键为图像路径，值为标题
        
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """从PDF中提取图像并保存到指定目录
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 图像输出目录，如果提供则覆盖初始化时的设置
            
        Returns:
            提取的图像文件路径列表
        """
        if not pdf_path:
            raise ValueError("PDF文件路径不能为空")
            
        # 如果提供了输出目录，则覆盖初始化时的设置
        if output_dir:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 重置图像列表
        self.images = []
        self.image_captions = {}
        
        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            # 获取页数并记录日志
            page_count = len(doc)
            logger.info(f"PDF文件{pdf_path}共有{page_count}页")
            
            # 逐页提取图像
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    
                    # 提取页面上的图像
                    image_list = page.get_images(full=True)
                    
                    # 如果页面上有图像
                    if image_list:
                        logger.info(f"第{page_num+1}页发现{len(image_list)}个图像")
                        
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]  # 图像的xref
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                
                                # 生成图像文件名
                                image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                                image_path = os.path.join(self.output_dir, image_filename)
                                
                                # 保存图像
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                
                                # 添加到图像列表
                                self.images.append(image_path)
                                
                                # 尝试提取图像标题（通常在图像下方的文本块中）
                                self._extract_image_caption(page, img, image_path)
                                
                                logger.info(f"保存图像: {image_path}")
                            except Exception as img_error:
                                logger.warning(f"提取第{page_num+1}页第{img_index+1}个图像时出错: {str(img_error)}")
                except Exception as page_error:
                    logger.warning(f"处理第{page_num+1}页时出错: {str(page_error)}")
            
            doc.close()
            logger.info(f"从{pdf_path}提取了{len(self.images)}个图像")
            return self.images
            
        except Exception as e:
            logger.error(f"提取PDF图像时出错: {str(e)}")
            raise
    
    def _extract_image_caption(self, page, img, image_path: str) -> None:
        """尝试提取图像标题
        
        Args:
            page: PDF页面对象
            img: 图像信息
            image_path: 图像文件路径
        """
        try:
            # 获取图像在页面上的位置
            # 注意：这是一个简化的实现，实际上需要更复杂的逻辑来准确匹配图像和标题
            blocks = page.get_text("dict")["blocks"]
            
            # 查找可能的图像标题（通常以"Figure"或"图"开头的文本块）
            for block in blocks:
                if block["type"] == 0:  # 文本块
                    text = "".join([span["text"] for line in block["lines"] for span in line["spans"]])
                    if text.strip().startswith(("Figure", "Fig.", "图")):
                        self.image_captions[image_path] = text.strip()
                        logger.info(f"提取图像标题: {text.strip()}")
                        break
        except Exception as e:
            logger.warning(f"提取图像标题时出错: {str(e)}")
    
    def get_image_base64(self, image_path: str) -> Optional[str]:
        """获取图像的base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像的base64编码，如果文件不存在则返回None
        """
        if not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            return None
            
        try:
            with open(image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                return img_base64
        except Exception as e:
            logger.error(f"获取图像base64编码时出错: {str(e)}")
            return None
    
    def get_image_caption(self, image_path: str) -> Optional[str]:
        """获取图像标题
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像标题，如果没有提取到则返回None
        """
        return self.image_captions.get(image_path)