import os
import logging
import base64
import fitz  # PyMuPDF
import io
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, some image quality checks will be disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available, some image processing features will be disabled")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageExtractor:
    """用于从PDF中提取图像的工具类"""
    
    def __init__(self, output_dir: str = None, quality_threshold: float = 0.95):
        """初始化图像提取器
        
        Args:
            output_dir: 图像输出目录，默认为None（使用相对路径'./image'）
            quality_threshold: 图像质量阈值，用于检测纯色图像，范围0-1，默认0.95
        """
        self.output_dir = output_dir if output_dir else './image'
        self.images = []  # 存储提取的图像路径
        self.image_captions = {}  # 存储图像标题，键为图像路径，值为标题
        self.quality_threshold = quality_threshold  # 图像质量阈值
        self.image_hashes = set()  # 存储图像哈希值，用于去重
        
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
        
        # 重置图像列表和哈希集合
        self.images = []
        self.image_captions = {}
        self.image_hashes = set()
        
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
                                
                                # 检测图像质量
                                is_low_quality = self._is_low_quality_image(image_bytes)
                                
                                # 如果是低质量图像，尝试使用替代方法提取
                                if is_low_quality:
                                    logger.info(f"检测到第{page_num+1}页第{img_index+1}个图像质量较低，尝试使用替代方法提取")
                                    
                                    # 保存原始图像以便比较
                                    original_image_bytes = image_bytes
                                    original_image_ext = image_ext
                                    
                                    # 存储所有提取结果
                                    extraction_results = []
                                    
                                    # 首先尝试使用截图方法
                                    alt_result = self._extract_image_by_screenshot(page, img)
                                    if alt_result:
                                        extraction_results.append(("截图方法", alt_result))
                                        logger.info(f"使用截图方法成功提取图像")
                                    
                                    # 然后尝试使用页面渲染方法
                                    alt_result = self._extract_image_by_page_render(page, img)
                                    if alt_result:
                                        extraction_results.append(("页面渲染方法", alt_result))
                                        logger.info(f"使用页面渲染方法成功提取图像")
                                    
                                    # 如果有替代提取结果，选择最佳的一个
                                    if extraction_results:
                                        # 评估每个结果的质量
                                        best_method = None
                                        best_quality = -1
                                        
                                        for method_name, (result_bytes, result_ext) in extraction_results:
                                            # 使用PIL评估图像质量
                                            if PIL_AVAILABLE:
                                                try:
                                                    img_pil = Image.open(io.BytesIO(result_bytes))
                                                    img_array = np.array(img_pil.convert('L'))
                                                    # 计算标准差和熵作为质量指标
                                                    std_dev = np.std(img_array)
                                                    hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
                                                    hist_norm = hist / np.sum(hist)
                                                    hist_norm = hist_norm[hist_norm > 0]
                                                    entropy = -np.sum(hist_norm * np.log2(hist_norm)) if len(hist_norm) > 0 else 0
                                                    
                                                    # 综合质量分数
                                                    quality_score = std_dev * 0.7 + entropy * 5.0
                                                    
                                                    logger.info(f"{method_name}提取结果质量评分: {quality_score:.2f} (标准差={std_dev:.2f}, 熵={entropy:.2f})")
                                                    
                                                    if quality_score > best_quality:
                                                        best_quality = quality_score
                                                        best_method = (method_name, (result_bytes, result_ext))
                                                except Exception as e:
                                                    logger.warning(f"评估{method_name}提取结果质量时出错: {str(e)}")
                                            else:
                                                # 如果PIL不可用，简单地选择第一个结果
                                                if best_method is None:
                                                    best_method = (method_name, (result_bytes, result_ext))
                                        
                                        # 使用最佳结果
                                        if best_method:
                                            logger.info(f"选择{best_method[0]}作为最佳提取结果")
                                            image_bytes, image_ext = best_method[1]
                                        else:
                                            logger.warning(f"无法评估替代提取结果质量，将使用原始图像")
                                    else:
                                        logger.warning(f"所有替代提取方法都失败，将使用原始图像")
                                        
                                    # 如果替代方法都失败，检查原始图像是否为纯黑或纯白
                                    if not extraction_results:
                                        # 使用PIL检查原始图像
                                        if PIL_AVAILABLE:
                                            try:
                                                img_pil = Image.open(io.BytesIO(original_image_bytes))
                                                img_array = np.array(img_pil.convert('L'))
                                                # 计算黑白像素比例
                                                black_ratio = np.sum(img_array < 10) / img_array.size
                                                white_ratio = np.sum(img_array > 245) / img_array.size
                                                
                                                # 如果几乎全是黑色或白色，记录警告
                                                if black_ratio > 0.98 or white_ratio > 0.98:
                                                    logger.warning(f"原始图像几乎是纯{'黑' if black_ratio > 0.98 else '白'}色 (比例: {max(black_ratio, white_ratio):.3f})")
                                            except Exception as e:
                                                logger.warning(f"检查原始图像质量时出错: {str(e)}")
                                
                                # 计算图像哈希值用于去重
                                image_hash = self._calculate_image_hash(image_bytes)
                                
                                # 检查是否为重复图像
                                if image_hash in self.image_hashes:
                                    logger.info(f"检测到重复图像，跳过保存")
                                    continue
                                
                                # 添加哈希值到集合中
                                self.image_hashes.add(image_hash)
                                
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
    
    def _is_low_quality_image(self, image_bytes: bytes) -> bool:
        """检测图像是否为低质量（纯黑或纯白或内容过少）
        
        Args:
            image_bytes: 图像字节数据
            
        Returns:
            如果图像是低质量的（纯黑或纯白或内容过少），则返回True，否则返回False
        """
        try:
            if not PIL_AVAILABLE:
                return False  # 如果PIL不可用，则无法检测图像质量
                
            # 使用PIL打开图像
            img = Image.open(io.BytesIO(image_bytes))
            
            # 获取图像尺寸
            width, height = img.size
            
            # 如果图像太小，可能不是有效图像
            if width < 20 or height < 20:
                logger.info(f"图像尺寸过小 ({width}x{height})，判定为低质量图像")
                return True
                
            # 转换为灰度图像
            if img.mode != 'L':
                img = img.convert('L')
                
            # 获取图像数据
            img_array = np.array(img)
            
            # 计算图像的直方图
            hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
            
            # 计算黑色和白色像素的比例
            total_pixels = img_array.size
            black_pixels = hist[0]  # 0是黑色
            white_pixels = hist[255]  # 255是白色
            
            # 计算暗色和亮色像素的比例（不仅仅是纯黑纯白）
            dark_pixels = np.sum(hist[:25])  # 前25个灰度值（较暗）
            light_pixels = np.sum(hist[230:])  # 后26个灰度值（较亮）
            
            black_ratio = black_pixels / total_pixels
            white_ratio = white_pixels / total_pixels
            dark_ratio = dark_pixels / total_pixels
            light_ratio = light_pixels / total_pixels
            
            # 计算图像的标准差，用于检测图像的复杂度
            std_dev = np.std(img_array)
            
            # 计算图像的熵，用于检测图像的信息量
            # 熵越高，图像包含的信息越多
            hist_norm = hist / total_pixels
            hist_norm = hist_norm[hist_norm > 0]  # 去除零值，避免log(0)
            entropy = -np.sum(hist_norm * np.log2(hist_norm)) if len(hist_norm) > 0 else 0
            
            # 检查图像是否为纯黑或纯白
            is_mostly_black_white = (black_ratio > self.quality_threshold or white_ratio > self.quality_threshold) and std_dev < 50
            
            # 检查图像是否为暗色或亮色（不仅仅是纯黑纯白）
            is_mostly_dark_light = (dark_ratio > 0.9 or light_ratio > 0.9) and std_dev < 40
            
            # 检查图像是否为整页图像（通常页面大小且内容少）
            is_full_page = width > 500 and height > 700 and entropy < 3.0
            
            # 检查图像是否内容过少（低熵值）
            is_low_content = entropy < 1.5 and std_dev < 30
            
            # 记录详细的判断信息，便于调试
            if is_mostly_black_white or is_mostly_dark_light or is_full_page or is_low_content:
                logger.info(f"图像质量检测结果: 黑色比例={black_ratio:.3f}, 白色比例={white_ratio:.3f}, 暗色比例={dark_ratio:.3f}, 亮色比例={light_ratio:.3f}")
                logger.info(f"图像复杂度: 标准差={std_dev:.3f}, 熵值={entropy:.3f}, 尺寸={width}x{height}")
                
                if is_mostly_black_white:
                    logger.info("检测为纯黑或纯白图像")
                if is_mostly_dark_light:
                    logger.info("检测为暗色或亮色图像")
                if is_full_page:
                    logger.info("检测为整页图像")
                if is_low_content:
                    logger.info("检测为低内容图像")
                    
                return True
              
            return False
        except Exception as e:
            logger.warning(f"检测图像质量时出错: {str(e)}")
            return False
    
    def _extract_image_by_screenshot(self, page, img) -> Optional[Tuple[bytes, str]]:
        """通过截图方式提取图像，并进行增强处理
        
        Args:
            page: PDF页面对象
            img: 图像信息
            
        Returns:
            提取的图像字节数据和扩展名的元组，如果提取失败则返回None
        """
        try:
            # 获取图像在页面上的位置
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            
            if not bbox or not all(isinstance(x, (int, float)) for x in bbox):
                logger.warning("无法获取有效的图像边界框")
                return None
                
            # 使用更高的缩放因子渲染页面为图像
            zoom_factor = 3  # 增加到3倍缩放以获得更好的质量
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
            img_bytes = pix.tobytes()
            
            if not CV2_AVAILABLE:
                logger.info("OpenCV不可用，返回原始渲染图像")
                return img_bytes, "png"
                
            # 使用OpenCV处理图像
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 计算图像在页面上的位置
            height, width = img_np.shape[:2]
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = int(x0 * zoom_factor), int(y0 * zoom_factor), int(x1 * zoom_factor), int(y1 * zoom_factor)  # 因为使用了缩放
            
            # 确保坐标在有效范围内
            x0 = max(0, min(x0, width))
            y0 = max(0, min(y0, height))
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            
            # 裁剪图像
            if x1 > x0 and y1 > y0:
                cropped = img_np[y0:y1, x0:x1]
                
                # 图像增强处理
                # 1. 转换为灰度图像
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                
                # 2. 自适应直方图均衡化，增强对比度
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # 3. 检查增强后的图像是否有足够的内容
                std_dev = np.std(enhanced)
                
                # 如果标准差太低，尝试边缘检测
                if std_dev < 20:
                    # 使用Canny边缘检测
                    edges = cv2.Canny(enhanced, 50, 150)
                    # 检查边缘数量
                    edge_count = np.count_nonzero(edges)
                    edge_ratio = edge_count / (edges.shape[0] * edges.shape[1])
                    
                    if edge_ratio > 0.01:  # 如果有足够的边缘
                        logger.info(f"使用边缘检测增强图像，边缘比例: {edge_ratio:.3f}")
                        # 转换回彩色图像
                        enhanced_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                        _, buffer = cv2.imencode(".png", enhanced_color)
                        return buffer.tobytes(), "png"
                
                # 4. 如果边缘检测不成功或标准差足够高，使用增强后的图像
                # 转换回彩色图像
                enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                
                # 5. 锐化处理
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced_color, -1, kernel)
                
                # 转换为PNG格式
                _, buffer = cv2.imencode(".png", sharpened)
                logger.info(f"成功增强并锐化图像，尺寸: {sharpened.shape[1]}x{sharpened.shape[0]}")
                return buffer.tobytes(), "png"
            else:
                logger.warning(f"裁剪区域无效: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
            
            return None
        except Exception as e:
            logger.warning(f"通过截图提取图像时出错: {str(e)}")
            return None
            
    def _extract_image_by_page_render(self, page, img) -> Optional[Tuple[bytes, str]]:
        """通过渲染整个页面的方式提取图像，并进行增强处理
        
        Args:
            page: PDF页面对象
            img: 图像信息
            
        Returns:
            提取的图像字节数据和扩展名的元组，如果提取失败则返回None
        """
        try:
            # 获取图像在页面上的位置
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            
            if not bbox or not all(isinstance(x, (int, float)) for x in bbox):
                logger.warning("无法获取有效的图像边界框")
                return None
                
            # 渲染整个页面为高分辨率图像
            zoom = 5  # 使用5倍缩放以获得更好的质量
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # 将pixmap转换为PIL图像
            if PIL_AVAILABLE:
                img_data = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_data))
                
                # 裁剪图像区域
                x0, y0, x1, y1 = bbox
                x0, y0, x1, y1 = int(x0 * zoom), int(y0 * zoom), int(x1 * zoom), int(y1 * zoom)  # 因为使用了zoom倍缩放
                
                # 确保坐标在有效范围内
                width, height = pil_img.size
                x0 = max(0, min(x0, width))
                y0 = max(0, min(y0, height))
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                
                # 裁剪图像
                if x1 > x0 and y1 > y0:
                    cropped_img = pil_img.crop((x0, y0, x1, y1))
                    
                    # 图像增强处理
                    # 1. 转换为灰度图像进行分析
                    gray_img = cropped_img.convert('L')
                    img_array = np.array(gray_img)
                    
                    # 2. 计算图像统计信息
                    std_dev = np.std(img_array)
                    
                    # 3. 如果图像标准差低（低对比度），尝试增强
                    if std_dev < 30:
                        logger.info(f"检测到低对比度图像，标准差: {std_dev:.3f}，尝试增强")
                        
                        # 尝试使用PIL的对比度增强
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Contrast(cropped_img)
                        enhanced_img = enhancer.enhance(2.0)  # 增强对比度
                        
                        # 锐化处理
                        enhancer = ImageEnhance.Sharpness(enhanced_img)
                        enhanced_img = enhancer.enhance(1.5)  # 增强锐度
                        
                        # 如果有OpenCV，尝试更高级的处理
                        if CV2_AVAILABLE:
                            # 转换为OpenCV格式
                            img_buffer = io.BytesIO()
                            enhanced_img.save(img_buffer, format="PNG")
                            img_buffer.seek(0)
                            img_bytes = np.frombuffer(img_buffer.getvalue(), np.uint8)
                            cv_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                            
                            # 自适应直方图均衡化
                            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            enhanced_gray = clahe.apply(gray)
                            
                            # 边缘检测
                            edges = cv2.Canny(enhanced_gray, 50, 150)
                            edge_count = np.count_nonzero(edges)
                            edge_ratio = edge_count / (edges.shape[0] * edges.shape[1])
                            
                            if edge_ratio > 0.01:  # 如果有足够的边缘
                                logger.info(f"使用边缘检测增强图像，边缘比例: {edge_ratio:.3f}")
                                # 使用边缘增强原图
                                enhanced_cv = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                                # 锐化
                                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                                sharpened = cv2.filter2D(enhanced_cv, -1, kernel)
                                
                                # 转换回PIL
                                _, buffer = cv2.imencode(".png", sharpened)
                                enhanced_img = Image.open(io.BytesIO(buffer.tobytes()))
                        
                        # 保存增强后的图像
                        img_buffer = io.BytesIO()
                        enhanced_img.save(img_buffer, format="PNG")
                        logger.info("成功增强图像")
                        return img_buffer.getvalue(), "png"
                    
                    # 保存为PNG格式
                    img_buffer = io.BytesIO()
                    cropped_img.save(img_buffer, format="PNG")
                    return img_buffer.getvalue(), "png"
                
                # 如果裁剪失败，返回None
                logger.warning(f"裁剪区域无效: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
                return None
            else:
                # 如果PIL不可用，直接返回PNG格式的图像
                logger.info("PIL不可用，返回原始渲染图像")
                return pix.tobytes("png"), "png"
        except Exception as e:
            logger.warning(f"通过页面渲染提取图像时出错: {str(e)}")
            return None
    
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
        
    def _calculate_image_hash(self, image_bytes: bytes) -> str:
        """计算图像的哈希值，用于图像去重
        
        Args:
            image_bytes: 图像字节数据
            
        Returns:
            图像的哈希值字符串
        """
        try:
            if not PIL_AVAILABLE:
                # 如果PIL不可用，使用简单的MD5哈希
                import hashlib
                return hashlib.md5(image_bytes).hexdigest()
                
            # 使用PIL计算感知哈希
            img = Image.open(io.BytesIO(image_bytes))
            
            # 调整大小为8x8
            img = img.convert('L').resize((8, 8), Image.LANCZOS)
            
            # 计算平均值
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            
            # 生成哈希值（大于平均值为1，否则为0）
            bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
            
            # 将二进制字符串转换为十六进制
            hex_hash = ''
            for i in range(0, len(bits), 4):
                hex_hash += hex(int(bits[i:i+4], 2))[2:]
                
            return hex_hash
        except Exception as e:
            logger.warning(f"计算图像哈希值时出错: {str(e)}")
            # 出错时返回随机哈希，避免跳过图像
            import random
            return str(random.getrandbits(128))

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    
    # 设置日志级别为INFO
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 设置PDF文件路径和输出目录
    pdf_path = project_root / "Data" / "example.pdf"
    output_dir = project_root / "Result" / "test_image_extractor"
    
    print(f"\n{'='*50}")
    print(f"图像提取测试开始")
    print(f"{'='*50}")
    print(f"PDF文件: {pdf_path}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}\n")
    
    # 检查PDF文件是否存在
    if not pdf_path.exists():
        print(f"错误: PDF文件 {pdf_path} 不存在!")
        sys.exit(1)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 初始化图像提取器
        extractor = ImageExtractor(quality_threshold=0.9)
        
        # 提取图像
        image_paths = extractor.extract_images_from_pdf(str(pdf_path), str(output_dir))
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"图像提取完成")
        print(f"{'='*50}")
        print(f"共提取了 {len(image_paths)} 个图像")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"{'='*50}\n")
        
        # 显示提取的图像信息
        if image_paths:
            print(f"提取的图像列表:")
            print(f"{'-'*50}")
            
            for i, image_path in enumerate(image_paths, 1):
                # 获取图像标题
                caption = extractor.get_image_caption(image_path)
                caption_text = f"标题: {caption}" if caption else "无标题"
                
                # 获取图像文件大小
                file_size = Path(image_path).stat().st_size / 1024  # KB
                
                print(f"{i}. {Path(image_path).name} ({file_size:.1f} KB) - {caption_text}")
            
            print(f"\n所有图像已保存到: {output_dir}")
        else:
            print("未提取到任何图像!")
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)