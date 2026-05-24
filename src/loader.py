"""文档加载模块，支持 PDF / TXT / Markdown / CSV 格式"""

from pathlib import Path
from typing import Union, List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_core.documents import Document

from src.config import config


class DocumentLoader:
    """统一文档加载器，根据文件扩展名自动选择加载器"""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".markdown": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def __init__(self):
        self.supported_formats = list(self.LOADER_MAP.keys())

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """加载单个文件，返回 Document 列表"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.LOADER_MAP:
            raise ValueError(
                f"不支持的文件格式: {suffix}，支持的格式: {self.supported_formats}"
            )

        loader_class = self.LOADER_MAP[suffix]
        loader = loader_class(str(path))
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = suffix

        return documents

    def load_from_directory(
        self, directory: Union[str, Path], recursive: bool = False
    ) -> List[Document]:
        """加载目录下的所有支持的文件"""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是有效的目录: {directory}")

        pattern = "**/*" if recursive else "*"
        all_docs = []

        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADER_MAP:
                try:
                    docs = self.load(file_path)
                    all_docs.extend(docs)
                    print(f"  加载成功: {file_path.name} ({len(docs)} 个文档块)")
                except Exception as e:
                    print(f"  加载失败: {file_path.name} -> {e}")

        return all_docs

    def load_with_metadata(
        self, file_path: Union[str, Path], extra_metadata: dict = None
    ) -> List[Document]:
        """加载文件并添加额外的 metadata"""
        docs = self.load(file_path)
        extra = extra_metadata or {}
        for doc in docs:
            doc.metadata.update(extra)
        return docs


def load_documents(
    source: Union[str, Path],
    recursive: bool = False,
    extra_metadata: dict = None,
) -> List[Document]:
    """便捷函数：加载文档或目录"""
    path = Path(source)
    loader = DocumentLoader()

    if path.is_file():
        docs = loader.load_with_metadata(path, extra_metadata)
    elif path.is_dir():
        docs = loader.load_from_directory(path, recursive)
    else:
        raise ValueError(f"无效的路径: {source}")

    return docs
