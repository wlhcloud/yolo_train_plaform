import os
from flask import current_app

from utils import get_app_root


class ProjectDirManager:
    """项目目录管理器，用于处理每个项目的独立目录结构"""

    @staticmethod
    def get_project_upload_dir(project_id):
        """
        获取项目上传目录路径

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目上传目录的绝对路径
        """
        return os.path.join(get_app_root(), "static", "uploads", str(project_id))

    @staticmethod
    def ensure_project_upload_dir(project_id):
        """
        确保项目上传目录存在，如果不存在则创建

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目上传目录的绝对路径
        """
        project_dir = ProjectDirManager.get_project_upload_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    @staticmethod
    def get_project_dataset_dir(project_id):
        """
        获取项目数据集目录路径

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目数据集目录的绝对路径
        """
        return os.path.join(get_app_root(), "static", "datasets", str(project_id))

    @staticmethod
    def ensure_project_dataset_dir(project_id):
        """
        确保项目数据集目录存在，如果不存在则创建

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目数据集目录的绝对路径
        """
        project_dir = ProjectDirManager.get_project_dataset_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    @staticmethod
    def get_project_model_dir(project_id):
        """
        获取项目模型目录路径

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目模型目录的绝对路径
        """
        return os.path.join(get_app_root(), "static", "models", str(project_id))

    @staticmethod
    def ensure_project_model_dir(project_id):
        """
        确保项目模型目录存在，如果不存在则创建

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目模型目录的绝对路径
        """
        project_dir = ProjectDirManager.get_project_model_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    @staticmethod
    def get_relative_path(full_path):
        """
        将绝对路径转换为相对于应用根目录的路径

        Args:
            full_path (str): 文件的绝对路径

        Returns:
            str: 相对于应用根目录的路径
        """
        return os.path.relpath(full_path, get_app_root())

    @staticmethod
    def get_posix_path(relative_path):
        """
        将相对路径转换为POSIX风格路径（使用正斜杠）

        Args:
            relative_path (str): 相对路径

        Returns:
            str: POSIX风格的路径
        """
        import posixpath
        import os

        return posixpath.join(*relative_path.split(os.sep))

    @staticmethod
    def get_project_material_dir(project_id):
        """
        获取项目数据集目录路径

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目数据集目录的绝对路径
        """
        return os.path.join(
            get_app_root(), "static", "inference_material", str(project_id)
        )

    @staticmethod
    def ensure_project_material_dir(project_id):
        """
        确保项目数据集目录存在，如果不存在则创建

        Args:
            project_id (int): 项目ID

        Returns:
            str: 项目数据集目录的绝对路径
        """
        project_dir = ProjectDirManager.get_project_material_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir
