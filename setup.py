# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="probmlpy",  # 库的名称
    version="0.1.0",  # 版本号
    description="A simple math library",
    author="xcz",
    author_email="xiaochengzhe0@qq.com",
    url="https://github.com/xcz0/probmlpy",
    packages=find_packages(),  # 自动查找所有Python包
    install_requires=[ # 这里可以列出你的依赖包
        "numpy",
    ],  
)
