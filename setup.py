#!/usr/bin/env python3
"""
Setup script for the Document Intelligence Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]
else:
    requirements = [
        "requests>=2.31.0",
        "tiktoken>=0.5.1", 
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0"
    ]

setup(
    name="markdown-for-llms",
    version="1.0.0",
    author="Document Intelligence Team",
    author_email="team@example.com",
    description="A comprehensive pipeline for converting documents to LLM-ready Markdown",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/markdown-for-llms",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "enhanced": [
            "colorama>=0.4.6",
            "rich>=13.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "markdown-pipeline=master_workflow:main",
            "markdown-config=config:main",
            "markdown-clean=clean_markdown:main",
            "markdown-validate=validate_markdown:main",
            "markdown-chunk=chunk_markdown:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/markdown-for-llms/issues",
        "Source": "https://github.com/example/markdown-for-llms",
        "Documentation": "https://github.com/example/markdown-for-llms/wiki",
    },
)
