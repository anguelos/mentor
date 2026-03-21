from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="mentor",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    version="0.1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "torch", "tqdm", "torchvision", "matplotlib", "seaborn", "tensorboard", "fargv"
    ],
    entry_points={
        "console_scripts": [
            "mtr_report_file=mentor.reporting:main_report_file",
            "mtr_plot_file_hist=mentor.reporting:main_plot_file_hist",
        ],
    },
)
