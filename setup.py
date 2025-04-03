from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forecasting-agent",
    version="0.1.0",
    author="Vikas Pandey",
    author_email="vikaspandey707@gmail.com",
    description="Forecasting Agent with LLM capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codeloop/forecasting-agent",
    project_urls={
        "Bug Tracker": "https://github.com/codeloop/forecasting-agent/issues",
        "Documentation": "https://github.com/codeloop/forecasting-agent#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "prophet>=1.1.4",
        "darts>=0.24.0",
        "tabulate>=0.9.0",
        "langchain>=0.1.0",
        "langchain-experimental>=0.0.43",
        "colorama>=0.4.6",
        "requests>=2.31.0",
        "langchain-community>=0.0.19",
        "langchain-core>=0.1.17",
        "ollama>=0.1.6",
        "plotly>=5.18.0",
        "statsmodels>=0.14.1",
        "scikit-learn>=1.3.2",
    ],
    entry_points={
        'console_scripts': [
            'fc-agent=fc_agent.main:main',
        ],
    },
)
