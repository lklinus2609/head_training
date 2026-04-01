from setuptools import setup, find_packages

setup(
    name="d4head_train",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["configs_schema"],
    python_requires=">=3.10",
    description="Adversarial full-face motion generation training pipeline for humanoid robots",
)
