import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # long_description=long_description,
    name="latr-VLSP",
    packages=setuptools.find_packages(),
    install_requires = [
        "ensure",
        "python-box",
        "PyYAML",
        "text-summarizer",
        "numpy",
        "tqdm",
        "pandas",
        "torch>=1.9.0",
        "pytesseract",
        # "google-colab",
        "dataset",
        "pyparsing",
        "torchvision",
        "pytorch_lightning",
        "transformers[torch]",
        "randompad_sequence"
    ]
)