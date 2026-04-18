from setuptools import setup, find_packages

setup(
    name='ProtoEEG-QA',
    version='0.1.0',
    description='Attention-Guided Few-Shot Prototypical Network for ICU EEG Abnormal Pattern Recognition',
    author='Anonymous',  # Do not include author details for anonymous submission
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.1',
        'opencv-python>=4.5.3.56',
        'scipy>=1.7.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'scikit-learn>=1.0.2',
        'tqdm>=4.61.2',
        'pyarrow>=6.0.1',
        'Pillow>=8.2.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
