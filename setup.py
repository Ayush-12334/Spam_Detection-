from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
#It installs dependencies inside an already activated venv ✅
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements 

setup(
    name='spam_detection', # Tip: Avoid spaces in names (use _ or -)
    version='0.0.1',
    author='Ayush',
    author_email='ayush23.ghadai@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)