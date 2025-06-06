from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            

        return requirements

setup(
    name = 'WaferFaultPrediction',
    version = '0.0.1',
    author = 'Monika',
    author_email = 'mgadewar12@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()
)
    
#packages=find_packages(where='src'),            # Automatically finds all packages in 'src'
#package_dir={'': 'src'},                        # Set 'src' as the root directory for packages