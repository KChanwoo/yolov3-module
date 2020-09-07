from setuptools import setup, find_packages

setup(
    name                = 'yolov3-module',
    version             = '0.1',
    description         = 'YOLOv3 module',
    author              = 'Chanwoo Gwon',
    author_email        = 'arknell@naver.com',
    url                 = 'https://github.com/KChanwoo/yolov3-module.git',
    install_requires    =  ["tensorflow"],
    packages            = find_packages(exclude = []),
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3.6',
    ],
)