from setuptools import setup

setup(name='mdlib',
      version='0.1.0',
      description='Move detection library',
      long_description='Move detection library',
      url='https://github.com/noxgle/mdlib.git',
      author='noxgle',
      author_email='dredlord0@gmail.com',
      license='MIT',
      packages=['mdlib'],
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy', 'cv2']
      )
