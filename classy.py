import os
from jinja2 import Environment, PackageLoader


BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
STATIC_PATH = os.path.join(BASE_DIRECTORY, 'static')
TEST_FOLDER = os.path.join(BASE_DIRECTORY, 'test')
INIT_FOLDER = os.path.join(BASE_DIRECTORY, 'initial')


def get_filename_class(full_name):
    return os.path.splitext(full_name)[0].split('_')[0]  # first letter of filename


def get_list_of_files(directory, ext):
    for file in os.listdir(directory):
        if file.endswith(ext):
            yield os.path.join(directory, file)
      
    
def get_folders_in(directory, full=False):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            yield file_path if full else filename
  

env = Environment(loader=PackageLoader('classy', 'templates'))
env.filters['filename'] = get_filename_class