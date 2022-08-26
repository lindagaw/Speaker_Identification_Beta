import os

def delete_dir(dir):
    try:
        shutil.rmtree(dir)
        print('deleted directory {}'.format(dir))
    except:
        print('directory {} did not exist. skipping...'.format(dir))
        pass
