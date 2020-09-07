import os

def get_all_file(root_dir, ext):
    """
    :param ext 확장자
    :return 확장자명을 가진 파일 리스트
    """
    list_dir = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if ext != None and name[-3:] == ext:
                list_dir.append(os.path.join(path, name))
            else:
                list_dir.append(os.path.join(path, name))
    
    return list_dir
