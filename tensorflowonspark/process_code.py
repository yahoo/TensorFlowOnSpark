import tarfile
import zipfile
import mimetypes
import gzip
import sys
import os
try:
  reload(sys)
  sys.setdefaultencoding('gbk')
except:
  from imp import reload
  reload(sys)

def is_zip(_file):
  if _file.endswith(".zip"):
    return True
  return mimetypes.guess_type(_file)[0] == "application/zip"


def un_zip(_file, target):
  with zipfile.ZipFile(_file, 'r') as zip_file:
    target_name = None
    for name in zip_file.filelist:
      if name and name.orig_filename:
        if not target_name or len(name.orig_filename) < len(target_name):
          target_name = name.orig_filename
    zip_file.extractall(target)
  if target_name:
    return "{0}/{1}".format(target, target_name)
  return target


def is_tar(_file):
  if _file.endswith(".tar"):
    return True
  types = mimetypes.guess_type(_file)
  return types[0] == "application/x-tar" and not types[1]


def is_tgz(_file):
  if _file.endswith(".tar.gz") or _file.endswith(".tgz"):
    return True
  types = mimetypes.guess_type(_file)
  if types[0] == "application/x-tar" and types[1] == "gzip":
    return True
  return False


def is_tar_bz2(_file):
  if _file.endswith(".tar.bz2"):
    return True
  types = mimetypes.guess_type(_file)
  if types[0] == "application/x-tar" and types[1] == "bzip2":
    return True
  return False


def un_tar(_file, target, mode):
  tar_handle = tarfile.open(_file, mode)
  names = tar_handle.getnames()
  name = names[0]
  for tmp in names:
    if len(tmp) < len(name):
      name = tmp
  tar_handle.extractall(target)
  tar_handle.close()
  return "{0}/{1}".format(target, name)


def is_gz(_file):
  if _file.endswith(".gz"):
    return True
  types = mimetypes.guess_type(_file)
  if not types[0] and types[1] == "gzip":
    return True
  return False


def un_gz(_file, target):
  dest = "{0}/{1}".format(target, _file[:-3])
  gz_file = gzip.GzipFile(_file, "rb")
  open(dest, "wb").write(gz_file.read())
  gz_file.close()
  return target


def is_py(_file):
  if _file.endswith(".py"):
    return True
  return mimetypes.guess_type(_file)[0] == "text/x-python"


def un_file(_file):
  target = os.getcwd()
  if not os.path.exists(target):
    os.mkdir(target, 504)
  if is_zip(_file):
    return un_zip(_file, target)
  elif is_tar_bz2(_file):
    return un_tar(_file, target, "r:bz2")
  elif is_tgz(_file):
    return un_tar(_file, target, "r:gz")
  elif is_tar(_file):
    return un_tar(_file, target, "r")
  elif is_gz(_file):
    return un_gz(_file, target)
  elif is_py(_file):
    return os.path.dirname(os.path.abspath(_file))
  else:
    raise Exception("not support un compress {0}".format(_file))

