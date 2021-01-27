"""This module allow to untar archives"""
import mimetypes
import zipfile as FZ
import bz2

from os import path


def untar(input_path, output_path):
    """
    function allowing to untar files
    """
    print("Reading archive at "+str(input_path))
    ext = mimetypes.guess_type(input_path)
    ext = str(ext[0]) + str(ext[1])

    # if mime type is None
    if ext is None:
        ext = mimetypes.read_mime_types(input_path)
    # if mime type is still None - read the extension
    if ext is None:
        raise UnknowMimeType(path.split(input_path)[1])

    if 'bzip2' in ext:
        with open(input_path, 'rb') as zipfile:

            data = bz2.decompress(zipfile.read())
            print(output_path+(path.split(input_path)[1].split('.')[0]))
            fzip = open(path.join(output_path,
                                  (path.split(input_path)[1].split('.')[0])+'.txt'), 'wb')
            fzip.write(data)
    elif 'zip' in ext:
        zip_ref = FZ.ZipFile(input_path, 'r')
        zip_ref.extractall(output_path)
        zip_ref.close()

    else:
        raise NotImplementedError()
###############################################
# Exception


class UnknowMimeType(Exception):
    """
    Exception throw if no mime type is found
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'MimeType is undefined for the file "'+self.value()+'"'
