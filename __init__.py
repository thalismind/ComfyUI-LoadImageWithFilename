from .nodes import LoadImageWithFilename, LoadImageFolder, SaveImageWithFilename, CropImageByMask

NODE_CLASS_MAPPINGS = {
    "LoadImageWithFilename": LoadImageWithFilename,
    "LoadImageFolder": LoadImageFolder,
    "SaveImageWithFilename": SaveImageWithFilename,
    "CropImageByMask": CropImageByMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithFilename": "Load Image With Filename",
    "LoadImageFolder": "Load Image Folder",
    "SaveImageWithFilename": "Save Image With Filename",
    "CropImageByMask": "Crop Image By Mask",
}