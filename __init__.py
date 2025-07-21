from .nodes import LoadImageWithFilename, LoadImageFolder

NODE_CLASS_MAPPINGS = {
    "LoadImageWithFilename": LoadImageWithFilename,
    "LoadImageFolder": LoadImageFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithFilename": "Load Image With Filename",
    "LoadImageFolder": "Load Image Folder",
}