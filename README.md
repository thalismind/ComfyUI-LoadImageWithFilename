# ComfyUI LoadImageWithFilename

This custom node extends ComfyUI's image loading functionality with filename output and folder loading capabilities.

## Features

### LoadImageWithFilename
- **Enhanced Load Image Node**: Based on the original ComfyUI LoadImage node
- **Filename Output**: Returns the filename of the loaded image as a STRING output
- **Compatible**: Maintains all original functionality (IMAGE and MASK outputs)

### LoadImageFolder
- **Folder Loading**: Load all images from a selected folder
- **Batch Processing**: Returns all images concatenated as a single tensor
- **Filename Tracking**: Returns a comma-separated list of all loaded filenames
- **Error Handling**: Gracefully handles corrupted or unsupported image files

## Installation

1. Place `nodes.py` in your ComfyUI `custom_nodes` directory
2. Restart ComfyUI
3. The new nodes will appear in the "image" category

## Usage

### LoadImageWithFilename
- **Input**: Select an image file from the dropdown
- **Outputs**:
  - `image`: The loaded image tensor
  - `mask`: The image mask (if available)
  - `filename`: The filename of the loaded image

### LoadImageFolder
- **Input**: Select a folder from the dropdown
- **Outputs**:
  - `image`: All images from the folder concatenated as a single tensor
  - `mask`: All masks concatenated as a single tensor
  - `filenames`: Comma-separated list of all loaded filenames

## Requirements

- ComfyUI
- PIL (Pillow)
- PyTorch
- NumPy

## Notes

- The LoadImageFolder node will skip any non-image files in the selected folder
- If no valid images are found in a folder, empty tensors will be returned
- All images in a folder must have the same dimensions for proper concatenation
- Error messages are printed to console for any files that fail to load

## Based On

- [ComfyUI LoadImage Node](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py)
- [Issue #8699](https://github.com/comfyanonymous/ComfyUI/issues/8699) - Request for filename output functionality