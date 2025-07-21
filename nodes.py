import os
import hashlib
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers

class LoadImageWithFilename:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Extract filename from the image path
        filename = os.path.basename(image_path)

        return (output_image, output_mask, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class LoadImageFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"folder_path": ("STRING", {"default": "", "multiline": False})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filenames")
    FUNCTION = "load_folder"

    def load_folder(self, folder_path):
        # If no path provided, return empty tensors
        if not folder_path or not os.path.exists(folder_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")

        # Check if it's a directory
        if not os.path.isdir(folder_path):
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")

        # Get all image files in the folder
        image_files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                # Check if it's an image file
                try:
                    with Image.open(file_path) as img:
                        image_files.append(file)
                except:
                    continue

        if not image_files:
            # Return empty tensors if no images found
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")

        # Load all images in the folder
        all_images = []
        all_masks = []
        all_filenames = []

        for filename in sorted(image_files):
            file_path = os.path.join(folder_path, filename)

            try:
                img = node_helpers.pillow(Image.open, file_path)

                output_images = []
                output_masks = []
                w, h = None, None

                excluded_formats = ['MPO']

                for i in ImageSequence.Iterator(img):
                    i = node_helpers.pillow(ImageOps.exif_transpose, i)

                    if i.mode == 'I':
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")

                    if len(output_images) == 0:
                        w = image.size[0]
                        h = image.size[1]

                    if image.size[0] != w or image.size[1] != h:
                        continue

                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    if 'A' in i.getbands():
                        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    elif i.mode == 'P' and 'transparency' in i.info:
                        mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                    output_images.append(image)
                    output_masks.append(mask.unsqueeze(0))

                if len(output_images) > 1 and img.format not in excluded_formats:
                    output_image = torch.cat(output_images, dim=0)
                    output_mask = torch.cat(output_masks, dim=0)
                else:
                    output_image = output_images[0]
                    output_mask = output_masks[0]

                all_images.append(output_image)
                all_masks.append(output_mask)
                all_filenames.append(filename)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        if not all_images:
            # Return empty tensors if no images could be loaded
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")

        # Concatenate all images and masks
        combined_images = torch.cat(all_images, dim=0)
        combined_masks = torch.cat(all_masks, dim=0)

        return (combined_images, combined_masks, all_filenames)

    @classmethod
    def IS_CHANGED(s, folder_path):
        if not folder_path or not os.path.exists(folder_path):
            return "INVALID"

        # Create a hash based on folder contents
        m = hashlib.sha256()
        try:
            for file in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            m.update(f.read())
                    except:
                        continue
        except:
            return "INVALID"
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, folder_path):
        if not folder_path:
            return "Empty path provided"
        if not os.path.exists(folder_path):
            return "Path does not exist: {}".format(folder_path)
        if not os.path.isdir(folder_path):
            return "Not a directory: {}".format(folder_path)
        return True