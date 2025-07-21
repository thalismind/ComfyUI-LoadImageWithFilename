import os
import hashlib
import torch
import numpy as np
import json
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
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


class SaveImageWithFilename:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filenames": ("STRING", {"default": "", "tooltip": "Single filename or comma-separated list of filenames. If empty, will use default naming."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images with specified filenames to your ComfyUI output directory."

    def save_images(self, images, filenames="", filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append

        # Parse filenames - handle both single filename and comma-separated list
        filename_list = []
        if filenames:
            # Handle both string and list inputs
            if isinstance(filenames, str):
                # Split by comma and strip whitespace
                filename_list = [f.strip() for f in filenames.split(",") if f.strip()]
            elif isinstance(filenames, list):
                # Already a list, just use it
                filename_list = [str(f).strip() for f in filenames if f]
            else:
                # Convert to string and try to split
                filename_str = str(filenames)
                filename_list = [f.strip() for f in filename_str.split(",") if f.strip()]

        # If no filenames provided or not enough filenames, use default naming
        if not filename_list or len(filename_list) < len(images):
            # Use default naming for remaining images
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

            results = list()
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # Use provided filename if available, otherwise use default naming
                if batch_number < len(filename_list):
                    # Use the provided filename
                    provided_filename = filename_list[batch_number]
                    # Remove extension if present and add .png
                    base_name = os.path.splitext(provided_filename)[0]
                    file = f"{base_name}.png"
                else:
                    # Use default naming
                    filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                    file = f"{filename_with_batch_num}_{counter:05}_.png"

                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
        else:
            # Use provided filenames for all images
            results = list()
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # Use the provided filename
                provided_filename = filename_list[batch_number]
                # Remove extension if present and add .png
                base_name = os.path.splitext(provided_filename)[0]
                file = f"{base_name}.png"

                # Save to output directory
                img.save(os.path.join(self.output_dir, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": "",
                    "type": self.type
                })

        return {"ui": {"images": results}}

    @classmethod
    def IS_CHANGED(s, images, filenames, filename_prefix, **kwargs):
        return hashlib.sha256(str(images).encode()).hexdigest()


class CropImageByMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to crop"}),
                "mask": ("MASK", {"tooltip": "The mask to use for cropping. White pixels (1.0) will be excluded, black pixels (0.0) will be included."})
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_by_mask"
    DESCRIPTION = "Crops an image based on its mask, keeping only areas with black pixels (0.0) in the mask and excluding white pixels (1.0)."

    def crop_by_mask(self, image, mask):
        # Ensure mask is binary (0.0 or 1.0)
        if mask.dim() == 2:
            # Single mask
            binary_mask = (mask > 0.5).float()
            return self._crop_single_image(image, binary_mask)
        else:
            # Batch of masks
            cropped_images = []
            cropped_masks = []

            for i in range(mask.shape[0]):
                single_mask = mask[i]
                single_image = image[i] if image.shape[0] > 1 else image[0]

                binary_mask = (single_mask > 0.5).float()
                cropped_img, cropped_mask = self._crop_single_image(single_image, binary_mask)

                cropped_images.append(cropped_img)
                cropped_masks.append(cropped_mask)

            # Stack results
            if len(cropped_images) > 1:
                return (torch.stack(cropped_images, dim=0), torch.stack(cropped_masks, dim=0))
            else:
                return (cropped_images[0].unsqueeze(0), cropped_masks[0].unsqueeze(0))

    def _crop_single_image(self, image, mask):
        # Convert to numpy for easier manipulation
        img_np = image.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Find the bounding box of black pixels (0.0) in the mask
        # We want to exclude white pixels (1.0), so we look for where mask == 0
        black_pixels = (mask_np == 0)

        if not np.any(black_pixels):
            # If no black pixels, return the original image
            return image, mask

        # Find the bounding box
        rows = np.any(black_pixels, axis=1)
        cols = np.any(black_pixels, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Crop the image and mask
        cropped_img = img_np[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask_np[y_min:y_max+1, x_min:x_max+1]

        # Convert back to torch tensors
        cropped_img_tensor = torch.from_numpy(cropped_img).float()
        cropped_mask_tensor = torch.from_numpy(cropped_mask).float()

        return cropped_img_tensor, cropped_mask_tensor

    @classmethod
    def IS_CHANGED(s, image, mask, **kwargs):
        return hashlib.sha256(str(image).encode() + str(mask).encode()).hexdigest()
