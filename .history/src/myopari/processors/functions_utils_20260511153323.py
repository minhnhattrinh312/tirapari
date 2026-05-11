import numpy as np
from skimage.morphology import remove_small_objects
import torch
from radiomics import featureextractor
from skimage.transform import resize
import onnxruntime as ort

def extract_radiomics_features(image_sitk, mask_sitk, label=2):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    data = extractor.execute(image_sitk, mask_sitk, label=label)
    removed_keys = ['Image', 'Mask', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy',
                         'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
                           'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings',
                           'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash',
                           'diagnostics_Image-original_Dimensionality', 'diagnostics_Image-original_Spacing',
                           'diagnostics_Image-original_Size', 'diagnostics_Image-original_Mean',
                           'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum',
                           'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing',
                           'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
                           'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
                           'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass',
                           'diagnostics_Image-interpolated_Spacing', 'diagnostics_Image-interpolated_Size',
                           'diagnostics_Image-interpolated_Mean', 'diagnostics_Image-interpolated_Minimum',
                           'diagnostics_Image-interpolated_Maximum', 'diagnostics_Mask-interpolated_Spacing',
                           'diagnostics_Mask-interpolated_Size', 'diagnostics_Mask-interpolated_BoundingBox',
                           'diagnostics_Mask-interpolated_VoxelNum', 'diagnostics_Mask-interpolated_VolumeNum',
                           'diagnostics_Mask-interpolated_CenterOfMassIndex', 'diagnostics_Mask-interpolated_CenterOfMass',
                           'diagnostics_Mask-interpolated_Mean', 'diagnostics_Mask-interpolated_Minimum',
                           'diagnostics_Mask-interpolated_Maximum']
    results = {k: v for k, v in data.items() if k not in removed_keys}
    return results
    
def min_max_normalize(image, low_perc=0.05, high_perc=99.05):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image


def pad_background(image, dim2pad=(128, 128), x_shift=0, y_shift=0):
    """
    Pads the image to dim2pad, cropping non-zero regions and shifting the crop by x_shift (width) and y_shift (height) pixels from the center.
    To invert the operation, use:
    inverted_image = np.zeros_like(image)
    inverted_image[crop_index] = padded_image[padded_index]
    """
    if len(image.shape) == 3:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2])
    elif len(image.shape) == 4:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2], image.shape[3])

    # Use np.nonzero to find the indices of all non-zero elements in the image
    nz = np.nonzero(image)

    # Get the minimum and maximum indices along each axis
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)

    # Crop the image to only include non-zero values
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    cropped_img = image[crop_index]
    padded_image = np.zeros(dim2pad)
    # Crop further if any axis is larger than dim2pad
    crop_index_new = crop_index
    if cropped_img.shape[0] > dim2pad[0]:
        cx, cx_pad = cropped_img.shape[0] // 2, dim2pad[0] // 2
        # Shift the crop by y_shift pixels from the center
        cx_shifted = cx + x_shift
        start_x = max(0, cx_shifted - cx_pad)  # Ensure start index is non-negative
        end_x = min(cropped_img.shape[0], start_x + dim2pad[0])  # Ensure end index doesn't exceed image height
        cropped_img = cropped_img[start_x:end_x, :, :]
        crop_index_new = (
            slice(crop_index[0].start + start_x, crop_index[0].start + end_x),
            crop_index[1],
            crop_index[2],
        )

    if cropped_img.shape[1] > dim2pad[1]:
        cy, cy_pad = cropped_img.shape[1] // 2, dim2pad[1] // 2
        # Shift the crop by x_shift pixels from the center
        cy_shifted = cy + y_shift
        start_y = max(0, cy_shifted - cy_pad)  # Ensure start index is non-negative
        end_y = min(cropped_img.shape[1], start_y + dim2pad[1])  # Ensure end index doesn't exceed image width
        cropped_img = cropped_img[:, start_y:end_y, :]
        crop_index_new = (
            crop_index_new[0],
            slice(crop_index[1].start + start_y, crop_index[1].start + end_y),
            crop_index_new[2],
        )

    # Calculate the amount of padding needed along each axis
    pad_widths = [(padded_image.shape[i] - cropped_img.shape[i]) // 2 for i in range(len(cropped_img.shape))]
    # Pad the image with zeros
    padded_index = tuple(
        slice(pad_widths[i], pad_widths[i] + cropped_img.shape[i]) for i in range(len(cropped_img.shape))
    )

    padded_image[padded_index] = cropped_img

    return padded_image, crop_index_new, padded_index


def pad_background_with_index(image, crop_index_new, padded_index, dim2pad=(128, 128)):
    if len(image.shape) == 3:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2])
    elif len(image.shape) == 4:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2], image.shape[3])
    padded_image = np.zeros(dim2pad)
    crop_image = image[crop_index_new]
    padded_image[padded_index] = crop_image
    return padded_image


def invert_padding(original_shape, padded_image, crop_index, padded_index):
    # crop the padded image to the size of the original image
    cropped_img = padded_image[padded_index]

    # create an array of zeros with the same shape as the original image
    inverted_image = np.zeros(original_shape)

    # insert the cropped padded image into the center of the array of zeros
    inverted_image[crop_index] = cropped_img

    return inverted_image


def remove_small_elements(segmentation_mask, min_size_remove=3):
    # Convert segmentation mask values greater than 0 to 1
    pred_mask = segmentation_mask > 0
    # Remove small objects (connected components) from the binary image
    mask = remove_small_objects(pred_mask, min_size_remove)
    # Multiply original segmentation mask with the mask to remove small objects
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation


def make_volume(ndarray, voxel_spacing):
    volume = np.prod(voxel_spacing) * (ndarray.sum())
    return volume

def crop_resize_image(image, new_dim=256):
    """
    Process a 3D numpy image by removing non-zero background, cropping to square,
    resizing, and saving crop_index as slice objects for restoration.
    
    Parameters:
    image (np.ndarray): Input image of shape (x, y, z)
    new_dim (int): Desired dimension for the output square image (new_dim, new_dim, z)
    Returns:
    tuple: (processed_image, restore_info)
        - processed_image: Processed image of shape (new_dim, new_dim, z)
        - restore_info: Dict containing original_shape, crop_index, original_dim, new_dim
    """
    # Step 1: Remove non-zero background using np.nonzero
    nz = np.nonzero(image)
    
    
    # Get min and max indices
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)
    
    # Create crop index for non-zero region
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    
    # Crop to non-zero region
    cropped = image[crop_index]
    
    # Step 2: Cut to min dimension of x, y to make square
    crop_h, crop_w = cropped.shape[:2]
    min_dim = min(crop_h, crop_w)
    
    # Calculate center crop
    start_h = (crop_h - min_dim) // 2
    start_w = (crop_w - min_dim) // 2
    
    square = cropped[start_h:start_h+min_dim, start_w:start_w+min_dim, :]
    
    # Calculate origin indices after square crop as slices
    orig_min_row = min_indices[0] + start_h
    orig_max_row = orig_min_row + min_dim
    orig_min_col = min_indices[1] + start_w
    orig_max_col = orig_min_col + min_dim
    orig_min_z = min_indices[2]
    orig_max_z = max_indices[2] + 1
    
    crop_index = (
        slice(orig_min_row, orig_max_row),
        slice(orig_min_col, orig_max_col),
        slice(orig_min_z, orig_max_z)
    )
    
    # Step 3: Resize to new_dim
    current_dim = square.shape[0]
    
    if current_dim != new_dim:
        resized = resize(square, (new_dim, new_dim, square.shape[2]), 
                       order=1,  # Linear interpolation
                       anti_aliasing=True,
)
        # Ensure output dtype matches input
        resized = resized.astype(square.dtype)
    else:
        resized = square.copy()
    
    # Step 4: Save crop_index for restoration
    restore_info = {
        'original_shape': image.shape,
        'crop_index': crop_index,
        'original_dim': current_dim,
        'new_dim': new_dim
    }
    
    return resized, restore_info

def crop_resize_mask(mask, restore_info):
    """
    Process a 3D numpy segmentation mask using restore_info from image processing,
    cropping to the same square region and resizing to the same dimension.
    
    Parameters:
    mask (np.ndarray): Input segmentation mask of shape (x, y, z), same shape as original image
    restore_info (dict): Restoration information from process_image, containing
                        original_shape, crop_index, original_dim, new_dim
    
    Returns:
    tuple: (processed_mask, restore_info)
        - processed_mask: Processed mask of shape (new_dim, new_dim, z)
        - restore_info: Same restore_info for consistency in restoration
    """
    # Validate mask shape
    if mask.shape != restore_info['original_shape']:
        raise ValueError("Mask shape must match original image shape")
    
    # Step 1: Crop to the square region using crop_index
    crop_index = restore_info['crop_index']
    square = mask[crop_index]
    
    # Step 2: Resize to new_dim using skimage
    current_dim = square.shape[0]
    new_dim = restore_info['new_dim']
    
    if current_dim != new_dim:
        resized = resize(square, output_shape=(new_dim, new_dim, square.shape[2]),
                       order=0,
                       anti_aliasing=False)
        # Ensure output dtype matches input
        resized = resized.astype(np.uint8)
    else:
        resized = square.copy()

    return resized


def restore_mask(processed_mask, restore_info):
    """
    Restore a processed 3D segmentation mask back to its original shape.
    
    Improvements:
    - Uses nearest-neighbor interpolation (order=0)
    - Disables anti-aliasing (to preserve discrete labels)
    - Ensures labels are integers after restoration
    - Handles edge cases (padding, rounding) more robustly
    """

    # Step 1: Resize back to the original square dimension (nearest-neighbor)
    current_dim = processed_mask.shape[0]
    original_dim = restore_info['original_dim']
    
    if current_dim != original_dim:
        resized = resize(
            processed_mask,
            output_shape=(original_dim, original_dim, processed_mask.shape[2]),
            order=0,               # nearest neighbor → preserves labels
            anti_aliasing=False,   # turn off to avoid soft edges
            preserve_range=True    # keep label values as-is
        )
    else:
        resized = processed_mask.copy()
    
    # Step 2: Initialize empty mask of the original shape
    original_shape = restore_info['original_shape']
    restored = np.zeros(original_shape, dtype=np.int16)
    
    # Step 3: Paste the restored square into its original crop position
    crop_index = restore_info['crop_index']
    restored[crop_index] = resized.astype(np.int16)  # round & ensure int labels

    return restored



def preprocess_data_array(image, dim_resize=256, onnx = False):
    data = {}
    image = min_max_normalize(image)

    resized_image, restore_info = crop_resize_image(image, dim_resize)
    data["restore_info"] = restore_info
    batch_images = []
    for i in range(resized_image.shape[-1]):
        slice_inputs = resized_image[..., i : i + 1]  
        slices_image = torch.from_numpy(slice_inputs.transpose(-1, 0, 1))  # shape (1, dim_resize, dim_resize)
        batch_images.append(slices_image)
    if onnx:
        # batch of numpy array with shape (n, 1, dim_resize, dim_resize)
        batch_images = np.stack(batch_images, axis=0).astype(np.float32)
    batch_images = torch.stack(batch_images).float()  # shape (n,1, dim_resize, dim_resize)
    data["image"] = batch_images
    return data

def predict_patches(images, model, num_classes=4, batch_size=8, device="cuda", fp16=False):
    """return the patches"""
    prediction = torch.zeros(
        (images.size(0), num_classes, images.size(2), images.size(3)),
        device=device,
    )

    batch_start = 0
    batch_end = batch_size
    while batch_start < images.size(0):
        image = images[batch_start:batch_end]
        with torch.no_grad():
            image = image.to(device)
            if fp16:
                image = image.half()
            y_pred = model(image)
            prediction[batch_start:batch_end] = y_pred
        batch_start += batch_size
        batch_end += batch_size
    return prediction.cpu().numpy()

def predict_data_model(data, model, num_classes=4, batch_size=8, device="cuda", min_size_remove=800, fp16=False, onnx=False):
    if onnx:
        probability_output = predict_patches_onnx_gpu(data["image"], model, num_classes=num_classes, batch_size=batch_size)
    else:
        probability_output = predict_patches(data["image"], model, num_classes=num_classes, batch_size=batch_size, device=device, fp16=fp16) # shape (n, num_classes, dim_resize, dim_resize)
    seg = np.argmax(probability_output, axis=1).transpose(1, 2, 0)  # shape (dim_resize, dim_resize, n)
    seg = remove_small_elements(seg, min_size_remove=min_size_remove)
    invert_seg = restore_mask(seg, data["restore_info"])

    return invert_seg

def predict_data_model_emidec(data, model, patient="P", batch_size=8, device="cuda", mvo=True, task="train_full", 
        min_size_remove=500, fp16=False, onnx=False):
    if onnx:
        probability_output = predict_patches_onnx_gpu(data["image"], model, num_classes=5, batch_size=batch_size)
    else:
        probability_output =  predict_patches(data["image"], model, num_classes=5, batch_size=batch_size, device=device, fp16=fp16)
    seg = np.argmax(probability_output, axis=1).transpose(1, 2, 0)
    seg = remove_small_elements(seg, min_size_remove=min_size_remove)

    myo = np.sum(seg == 2) + np.sum(seg == 3) + np.sum(seg == 4)
    infarction = np.sum(seg == 3) + np.sum(seg == 4)
    frequency_infarction = infarction / myo
    if frequency_infarction < 0.025:
        seg[seg == 3] = 2
        seg[seg == 4] = 2
    elif patient == "P" and not mvo:
        seg[seg == 4] = 3

    if task == "train_combine":
        seg[seg == 4] = 3
    invert_seg = restore_mask(seg, data["restore_info"])
    return invert_seg





def predict_patches_onnx_gpu(images, ort_session, num_classes=4, batch_size=8):
    """Returns the patches using ONNX Runtime on GPU"""

    # Prepare an empty array for predictions
    prediction = np.zeros(
        (images.shape[0], num_classes, images.shape[2], images.shape[3]), dtype=np.float32
    )

    batch_start = 0
    batch_end = batch_size
    while batch_start < images.shape[0]:
        image = images[batch_start:batch_end]
        # Run inference with ONNX Runtime
        inputs = {ort_session.get_inputs()[0].name: image}
        y_pred = ort_session.run(None, inputs)

        # Store the prediction result
        prediction[batch_start:batch_end] = y_pred[0]

        batch_start += batch_size
        batch_end += batch_size

    return prediction