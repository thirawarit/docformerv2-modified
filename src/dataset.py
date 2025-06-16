import math
import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from ocrmac import ocrmac
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

PAD_TOKEN_BBOX = (0, 0, 0, 0)


class FormatFileError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class FileExistError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    filename = image.filename
    image = image.convert("RGB")
    image.filename = filename
    return image


def apply_ocr(input: Union[str | Image.Image]) -> Dict:
    """_summary_

    Args:
        input (Union[str  |  Image.Image]): Image's path or image object for applying ocr.

    Returns:
        Dict: _description_
    """

    if isinstance(input, Image.Image):
        image = input
    elif isinstance(input, str):
        if not input.lower().endswith(("jpeg", "jpg", "png")):
            raise FormatFileError("input must be \"jpeg\", \"jpg\", and \"png\""
                                         f", but {os.path.splitext(input)[-1]}.")
        if not os.path.exists(input):
            raise FileNotFoundError(f"{input} does not exist.")
        
        image = Image.open(input)
    else:
        raise TypeError(f"Type of input must be `str` or `Image.Image`, but `{type(input).__name__}`")

    # On a Mac, the origin point of a bounding box is typically defined as the bottom-left corner of the image 
    # when dealing with normalized coordinates.
    # When you use ocrmac.OCR.recognize(px=False), each `bbox` is normalized xywh-format in ** a Mac coordinate system **.
    # When you use ocrmac.OCR.recognize(px=True), each `bbox` is absoluted xyxy-format in ** a general coordinate system **.
    annotations = ocrmac.OCR(image, language_preference=['th-TH'], recognition_level='accurate').recognize(px=True)

    size_image: Tuple[int] = image.size # (width_image, height_image)
    sl_texts, _, bboxes = map(list, zip(*annotations)) if len(annotations) and len(annotations[0]) else ([], [], [])

    norm_bboxes: list[Tuple] = []
    for x0, y0, x1, y1 in bboxes:
        norm_bboxes.append((
            x0 / size_image[0],
            y0 / size_image[1],
            x1 / size_image[0],
            y1 / size_image[1],
        ))
    return {"single_line_text": sl_texts, "bbox": bboxes, "norm_bbox": norm_bboxes, "size_image": size_image}


def smart_resize(
    width: int, height: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int]:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


def get_token_bboxes(
    norm_text_bboxes: List[Tuple[float]], 
    pad_token_bbox: Tuple[float], 
    word_ids: List[Optional[int]],
) -> Tuple[List[Tuple[float]], int]:
    """_summary_

    Args:
        norm_text_bboxes (List[Tuple[float]]): _description_
        pad_token_bbox (Tuple[float]): _description_
        word_ids (List[Optional[int]]): _description_

    Returns:
        Tuple[List[Tuple[float]], int]: _description_
    """
    token_bboxes = []
    for pad_start_idx, bbox_idx in enumerate(word_ids):
        # break loop when it is now pad_token_id (word_id = None).
        if bbox_idx is None:
            break
        token_bboxes.append(norm_text_bboxes[bbox_idx])
    len_pad = len(word_ids) - (pad_start_idx)
    token_bboxes.extend([pad_token_bbox] * len_pad)
    return token_bboxes, pad_start_idx


def get_centroid(xyxy_bboxes):
    centroids = []
    for bbox in xyxy_bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        centroid_x = bbox[0] + width // 2
        centroid_y = bbox[1] + height // 2
        centroids.append([centroid_x, centroid_y])
    return centroids


def get_relative_distance(bboxes, centroids, pad_tokens_start_idx):
    a_rel_x = []
    a_rel_y = []
    for idx, bbox in enumerate(bboxes):

        if idx > pad_tokens_start_idx or idx == len(bboxes) - 1:
            a_rel_x.append([0]*8)
            a_rel_y.append([0]*8)
            continue

        next_bbox = bboxes[idx+1]

        a_rel_x.append(
            [
                bbox[0], # top-left x
                bbox[2], # buttom-right x
                bbox[2] - bbox[0], # width
                next_bbox[0] - bbox[0], # difference top-left x (CW)
                next_bbox[2] - bbox[2], # difference top-right x (CW)
                next_bbox[2] - bbox[2], # difference buttom-right x (CW)
                next_bbox[0] - bbox[0], # difference buttom-left x (CW)
                centroids[idx+1][0] - centroids[idx][0],
            ]
        )

        a_rel_y.append(
            [
                bbox[1], # top-left y
                bbox[3], # buttom-right y
                bbox[3] - bbox[1], # height
                next_bbox[1] - bbox[1], # difference top-left y (CW)
                next_bbox[3] - bbox[3], # difference top-right y (CW)
                next_bbox[3] - bbox[3], # difference buttom-right y (CW)
                next_bbox[1] - bbox[1], # difference buttom-left y (CW)
                centroids[idx+1][1] - centroids[idx][1],
            ]
        )

    return a_rel_x, a_rel_y


def create_mask_indices(input_ids: torch.Tensor, special_tokens_mask: torch.Tensor, mlm_probability: float=0.15):
    """input_token_ids without special_token_ids are selected by given probability for some purposes.

    Args:
        input_ids (_type_): _description_
        special_tokens_mask (_type_): _description_
        mlm_probability (float, optional): _description_. Defaults to 0.15.

    Returns:
        _type_: _description_
    """
    if not isinstance(input_ids, torch.Tensor) or not isinstance(special_tokens_mask, torch.Tensor):
        raise ValueError(f"received an invalid combination of arguments - got ({type(input_ids).__name__}, {type(special_tokens_mask)[0].__name__}), but expected (torch.Tensor, torch.Tensor)")
    num_tokens = input_ids.size()[0]
    num_mask = int(mlm_probability * num_tokens)
    index_special_token = special_tokens_mask[0].argwhere()
    index_special_token = index_special_token.squeeze()

    masked_indices = torch.randperm(num_tokens)
    masked_indices = [v for v in masked_indices if v not in index_special_token]
    masked_indices = torch.tensor(masked_indices, dtype=torch.long)
    return masked_indices[:num_mask]


def apply_mask(inputs, tokenizer, mlm_probability=0.5):
    labels = inputs.clone()
    input_ids = labels.clone()

    special_tokens_mask = [tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)]
    special_tokens_mask = torch.as_tensor(special_tokens_mask, dtype=torch.bool)

    masked_indices = create_mask_indices(labels, special_tokens_mask, mlm_probability)
    mask = torch.zeros(labels.shape, dtype=torch.bool)
    mask[masked_indices] = True
    labels[~mask] = -100 # We only compute loss on masked tokens

    # 80% of selected tokens, we replace masked input tokens with tokenizer.mask_token ([MASK])
    replace_probability = 0.8
    num_replaced_mask = int(replace_probability * len(masked_indices))
    indices_replaced = masked_indices[:num_replaced_mask]
    mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    mask[indices_replaced] = True
    if tokenizer.mask_token_id is None and tokenizer.unk_token_id is None:
        raise ValueError("`tokenizer.mask_token` does not exist, please check tokenizer again. \nNote that: Use `tokenizer.add_special_tokens({\"mask_token\": \"<|mask|>\"})`")
    elif tokenizer.unk_token_id is not None:
        input_ids[mask] = tokenizer.unk_token_id
    else:
        input_ids[mask] = tokenizer.mask_token_id

    # 10% of selected tokens, we replace masked input tokens with random word
    # The rest of selected tokens (10% of selected tokens) we keep the masked input tokens unchanged
    random_probability = 0.5
    indices_remain = masked_indices[num_replaced_mask:]
    indices_random, indices_unchanged = (indices_remain, torch.as_tensor([])) if len(indices_remain) == 1 else \
        indices_remain.split(max(int(random_probability * len(indices_remain)) + 1, 1))
    mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    mask[indices_random] = True
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[mask] = random_words[indices_random]

    return input_ids, labels

def get_align_bbox(bbox: Tuple[float], resized_image_size: Tuple[int]) -> Tuple[int]:
    resized_w, resized_h = resized_image_size
    norm_x1, norm_y1, norm_x2, norm_y2 = bbox 

    x1 = int(resized_w * norm_x1)
    x2 = int(resized_w * norm_x2)
    y1 = int(resized_h * norm_y1)
    y2 = int(resized_h * norm_y2)
    return (x1, y1, x2, y2)


def create_feature(
    image: Union[str | Image.Image],
    tokenizer,
    use_ocr: bool = True,
    texts: Optional[List[str]] = None,
    bboxes: Optional[List[int]] = None,
    target_size: Optional[Tuple[int]] = (386, 500),
    max_seq_length: int = 512,
    apply_mask_for_mlm: bool = False,
    save_to_disk: bool = False,
    path_to_save: Optional[str] = "./cache",
    add_batch_dim: bool = False,
    patch_size: int = 14,
    merge_size: int = 2,
):
    """

    Args:
        image (Union[str  |  Image.Image]): _description_
        tokenizer (_type_): _description_
        use_ocr (bool, optional): _description_. Defaults to True.
        texts (Optional[List[str]], optional): _description_. Defaults to None.
        bboxes (Optional[List[int]], optional): _description_. Defaults to None.
        target_size (Optional[Tuple[int]], optional): Size of image that you want to change (width, height). Defaults to (384, 500).
        max_seq_length (int, optional): _description_. Defaults to 512.
        apply_mask_for_mlm (bool, optional): _description_. Defaults to False.
        save_to_disk (bool, optional): _description_. Defaults to False.
        path_to_save (Optional[str], optional): _description_. Defaults to "./cache".
        add_batch_dim (bool, optional): _description_. Defaults to False.
        patch_size (int, optional): _description_. Defaults to 14.
        merge_size (int, optional): _description_. Defaults to 2.

    Raises:
        FormatFileError: _description_
        FileNotFoundError: _description_
        TypeError: _description_
        BrokenPipeError: _description_

    Returns:
        _type_: _description_
    """

    # Step 1: Load original image and extract OCR entries.
    if isinstance(image, str):
        if not image.lower().endswith(("jpeg", "jpg", "png")):
            raise FormatFileError("image must be \"jpeg\", \"jpg\", and \"png\""
                                         f", but {os.path.splitext(image)[-1]}.")
        if not os.path.exists(image):
            raise FileNotFoundError(f"{image} does not exist.")
        
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Type of image must be `str` or `Image.Image`, but `{type(image).__name__}`")
    
    image = convert_to_rgb(image)

    if (texts is None and bboxes is None) and use_ocr:
        ocr_output: dict = apply_ocr(image)
        texts = ocr_output["single_line_text"]
        bboxes = ocr_output["norm_bbox"]

    if texts is None or bboxes is None:
        raise BrokenPipeError("texts is None or bboxes is None, Please check step 1")
    
    # Step 2: Resize the image.
    if target_size is not None:
        resized_width, resized_height = smart_resize(*target_size)
    else:
        resized_width, resized_height = smart_resize(*image.size)
    resized_image = image.resize((resized_width, resized_height))

    # Step 3: Tokenize single line texts and get their bounding boxes.
    encoding = tokenizer(
        texts,
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
        # return_tensors="pt",
        is_split_into_words=True
    )

    PAD_TOKEN_BBOX = (0., 0., 0., 0.)
    norm_token_bboxes, pad_start_idx = get_token_bboxes(bboxes, PAD_TOKEN_BBOX, encoding.word_ids())
    
    assert len(encoding["input_ids"]) == len(norm_token_bboxes), (
        "input_ids does not match norm_token_bboxes, "
        f'{len(encoding["input_ids"])} != {len(norm_token_bboxes)}'
    )

    encoding["norm_bboxes"] = norm_token_bboxes

    # Step 4: Apply mask for the sake of pre-training.
    if apply_mask_for_mlm:
        encoding["input_ids"] = torch.as_tensor(encoding["input_ids"])
        encoding["input_ids"], encoding["mlm_labels"] = apply_mask(encoding["input_ids"], tokenizer)
        assert len(encoding["mlm_labels"]) == max_seq_length, f"Length of mlm_labels ({len(encoding['mlm_labels'])}) != Length of max_seq_length ({max_seq_length})"

    assert len(encoding["input_ids"]) == max_seq_length, f"Length of input_ids ({len(encoding['input_ids'])}) != Length of max_seq_length ({max_seq_length})"

    # Step 5: Normalize the image
    patch = ToTensor()(resized_image) # (W, H, C) -> (C, H, W)
    channel = patch.shape[0]
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patch = patch.reshape(
        channel,                #0
        grid_h // merge_size,   #1
        merge_size,             #2
        patch_size,             #3
        grid_w // merge_size,   #4
        merge_size,             #5
        patch_size,             #6
    )
    patch = patch.permute(1, 4, 2, 5, 0, 3, 6)
    flatten_patch = patch.reshape(
        grid_h * grid_w, channel * patch_size * patch_size
    )
    encoding["pixel_values"] = flatten_patch
    encoding["image_grid_hw"] = (grid_h, grid_w)

    # Step 6: Align bounding boxes.
    aligned_token_bboxes = [get_align_bbox(bbox, resized_image.size) for bbox in norm_token_bboxes]

    # Step 6: Add the relative distances in the normalized grid.
    bboxes_centroids = get_centroid(aligned_token_bboxes)
    a_rel_x, a_rel_y = get_relative_distance(aligned_token_bboxes, bboxes_centroids, pad_start_idx)

    # Step 7: Convert all to tensors.
    encoding = {k: torch.as_tensor(v) for k, v in encoding.items()}
    encoding["x_features"] = torch.as_tensor(a_rel_x)
    encoding["y_features"] = torch.as_tensor(a_rel_y)

    # Step 8: Add tokens for debugging.
    input_ids = encoding["input_ids"]
    encoding["token_without_padding"] = tokenizer.convert_ids_to_tokens(input_ids)
    encoding["texts"] = texts

    # Step 9: Save to disk.
    if save_to_disk:
        if path_to_save is None:
            path_to_save = "./cache"
        os.makedirs(path_to_save, exist_ok=True)
        image_name = os.path.basename(image.filename if isinstance(image, Image.Image) else image)
        image_name, _ = os.path.splitext(image_name)
        tmn_path = os.path.join(path_to_save, f"{image_name}.joblib")
        with open(tmn_path, "wb") as f:
            joblib.dump(encoding, f, compress=3)

    # step 10: keys to keep, norm_bboxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
    keys = ['input_ids', 'x_features', 'y_features', 'norm_bboxes', 'image_grid_hw']

    if apply_mask_for_mlm:
        keys.append('mlm_labels')

    final_encoding = {k: encoding[k] for k in keys}

    # Step 11: Add extra dim for batch.
    if add_batch_dim:
        final_encoding = {k: v.unsqueeze(dim=0) for k, v in final_encoding.items()}

    final_encoding.update({'pixel_values': encoding['pixel_values']}) # except batch

    del encoding
    return final_encoding