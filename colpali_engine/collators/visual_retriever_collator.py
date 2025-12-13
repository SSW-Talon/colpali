from io import BytesIO
from PIL import Image as PILImage

def auto_collate(self, batch, key_prefix: str = ""):
    """
    Automatically collate a batch of documents.
    Supports:
    - str
    - PIL.Image
    - dict {bytes, path}
    """

    def normalize_item(x):
        # ✅ 情况 1：已经是 PIL Image
        if isinstance(x, PILImage.Image):
            return x

        # ✅ 情况 2：dict {'bytes', 'path'}
        if isinstance(x, dict):
            if "bytes" in x and isinstance(x["bytes"], (bytes, bytearray)):
                try:
                    return PILImage.open(BytesIO(x["bytes"])).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Failed to decode image bytes: {e}")
            elif "path" in x and isinstance(x["path"], str):
                try:
                    return PILImage.open(x["path"]).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Failed to open image path {x['path']}: {e}")
            else:
                raise ValueError(f"Unsupported image dict format: {x.keys()}")

        # ✅ 情况 3：str（文本 or 路径）
        if isinstance(x, str):
            return x

        raise ValueError(f"Unsupported batch item type: {type(x)}")

    # ========= normalize =========
    batch = [normalize_item(x) for x in batch]

    # ========= original logic =========
    if isinstance(batch[0], str):
        proc_batch = self.processor.process_texts(texts=batch)

    elif isinstance(batch[0], PILImage.Image):
        proc_batch = self.processor.process_images(images=batch)

    elif isinstance(batch[0], list):
        if isinstance(batch[0][0], str):
            batch_size = len(batch)
            all_texts = [t for sub in batch for t in sub]
            num_negatives = len(all_texts) // batch_size
            proc_batch = self.processor.process_texts(texts=all_texts)

        elif isinstance(batch[0][0], PILImage.Image):
            batch_size = len(batch)
            all_imgs = [img for sub in batch for img in sub]
            num_negatives = len(all_imgs) // batch_size
            proc_batch = self.processor.process_images(images=all_imgs)

        else:
            raise ValueError(f"Unsupported nested batch type: {type(batch[0][0])}")

        for k, v in proc_batch.items():
            if isinstance(v, torch.Tensor):
                proc_batch[k] = v.view(batch_size, num_negatives, *v.shape[1:])

    else:
        raise ValueError(f"Unsupported batch type: {type(batch[0])}")

    return prefix_keys(proc_batch, key_prefix)
