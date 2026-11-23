import cv2
def apply_clahe(image: cv2.typing.MatLike, clipLimit=3.0, tileGridSize: cv2.typing.Size=(8,8)) -> cv2.typing.MatLike:
    """
    Apply CLAHE on an image to increase it's contrast on local regions

    :param image: the image to process.
    :type image: MatLike
    :param clipLimit: CLAHE's clip limit, caps how many pixels can fall into a single histogram bin within an image tile.
    :type clipLimit: float
    :param tileGridSize: CLAHE's tile/window that will walk on the image.
    :type tileGridSize: Size
    :return: high contrast image.
    :rtype: MatLike
    """
    # Convert from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L (Lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)

    # Merge channels back and convert to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    high_contrast = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return high_contrast
