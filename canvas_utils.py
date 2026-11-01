def canvas_image_data(result):
    """An unsubmitted canvas has no image yet in newer canvas releases."""
    if result is None:
        return None
    try:
        return result.image_data
    except RuntimeError:
        # The component raises until its first image payload is submitted.
        return None
