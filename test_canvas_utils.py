from types import SimpleNamespace
from canvas_utils import canvas_image_data


def test_no_canvas_result():
    assert canvas_image_data(None) is None


def test_no_image_payload_yet():
    class UnsubmittedCanvas:
        @property
        def image_data(self):
            raise RuntimeError("No image data URL available")
    assert canvas_image_data(UnsubmittedCanvas()) is None


def test_submitted_payload_preserved():
    payload = object()
    assert canvas_image_data(SimpleNamespace(image_data=payload)) is payload
