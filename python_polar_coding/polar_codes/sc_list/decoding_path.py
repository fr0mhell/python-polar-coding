from python_polar_coding.polar_codes.base import DecodingPathMixin
from python_polar_coding.polar_codes.sc import SCDecoder


class SCPath(DecodingPathMixin, SCDecoder):
    """Decoding path of SC List decoder."""
