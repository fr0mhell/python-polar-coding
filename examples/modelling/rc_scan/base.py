from functools import partial

from examples.modelling.runner import executor
from python_polar_coding.polar_codes import RCSCANPolarCode

COLLECTION = 'rc_scan'


rc_scan_executor = partial(
    executor,
    code_class=RCSCANPolarCode,
    collection_name=COLLECTION,
)
