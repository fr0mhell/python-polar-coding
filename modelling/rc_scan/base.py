from functools import partial

from modelling.runner import executor
from polar_codes import RCSCANPolarCode

COLLECTION = 'rc_scan'


rc_scan_executor = partial(
    executor,
    code_class=RCSCANPolarCode,
    collection_name=COLLECTION,
)
