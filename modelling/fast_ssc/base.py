from functools import partial

from modelling.runner import executor
from polar_codes import FastSSCPolarCode

COLLECTION = 'fast_ssc'


fast_ssc_executor = partial(
    executor,
    code_class=FastSSCPolarCode,
    collection_name=COLLECTION,
)
