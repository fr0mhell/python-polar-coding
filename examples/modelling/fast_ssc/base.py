from functools import partial

from examples.modelling.runner import executor
from python_polar_coding.polar_codes import FastSSCPolarCode

COLLECTION = 'fast_ssc'


fast_ssc_executor = partial(
    executor,
    code_class=FastSSCPolarCode,
    collection_name=COLLECTION,
)
