from examples.modelling.fast_ssc.base import fast_ssc_executor

CODE_RATES = [0.5, ]
# From 1.5 to 3 with step 0.25 dB
SNR_RANGE = [i/4 for i in range(6, 13)]
MESSAGES_PER_EXPERIMENT = 100
REPETITIONS = 1000
CODE_LENGTH = 8192
MAX_WORKERS = 7
ITERATIONS = [
    {'design_snr': 1.4, },
]


if __name__ == '__main__':
    fast_ssc_executor(
        codeword_length=CODE_LENGTH,
        code_rates=CODE_RATES,
        snr_range=SNR_RANGE,
        task_repetitions=REPETITIONS,
        messages_per_task=MESSAGES_PER_EXPERIMENT,
        additional_code_params=ITERATIONS,
    )
