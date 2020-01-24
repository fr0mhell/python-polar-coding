from examples.modelling.fast_ssc.base import fast_ssc_executor

# (4096, 3072) code built with Bhattacharya parameters method
CODE_RATES = [0.75, ]
# From 2 to 5 with step 0.25 dB
SNR_RANGE = [i/4 for i in range(8, 21)]
MESSAGES_PER_EXPERIMENT = 100
REPETITIONS = 1000
CODE_LENGTH = 4096
MAX_WORKERS = 4


if __name__ == '__main__':
    fast_ssc_executor(
        codeword_length=CODE_LENGTH,
        code_rates=CODE_RATES,
        snr_range=SNR_RANGE,
        task_repetitions=REPETITIONS,
        messages_per_task=MESSAGES_PER_EXPERIMENT,
    )
