from examples.modelling.fast_ssc.base import fast_ssc_executor

CODE_RATES = [0.25, 0.33, 0.5, 0.66, 0.75, ]
SNR_RANGE = [i/2 for i in range(2, 9)]
MESSAGES_PER_EXPERIMENT = 1000
REPETITIONS = 50
CODE_LENGTH = 2048
MAX_WORKERS = 7


if __name__ == '__main__':
    fast_ssc_executor(
        codeword_length=CODE_LENGTH,
        code_rates=CODE_RATES,
        snr_range=SNR_RANGE,
        task_repetitions=REPETITIONS,
        messages_per_task=MESSAGES_PER_EXPERIMENT,
    )
