from PyCRC.CRCCCITT import CRCCCITT


def bitstring_to_bytes(s):
    """Converts bit string into bytes."""
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')


def calculate_crc_16(message):
    """Calculate CRC 16 with CRCCCITT polynome.

    Args:
        message(np.array): message in binary representation.

    """
    bit_string = ''.join(str(m) for m in message)
    byte_string = bitstring_to_bytes(bit_string)
    return CRCCCITT().calculate(byte_string)


def check_crc_16(message):
    """Using CRC check if message has errors or not.

    Args:
        message(np.array): message in binary representation.

    """
    received_crc = int(''.join([str(m) for m in message[-16::]]), 2)
    check_crc = calculate_crc_16(message[:-16])
    return received_crc == check_crc
