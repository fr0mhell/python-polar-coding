import utils
from .base import BasicPolarCode
from .successive_cancellation import SCBranch


class SCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""

    def decode(self, received_message):
        """Decode Polar code with SC decoding algorithm."""
        return self._sc_decode(received_message)

    def _sc_decode(self, llr_estimated_msg):
        """Successive cancellation decoder

        Based on: https://arxiv.org/abs/0807.3917 (page 15).

        """
        decoding_branch = SCBranch(received_message=llr_estimated_msg)

        for j in range(self.N):
            # bit reversing of index allow to first deal with not XORed bits
            i = utils.bitreversed(j, self.n)
            decoding_branch.update_decoding_position(i)
            decoding_branch.update_llrs()
            if self.polar_mask[i] == 1:
                if decoding_branch.llrs[0] > 0:
                    decoding_branch.decoded[i] = 0
                else:
                    decoding_branch.decoded[i] = 1
            else:
                decoding_branch.set_bit_as_frozen()
            decoding_branch.update_bits()

        if self.is_systematic:
            # for systematic code first need to mul decoding result with
            # code generator matrix, and then extract information bits due to
            # polar coding matrix
            return self._extract(self._mul_matrix(decoding_branch.decoded))
        return self._extract(decoding_branch.decoded)
