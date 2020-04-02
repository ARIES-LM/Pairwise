import torch
import _pickle as pickle
import os

from torchtext import data, datasets
from contextlib import ExitStack
import fractions

class NormalField(data.Field):
    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch


