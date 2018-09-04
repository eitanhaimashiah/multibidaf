import torch


def sentence_start_mask(metadata, passage_length):
    """
    Returns a mask tensor of size (batch_size, passage_length) with 1 where
    the tokens are sentence starts and 0 otherwise.
    """
    batch_size = len(metadata)
    sentence_start_lists = [met_sen['sentence_starts'] for met_sen in metadata]
    res = torch.zeros(batch_size, passage_length).cuda()
    # res = torch.zeros(batch_size, passage_length)
    for i, sentence_start_list in enumerate(sentence_start_lists):
        for j in sentence_start_list:
            res[i, j] = 1
    return res
