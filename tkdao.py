from transformers import BartTokenizer
from torch import tensor
_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
idx=tensor([    0,    20, 10230,  1661,    58,  2216,  2156,   182, 22307,     8,
          2310,    31,     5, 17988,  2241,   687,  3443,  2156,   579,  1120,
          3141,    19, 31729,  2156,   739,  1086, 22126,     7,     5,  2770,
         32617,  1488,  1020,  2480,  6353,    36,     5,   275,     8, 21862,
         20921,    38,   128,   548,   655,    56,  4839,   479,     2])
tokens=_tokenizer.convert_ids_to_tokens(idx)
print(tokens)