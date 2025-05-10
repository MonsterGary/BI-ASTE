import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')
from peng.data.piptxt import BartBPEABSAPipe
from peng.model.bart_absa import BartmarkSeq2SeqModel

from peng.model.debidirectionaltrainer import Trainer
from peng.model.metrics import Seq2SeqSpanMetric
from peng.model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback,EarlyStopCallback,Callback
from fastNLP import Tester
from peng.model.callbacks import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from peng.model.debigenerator import SequenceGeneratorModel
import fitlog
from peng.Vocab import Vocab
from peng.prepare_vocab import VocabHelp

# fitlog.debug()
fitlog.set_log_dir('logs')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='acos/Restaurant-ACOS', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--gatlr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first',  default='AO', type=str, choices=['A','O','AO'])
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score','avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--use_dual_encoder', type=bool, default=True)
parser.add_argument('--use_syn_embed_mlp', type=int, default=0)
parser.add_argument('--save_model', type=int, default=0)

args= parser.parse_args()

lr = args.lr
gatlr = args.gatlr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
use_dual_encoder=args.use_dual_encoder
use_syn_embed_mlp=args.use_syn_embed_mlp
fitlog.add_hyper(args)

#######hyper
#######hyper


demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"

post_vocab = Vocab.load_vocab(f'../data/{dataset_name}/vocab_post.vocab')        #138
deprel_vocab = Vocab.load_vocab(f'../data/{dataset_name}/vocab_deprel.vocab')    #45
postag_vocab = Vocab.load_vocab(f'../data/{dataset_name}/vocab_postag.vocab')    #46
synpost_vocab = VocabHelp.load_vocab(f'../data/{dataset_name}/vocab_synpost.vocab')    #7
vocab = (post_vocab, deprel_vocab, postag_vocab, synpost_vocab)
args.post_size = len(post_vocab)
args.deprel_size = len(deprel_vocab)
args.postag_size = len(postag_vocab)
args.synpost_size = len(synpost_vocab)

@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first, vocab=vocab, paths=f'../data/{dataset_name}/all.tsv')
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
data_bundle.set_input('aftgt_tokens','oftgt_tokens', 'afsrc_tokens','ofsrc_tokens', 'src_seq_len', 'tgt_seq_len', 'head', 'pos_tag',
                      'head_len', 'deprel_ids','word_pair_position', 'tree_based_word_pair_position','word_pair_deprel','matrix_mask',)
data_bundle.set_target('aftgt_tokens','oftgt_tokens', 'tgt_seq_len', 'oe_target_span', 'aesc_target_span',
                       'ae_target_span', 'head', 'pos_tag',
                       'head_len', 'deprel_ids','word_pair_position', 'tree_based_word_pair_position','word_pair_deprel','matrix_mask',)


max_len = 100*3
max_len_a = {
    'penga/14lap': 0.9,
    'penga/14res': 1,
    'penga/15res': 1.2,
    'penga/16res': 0.9,
    'pengb/14lap': 1.1,
    'pengb/14res': 1.2,
    'pengb/15res': 0.9,
    'pengb/16res': 1.2,
    'acos/Restaurant-ACOS': 1.2,
    'acos/Laptop-ACOS': 1.2
}[dataset_name]
max_len_a=max_len_a*3

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
# model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
#                                      copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
model = BartmarkSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False,
                                     use_dual_encoder=use_dual_encoder, use_syn_embed_mlp=use_syn_embed_mlp)

vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None, opinion_first=opinion_first)

import torch
if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    device = 'cuda'
else:
    device = 'cpu'

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
named_parameters=model.named_parameters()
params['params'] = [param for name, param in model.named_parameters() if not ('encoder' in name or 'decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('encoder' in name or 'decoder' in name) and not ('layernorm' in name or 'layer_norm' in name)\
            and not ('GAT' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('encoder' in name or 'decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)
'''

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
named_paremeters=model.named_parameters()
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name or'GAT' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)
'''
params = {'lr':gatlr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('encoder' in name or 'decoder' in name) and ('GAT' in name)and not('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)


callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
#callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))

#callbacks.append(EarlyStopCallback(5))
class CustomCallback(Callback):
    def on_epoch_begin(self):
        # 在这里设置模型的epoch
        self.model.seq2seq_model.set_epoch(self.epoch,self.n_epochs)
# callbacks.append(CustomCallback())


sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)


model_path = None
if save_model:
    model_path = 'save_models/'


if '14res' in dataset_name:
    dev_batch_size=4
elif '15res' in dataset_name:
    dev_batch_size = 16
elif '16res' in dataset_name:
    dev_batch_size = 2
else:
    dev_batch_size = 8

callbacks.append(FitlogCallback(tester={
    'testa': Tester(data=data_bundle.get_dataset('test'), model=model,
                    metrics=metric,
                    batch_size=dev_batch_size, num_workers=0, device=None, verbose=0, use_tqdm=False,
                    fp16=False)
    }
))

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=4,
                  num_workers=0, n_epochs=n_epochs, print_every=5,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                  validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=dev_batch_size)

trainer.train(load_best_model=False)
