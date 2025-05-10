# from transformers import BartTokenizer, BartModel
#
# # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# # model = BartModel.from_pretrained('facebook/bart-base')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# model = BartModel.from_pretrained('huggingface/model', cache_dir='./huggingface/model', force_download=True,
#                                   resume_download=True)
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)  #decoder_outputs + encoder_outputs
#
# last_hidden_states = outputs.last_hidden_state

list=[('NULL', 'NULL', '<<restaurant#general>>', 'positive')]
string='abc'
a=string+str(list)
print(a)