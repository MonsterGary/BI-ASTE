import spacy
spacy.prefer_gpu(0)
nlp = spacy.load('zh_core_web_sm')
raw_words=['番茄酱','和','配料','很棒','，','但','很贵','。']
from spacy.tokens import Doc
# tokenized = self.tokenizer.tokenize(words)   # 不能对句分词，因为上面是对词分词，两种分词结果不一样


pos=Doc(nlp.vocab,
        # words=tokenizedbpes)
        words=raw_words)
for spacyname, tool in nlp.pipeline:
    tool(pos)
pos_tag = []
dep_tag = []
word_index = []
itrator=1   # 第一个和最后一个为bos，eos
for t in pos:
    # print(t.text, t.i, t.head, t.head.i)
    conpos='<<'+t.pos_+'>>'              #<<SPACE>>变成unk
    dep=t.dep_
    pos_tag.append(conpos)
    dep_tag.append(dep)
    itrator += 1
doc = Doc(nlp.vocab,
          # words=tokenizedbpes)
          words=raw_words)
# Tagger(doc)
for spacyname, tool in nlp.pipeline:
    tool(doc)
head = []
for t in doc:
    # print(t.text, t.i, t.head, t.head.i)
    head.append(t.head.i)
print(head)