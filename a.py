import stanza
nlp=stanza.Pipeline('en')
doc = nlp('this is a text')
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.lemma, word.pos)