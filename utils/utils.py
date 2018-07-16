import re
import unicodedata
import nltk


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_file(file_name):
    '''
        loads the file, reads line by line, trims last character, and returns a list.
    '''
    f = open(file_name).readlines()
    last = f[-1]
    f = [i[:-1] for i in f[:-1]]
    f = f + [last]
    return f

def preProcessing(text_sequence):
	print "the text sequence is ", text_sequence
	_a = nltk.word_tokenize(text_sequence.replace('.','').replace("(","").replace(")","").replace('?',""))
	a = [normalizeString(l.strip()).strip() for l in _a]
	new_a = []
	for i in a:
		if " " in i:
			temp = i.split(" ")
			for h in temp:
				new_a.append(h)
		else:
			new_a.append(i)
	return new_a