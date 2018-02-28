import re
import sys

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.insert(0, '../../')
from segmentation.code import tools, splitters, representations


punctuation_pat = re.compile(r"""([!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~])""")
hyphenline_pat = re.compile(r"-\s*\n\s*")
multiwhite_pat = re.compile(r"\s+")
cid_pat = re.compile(r"\(cid:\d+\)")
nonlet = re.compile(r"([^A-Za-z0-9 ])")


def clean_text(txt):
    # txt = txt.decode("utf-8")

    txt = txt.lower()
    txt = cid_pat.sub(" UNK ", txt)
    txt = hyphenline_pat.sub("", txt)
    # print punctuation_pat.findall(txt)
    txt = punctuation_pat.sub(r" \1 ", txt)
    txt = re.sub("\n"," NL ", txt)
    txt = nonlet.sub(r" \1 ", txt)

    # txt = punctuation_pat.sub(r"", txt)
    # txt = nonlet.sub(r"", txt)

    txt = multiwhite_pat.sub(" ", txt)
    # txt = txt.encode('utf-8')
    return ''.join(['start ', txt.strip(), ' end'])


def segment_text(txt, fasttext_model, num_segments=10):
    txt = clean_text(txt).split()

    # print("article length:", len(txt))

    X = []

    mapper = {}
    count = 0
    for i, word in enumerate(txt):
        if word in fasttext_model:
            X.append(fasttext_model[word])
            mapper[i] = count
            count += 1

    mapperr = { v:k for k,v in mapper.items() }

    X = np.array(X)
    # print("X length:", X.shape)

    sig = splitters.gensig_model(X)
    # print "Splitting..."
    splits, e = splitters.greedysplit(X.shape[0], num_segments, sig)
    # print splits
    # print "Refining..."
    splits = splitters.refine(splits, sig, 20)
    # print splits

    # print "Printing refined splits... "

    # for i,s in enumerate(splits[:-1]):
    #     k = mapperr[s]
    #     print
    #     print i,s
    #     print " ".join(txt[k-100:k]), "\n\n", " ".join(txt[k:k+100])

    # with open("result{}.txt".format(K),"w") as f:
    #     prev = 0
    #     for s in splits:
    #         k = mapperr.get(s,len(txt))
    #         f.write(" ".join(txt[prev:k]).replace("NL","\n"))
    #         f.write("\nBREAK\n")
    #         prev = k

    segments = []
    prev = 0
    for s in splits:
        # get the mapping for this indice
        k = mapperr.get(s, len(txt))
        segment_text = ' '.join(txt[prev:k]).replace('NL', '\n')
        prev = k
        segments.append(segment_text)

    return segments

