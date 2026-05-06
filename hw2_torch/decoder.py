import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            x = torch.LongTensor(features).unsqueeze(0)
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim = 1)[0]
            outputs = torch.argsort(probs, descending = True)

            for i in outputs:
                
                output_pair = self.output_labels[int(i.item())]
                if len(state.stack) == 0 and output_pair[0] != "shift": continue
                if output_pair[0] == "shift" and len(state.buffer) == 1 and state.stack: continue
                if state.stack and words[state.stack[-1]] == "ROOT" and output_pair[0] == "left-arc": continue
                else: 
                    if output_pair[0] == "shift":  
                        state.shift()
                        break
                    elif output_pair[0] == "left-arc":
                        state.left_arc(output_pair[1])
                        break
                    else:
                        state.right_arc(output_pair[1])
                        break






  

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
