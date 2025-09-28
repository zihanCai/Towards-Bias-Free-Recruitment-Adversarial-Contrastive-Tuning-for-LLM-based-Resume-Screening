''' Main script for loading BERT models and running WEAT tests '''

import os
import sys
import random
import re
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)  # noqa

from csv import DictWriter
from enum import Enum

import numpy as np
from data import load_json, load_encodings, save_encodings
import weat as weat
from transformers import BertModel,BertTokenizer
import torch
# import encoders.bert as bert

def load_model(version='bert-large-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = BertTokenizer.from_pretrained('D:/wfy/code/model/bert_model')
    # model = BertModel.from_pretrained('D:/wfy/code/LLM-fairness/save_model/models/'+version)
    model = BertModel.from_pretrained('D:/wfy/code/BERT-ASE-main/model_save/bert_debias_augmented/epoch_50')
    model.eval()

    return model, tokenizer


def encode(model, tokenizer, texts):
    ''' Use tokenizer and model to encode texts '''
    encs = {}
    for text in texts:
        # tokenized = tokenizer.tokenize(text)
        # indexed = tokenizer.convert_tokens_to_ids(tokenized)
        # segment_idxs = [0] * len(tokenized)
        # tokens_tensor = torch.tensor([indexed])
        # segments_tensor = torch.tensor([segment_idxs])
        # enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)
        # enc = enc[:, 0, :]  # extract the last rep of the first input
        # encs[text] = enc.detach().view(-1).numpy()

        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        cls_embedding = output.last_hidden_state[:, 0, :]
        encs[text] = cls_embedding.detach().view(-1).numpy()

    return encs




class ModelName(Enum):
    BERT = 'bert'

TEST_EXT = '.jsonl'
BERT_VERSIONS = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased"]

def test_sort_key(test):
    ''' Sort key for test names '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)
    return key

def handle_arguments(arguments):
    ''' Parse arguments '''
    parser = argparse.ArgumentParser(
        description='Run specified WEAT tests on BERT models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tests', '-t', type=str,
                        help=f"WEAT tests to run (comma-separated list; test files should be in `data_dir` "
                             f"and have corresponding names, with extension {TEST_EXT}). Default: all tests.")
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str,
                        help="File to log to")
    parser.add_argument('--results_path', type=str,
                        help="Path where TSV results file will be written",
                        default='D:/wfy/code/LLM-fairness/eval/seat/results/')
    parser.add_argument('--ignore_cached_encs', '-i', action='store_true',
                        help="If set, ignore existing encodings and encode from scratch.")
    parser.add_argument('--dont_cache_encs', action='store_true',
                        help="If set, don't cache encodings to disk.")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test",
                        default='D:/wfy/code/LLM-fairness/eval/tests')
    parser.add_argument('--exp_dir', type=str,
                        help="Directory to load and save vectors (stored as h5py files).",
                        default='output')
    parser.add_argument('--n_samples', type=int,
                        help="Number of permutation test samples for p-values. "
                             "(Exact test is used if fewer permutations exist).",
                        default=100000)
    parser.add_argument('--parametric', action='store_true',
                        help='Use parametric test (normal assumption) for p-values.')
    parser.add_argument('--bert_version', type=str, choices=BERT_VERSIONS,
                        help="Version of BERT to use.", default="epoch_50")
    return parser.parse_args(arguments)

def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Validate comma-separated items '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError(f"Unknown {item_type}: {item}!")
    return items

def maybe_make_dir(dirname):
    ''' Create directory if it does not exist '''
    os.makedirs(dirname, exist_ok=True)

def main(arguments):
    ''' Main logic for running WEAT tests on BERT '''
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    args = handle_arguments(arguments)
    if args.seed >= 0:
        log.info(f'Seeding random number generators with {args.seed}')
        random.seed(args.seed)
        np.random.seed(args.seed)
    maybe_make_dir(args.exp_dir)
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    all_tests = sorted(
        [
            entry[:-len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        key=test_sort_key
    )
    tests = split_comma_and_check(args.tests, all_tests, "test") if args.tests is not None else all_tests
    log.info('Tests selected: %s', tests)

    results = []
    for test in tests:
        log.info('Running test %s for BERT model', test)
        enc_file = os.path.join(args.exp_dir, f"bert-{args.bert_version}.{test}.h5")
        if not args.ignore_cached_encs and os.path.isfile(enc_file):
            log.info("Loading encodings from %s", enc_file)
            encs = load_encodings(enc_file)
        else:
            encs = load_json(os.path.join(args.data_dir, f"{test}{TEST_EXT}"))

            # Load BERT and encode sentences
            model, tokenizer = load_model(args.bert_version)
            log.info("Encoding sentences...")
            encs["targ1"]["encs"] = encode(model, tokenizer, encs["targ1"]["examples"])
            encs["targ2"]["encs"] = encode(model, tokenizer, encs["targ2"]["examples"])
            encs["attr1"]["encs"] = encode(model, tokenizer, encs["attr1"]["examples"])
            encs["attr2"]["encs"] = encode(model, tokenizer, encs["attr2"]["examples"])

            if not args.dont_cache_encs:
                log.info("Saving encodings to %s", enc_file)
                save_encodings(encs, enc_file)

        # Run the WEAT test
        log.info("Running WEAT...")
        esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
        results.append(dict(
            model="bert",
            options=f"version={args.bert_version}",
            test=test,
            p_value=pval,
            effect_size=esize,
            num_targ1=len(encs['targ1']['encs']),
            num_targ2=len(encs['targ2']['encs']),
            num_attr1=len(encs['attr1']['encs']),
            num_attr2=len(encs['attr2']['encs'])))

    for r in results:
        log.info("\tTest {test}:\tp-val: {p_value:.9f}\tesize: {effect_size:.2f}".format(**r))

    if args.results_path is not None:
        log.info('Writing results to %s', args.results_path)
        results_path = args.results_path+args.bert_version+'.tsv'
        with open(results_path, 'w') as f:
            writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

if __name__ == "__main__":
    main(sys.argv[1:])
