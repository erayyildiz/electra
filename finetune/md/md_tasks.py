from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import re

import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from model import tokenization
from pretrain import pretrain_helpers

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../data/electra_tr_model/'


class MdExample(task.Example):

    analysis_regex = re.compile(r"^([^\+]*)\+(.+)$", re.UNICODE)

    def __init__(self, eid, task_name, words, candidate_analyzes, tokenizer, is_cased=False):
        super(MdExample, self).__init__(task_name)
        self.eid = eid
        self.words = words
        self.candidate_roots = []
        self.candidate_tags = []

        self.word_ids = []
        self.sub_tokens = []
        self.labels = []

        for word_id, (word, word_candidate_analyzes) in enumerate(zip(words, candidate_analyzes)):
            if not is_cased:
                word = word.replace('İ', 'i').replace('I', 'ı').replace('Ç', 'ç').replace('Ş', 'ş').lower()
            sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(word)
            self.sub_tokens += sub_tokens
            self.word_ids += [word_id] * len(sub_tokens)
            self.labels.append(0)
            self.candidate_roots.append(
                [self._get_root_from_analysis(analysis) for analysis in word_candidate_analyzes])
            self.candidate_tags.append(
                [self._get_tag_from_analysis(analysis) for analysis in word_candidate_analyzes])

    @classmethod
    def _get_root_from_analysis(cls, analysis):
        if analysis.startswith("+"):
            return "+"
        else:
            return cls.analysis_regex.sub(r"\1", analysis)

    @classmethod
    def _get_tag_from_analysis(cls, analysis):
        if analysis.startswith("+"):
            return analysis[2:]
        else:
            return cls.analysis_regex.sub(r"\2", analysis)


class MdTask(task.Task):
  """Defines a sequence tagging task (e.g., part-of-speech tagging)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name, tokenizer):
    super(MdTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self.tag2id = None

  def get_examples(self, split):
    sentences = self._get_sentences(split)
    examples = []
    for i, (words, candidate_roots, candidate_tags) in enumerate(sentences):
      examples.append(MdExample(
          i, self.name, words, candidate_roots, candidate_tags, self._tokenizer
      ))
    return examples

  def featurize(self, example: MdExample, is_training, log=False):
    input_ids = []
    tagged_positions = []
    for word_tokens in words_to_tokens:
      if len(words_to_tokens) + len(input_ids) + 1 > self.config.max_seq_length:
        input_ids.append(self._tokenizer.vocab["[SEP]"])
        break
      if "[CLS]" not in word_tokens and "[SEP]" not in word_tokens:
        tagged_positions.append(len(input_ids))
      for token in word_tokens:
        input_ids.append(self._tokenizer.vocab[token])

    pad = lambda x: x + [0] * (self.config.max_seq_length - len(x))
    labels = pad(example.labels[:self.config.max_seq_length])
    labeled_positions = pad(tagged_positions)
    labels_mask = pad([1.0] * len(tagged_positions))
    segment_ids = pad([1] * len(input_ids))
    input_mask = pad([1] * len(input_ids))
    input_ids = pad(input_ids)
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length
    assert len(labels) == self.config.max_seq_length
    assert len(labels_mask) == self.config.max_seq_length

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": example.eid,
        self.name + "_labels": labels,
        self.name + "_labels_mask": labels_mask,
        self.name + "_labeled_positions": labeled_positions
    }

  def get_scorer(self):
    #TODO implement this
    return None

  def get_feature_specs(self):
    return [
        feature_spec.FeatureSpec(self.name + "_eid", []),
        feature_spec.FeatureSpec(self.name + "_labels",
                                 [self.config.max_seq_length]),
        feature_spec.FeatureSpec(self.name + "_labels_mask",
                                 [self.config.max_seq_length],
                                 is_int_feature=False),
        feature_spec.FeatureSpec(self.name + "_labeled_positions",
                                 [self.config.max_seq_length]),
    ]

  def get_prediction_module(
      self, bert_model, features, is_training, percent_done):

    reprs = bert_model.get_sequence_output()
    reprs = pretrain_helpers.gather_positions(
        reprs, features[self.name + "_labeled_positions"])
    logits = tf.layers.dense(reprs, n_classes)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(features[self.name + "_labels"], n_classes),
        logits=logits)
    losses *= features[self.name + "_labels_mask"]
    losses = tf.reduce_sum(losses, axis=-1)
    return losses, dict(
        loss=losses,
        logits=logits,
        predictions=tf.argmax(logits, axis=-1),
        labels=features[self.name + "_labels"],
        labels_mask=features[self.name + "_labels_mask"],
        eid=features[self.name + "_eid"],
    )

  def _get_sentences(self, split):
      # TODO implement this
      return []


def test_MdExample():
    tokenizer = tokenization.FullTokenizer(
        vocab_file=DATA_DIR + 'vocab.txt',
        do_lower_case=True)

    words = ['Refah', 'da', 'Türkçe', 'için', 'görüş', 'istedi']
    analyzes = [['Refah+Noun+Prop+A3sg+Pnon+Nom', 'refah+Noun+A3sg+Pnon+Nom'], ['da+Conj'],
                ['Türkçe+Noun+Prop+A3sg+Pnon+Nom', 'türkçe+Adj', 'türk+Noun+A3sg+Pnon+Equ', 'türk+Adj^DB+Adverb+Ly',
                 'türk+Adj^DB+Adj+AsIf'],
                ['için+Postp+PCNom', 'iç+Noun+A3sg+P2sg+Nom', 'iç+Noun+A3sg+Pnon+Gen', 'iç+Verb+Pos+Imp+A2pl'],
                ['görüş+Noun+A3sg+Pnon+Nom', 'gör+Verb+Recip+Pos+Imp+A2sg', 'gör+Verb+Pos^DB+Noun+Inf3+A3sg+Pnon+Nom',
                 'görüş+Verb+Pos+Imp+A2sg'], ['iste+Verb+Pos+Past+A3sg']]

    example = MdExample(0, 'md', words, analyzes, tokenizer)
    print('candidate_roots', example.candidate_roots)
    print('candidate_tags', example.candidate_tags)
    print('word_ids', example.word_ids)
    print('sub_tokens', example.sub_tokens)
    print('words', example.words)

if __name__ == '__main__':
    test_MdExample()
