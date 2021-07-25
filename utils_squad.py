# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open

from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def is_english_or_number(c):
        return (ord(c) > 64 and ord(c) < 91) or (ord(c) < 123 and ord(c) > 96)

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            # print("paragraph_text=",paragraph_text)
            for c in paragraph_text:
                if is_whitespace(c):
                    continue
                char_to_word_offset.append(len(doc_tokens) - 1)
                doc_tokens.append(c)
            
            # print("paragraph[qas]=",paragraph["qas"])
            
            for qa in paragraph["qas"]:
                print("==============")
                print("paragraph[qas]")
                print(paragraph["qas"])
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    answer = qa["answers"][0]
                    print("answer=",answer)
                    orig_answer_text = answer["text"]
                    print("orig_answer_text=",orig_answer_text)
                    
                    answer_offset = answer["answer_start"]
                    print("answer_offset=",answer_offset)
                    
                    answer_length = len(orig_answer_text)
                    
                    print("answer_length=",answer_length)

                    if answer_offset > len(char_to_word_offset) - 1:
                        logger.warning("样本错误: '%s'  offfset vs. length'%s'",
                                       answer_offset, len(char_to_word_offset))
                        continue
                    start_position = char_to_word_offset[answer_offset]
                    end_position = answer_offset + answer_length - 1
                    if end_position > len(char_to_word_offset) - 1:
                        logger.warning("样本错误: '%s' vs. '%s'", end_position, len(char_to_word_offset))
                        continue
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = "".join(
                        doc_tokens[start_position:(end_position + 1)])
                    print("actual_text=",actual_text)
                    cleaned_answer_text = "".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("样本错误: '%s' vs. '%s'", actual_text,
                                       cleaned_answer_text)
                        continue


                    print("qas_id=",qas_id)
                    print("question_text=",question_text)
                    print("doc_tokens=",doc_tokens)
                    print("orig_answer_text=",orig_answer_text)
                    print("start_position=",start_position)
                    print("end_position=",end_position)
                    print("is_impossible=",is_impossible)

                    # time.sleep(60000)
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    # print("examples=",examples)
    return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)
    # examples=examples[0]
    features = []
    for (example_index, example) in enumerate(examples):
        if example_index==1:
            break
        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)
        print("example=",example)
        # qas_id: TRAIN_54_QUERY_3, question_text: 毕业后的安雅·罗素法职业是什么？, 
        # doc_tokens: [安 雅 · 罗 素 法 （ ，  ） ， 来 自 俄 罗 斯 圣 彼 得 堡 的 模 特 儿 。 她 是 《 全 美 超 级 模 特 儿 新 秀 大 赛 》 第 十 季 的 亚 军 。 2 0 0 8 年 ， 安 雅 宣 布 改 回 出 生 时 的 名 字 ： 安 雅 · 罗 素 法 （ A n y a R o z o v a ） ， 在 此 之 前 是 使 用 安 雅 · 冈 （ ） 。 安 雅 于 俄 罗 斯 出 生 ， 后 来 被 一 个 居 住 在 美 国 夏 威 夷 群 岛 欧 胡 岛 檀 香 山 的 家 庭 领 养  。 安 雅 十 七 岁 时 曾 参 与 香 奈 儿 、 路 易 · 威 登 及 芬 迪 （ F e n d i ） 等 品 牌 的 非 正 式 时 装 秀 。 2 0 0 7 年 ， 她 于 瓦 伊 帕 胡 高 级 中 学 毕 业 。 毕 业 后 ， 她 当 了 一 名 售 货 员 。 她 曾 为 R u s s e l l T a n o u e 拍 摄 照 片 ， R u s s e l l T a n o u e 称 赞 她 是 「 有 前 途 的 新 面 孔 」 。 安 雅 在 半 准 决 赛 面 试 时 说 她 对 模 特 儿 行 业 充 满 热 诚 ， 所 以 参 加 全 美 超 级 模 特 儿 新 秀 大 赛 。 她 于 比 赛 中 表 现 出 色 ， 曾 五 次 首 名 入 围 ， 平 均 入 围 顺 序 更 拿 下 历 届 以 来 最 优 异 的 成 绩 ( 2 . 6 4 ) ， 另 外 胜 出 三 次 小 挑 战 ， 分 别 获 得 与 评 判 尼 祖 · 百 克 拍 照 、 为 柠 檬 味 道 的 七 喜 拍 摄 广 告 的 机 会 及 十 万 美 元 、 和 盖 马 蒂 洛  （ G a i M a t t i o l o ） 设 计 的 晚 装 。 在 最 后 两 强 中 ， 安 雅 与 另 一 名 参 赛 者 惠 妮 · 汤 姆 森 为 范 思 哲 走 秀 ， 但 评 判 认 为 她 在 台 上 不 够 惠 妮 突 出 ， 所 以 选 了 惠 妮 当 冠 军 ， 安 雅 屈 居 亚 军 ( 但 就 整  体 表 现 来 说 ， 部 份 网 友 认 为 安 雅 才 是 第 十 季 名 副 其 实 的 冠 军 。 ) 安 雅 在 比 赛 拿 五 次 第 一 ， 也  胜 出 多 次 小 挑 战 。 安 雅 赛 后 再 次 与 R u s s e l l T a n o u e 合 作 ， 为 2 0 0 8 年 4 月 3 0 日 出 版 的 M i d W e e k 杂 志 拍 摄 封 面 及 内 页 照 。 其 后 她 参 加 了 V 杂 志 与 S u p r e m e 模 特 儿 公 司 合 办 的 模 特 儿 选 拔 赛 2 0 0 8 。 她 其 后 更 与 E l i t e 签 约 。 最 近 她 与 香 港 的 模 特 儿 公 司 S t y l e I n t e r n a t i o n a l M a n a g e m e n t 签 约 ， 并 在 香 港 发 展 其 模 特 儿 事 业 。 她 曾 在 很 多 香 港 的 时 装 杂 志 中 任 模 特 儿 ， 
        # 《 J e t 》 、 《 东 方 日 报 》 、 《 E l l e 》 等 。], start_position: 201, end_position: 203
        print("example.question_text=",example.question_text)

        query_tokens = tokenizer.tokenize(example.question_text)
        print("query_tokens=",query_tokens)
        print("len(query_tokens)=",len(query_tokens),"max_query_length=",max_query_length)
        # query_tokens= ['毕', '业', '后', '的', '安', '雅', '·', '罗', '素', '法', '职', '业', '是', '什', '么', '？']
        # len(query_tokens)= 16 max_query_length= 64
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        # print("tok_to_orig_index=",tok_to_orig_index)
        # tok_to_orig_index= [0, 1, 2, 3 .., 715, 716, 717]
        # print("orig_to_tok_index=",orig_to_tok_index)
        # orig_to_tok_index= [0, 1, 2, ... 715, 716, 717]
        # print("all_doc_tokens=",all_doc_tokens)
        # all_doc_tokens= ['安', '雅', '·', '罗', '素', '法'... [UNK]', 'l', 'l', 'e', '》', '等', '。']
        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            # print("example.start_position=",example.start_position)
            # example.start_position= 201
            tok_start_position = orig_to_tok_index[example.start_position]
            # print("tok_start_position=",tok_start_position)
            # tok_start_position= 201
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                tokenizer, example.orig_answer_text)
            # print("tok_start_position",tok_start_position)
            # print("tok_end_position",tok_end_position)
            # tok_start_position 201
            # tok_end_position 203
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3 #減問題
        # print("max_seq_length=",max_seq_length,"max_tokens_for_doc",max_tokens_for_doc)
        # max_seq_length= 384 max_tokens_for_doc 365
        

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):#718 移動窗格可以容納的長度
            print("len(all_doc_tokens)=",len(all_doc_tokens),"start_offset=",start_offset)
            length = len(all_doc_tokens) - start_offset#剩下的長度
            print("length=",length,"max_tokens_for_doc",max_tokens_for_doc)
            if length > max_tokens_for_doc:#剩下的長度>文本長度(扣問題)
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))#目前位置, 到文本最後來有幾個字
            print("_DocSpan(start=start_offset, length=length)=")
            print(_DocSpan(start=start_offset, length=length))
            
            if start_offset + length == len(all_doc_tokens):#滑完了
                break
            start_offset += min(length, doc_stride)
            print("min(length, doc_stride)=",min(length, doc_stride))#滑動窗格

        for (doc_span_index, doc_span) in enumerate(doc_spans):#全部滑動窗格
            print("===============================",doc_span_index,"===============================")
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []
            # print("doc_span_index",doc_span_index,"doc_span",doc_span)
            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)
            # print("tokens a",tokens)
            # print("segment_ids a",segment_ids)
            # print("p_mask a",p_mask)

            # Paragraph
            # print("doc_span.length=",doc_span.length)#365
            for i in range(doc_span.length):#該滑動窗格的所有字
                split_token_index = doc_span.start + i
                # print("i=",i,"doc_span.start =",doc_span.start ,"split_token_index=",split_token_index)
                # i= 0 doc_span.start = 0 split_token_index= 0
                
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                # print("token_to_orig_map[",len(tokens),"]=",tok_to_orig_index[split_token_index])
                # token_to_orig_map[ 18 ]= 0....

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context#該字在該滑動窗格,相較於在其他滑動窗格, 是否有最高的分數
                # print("token_is_max_context[",len(tokens),"]=",is_max_context)#(包含問題)紀錄是否為最高分數
                tokens.append(all_doc_tokens[split_token_index])
                # print("all_doc_tokens[",split_token_index,"]=",all_doc_tokens[split_token_index])#all_doc_tokens= ['安', '雅', '·', .... '柠', '檬', '味']
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)#cls+問題a+sep+滑動窗格所有字(365)b+sep
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            # print("tokens b",tokens)
            # print("segment_ids b",segment_ids)
            # print("p_mask b",p_mask)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            # print("len(input_ids)=",len(input_ids),"max_seq_length",max_seq_length)
            # len(input_ids)= 353 max_seq_length 384
            while len(input_ids) < max_seq_length:#384
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            print("input_ids=",input_ids)
            print("input_mask=",input_mask)
            print("segment_ids=",segment_ids)

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                # print("is_training and not span_is_impossible:")
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                # print("out_of_span=",out_of_span)
                if out_of_span:#若答案不在这个文本范围内，则答案开始和结束位置设为0
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2#問題的長度
                    # print("tok_start_position",tok_start_position,"doc_start",doc_start,"doc_offset=",doc_offset)
                    # tok_start_position 201 doc_start 0 doc_offset= 18
                    start_position = tok_start_position - doc_start + doc_offset #加上問題長度後 答案的開始位置
                    # print("tok_end_position",tok_end_position,"doc_start",doc_start,"doc_offset=",doc_offset)
                    # tok_end_position 203 doc_start 0 doc_offset= 18

                    end_position = tok_end_position - doc_start + doc_offset #加上問題長度後 答案的結束位置

            if is_training and span_is_impossible:
                # print("is_training and  span_is_impossible:")
                start_position = cls_index
                end_position = cls_index

            print("start_position=",start_position)
            print("end_position=",end_position)
            # start_position= 219
            # end_position= 221


            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in token_is_max_context.items()
                ]))
                logger.info(
                    "input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x)
                                                  for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   *** Example ***
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   unique_id: 1000000000
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   example_index: 0
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   doc_span_index: 0
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   tokens: [CLS] 毕 业 后 的 安 雅 · 罗 素 法 职 业 是 什 么 ？ [SEP] 安 雅 · 罗 素 法 （ ， ） ， 来 自 俄 罗 斯 圣 彼 得 堡 的 模 特 儿 。 她 是 《 全 美 超 级 模 特 儿 新 秀 大 赛 》 第 十 季 的 亚 军 。 2 0 0 8 年 ， 安 雅 宣 布 改 回 出 生 时 的 名 字 ： 安 雅 · 罗 素 法 （ [UNK] n y a [UNK] o z o v a ） ， 在  此 之 前 是 使 用 安 雅 · 冈 （ ） 。 安 雅 于 俄 罗 斯 出 生 ， 后 来 被 一 个 居 住 在 美 国 夏 威 夷 群 岛 欧 胡 岛  檀 香 山 的 家 庭 领 养 。 安 雅 十 七 岁 时 曾 参 与 香 奈 儿 、 路 易 · 威 登 及 芬 迪 （ [UNK] e n d i ） 等 品 牌 的 非 正 式 时 装 秀 。 2 0 0 7 年 ， 她 于 瓦 伊 帕 胡 高 级 中 学 毕 业 。 毕 业 后 ， 她 当 了 一 名 售 货 员 。 她 曾 为 [UNK] u s s e l l [UNK] a n o u e 拍 摄 照 片 ， [UNK] u s s e l l [UNK] a n o u e 称 赞 她 是 「 有 前 途 的 新 面  孔 」 。 安 雅 在 半 准 决 赛 面 试 时 说 她 对 模 特 儿 行 业 充 满 热 诚 ， 所 以 参 加 全 美 超 级 模 特 儿 新 秀 大 赛 。 她 于 比 赛 中 表 现 出 色 ， 曾 五 次 首 名 入 围 ， 平 均 入 围 顺 序 更 拿 下 历 届 以 来 最 优 异 的 成 绩 ( 2 . 6 4 ) ， 另 外 胜 出 三 次 小 挑 战 ， 分 别 获 得 与 评 判 尼 祖 · 百 克 拍 照 、 为 柠 檬 味 [SEP]
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   token_to_orig_map: 18:0 19:1 20:2 21:3 22:4 23:5 24:6 25:7 26:8 27:9 28:10 29:11 30:12 31:13 32:14 33:15 34:16 35:17 36:18 37:19 38:20 39:21 40:22 41:23 42:24 43:25 44:26 45:27 46:28 47:29 48:30 49:31 50:32 51:33 52:34 53:35 54:36 55:37 56:38 57:39 58:40 59:41 60:42 61:43 62:44 63:45 64:46 65:47 66:48 67:49 68:50 69:51 70:52 71:53 72:54 73:55 74:56 75:57 76:58 77:59 78:60 79:61 80:62 81:63 82:64 83:65 84:66 85:67 86:68 87:69 88:70 89:71 90:72 91:73 92:74 93:75 94:76 95:77 96:78 97:79 98:80 99:81 100:82 101:83 102:84 103:85 104:86 105:87 106:88 107:89 108:90 109:91 110:92 111:93 112:94 113:95 114:96 115:97 116:98 117:99 118:100 119:101 120:102 121:103 122:104 123:105 124:106 125:107 126:108 127:109 128:110 129:111 130:112 131:113 132:114 133:115 134:116 135:117 136:118 137:119 138:120 139:121 140:122 141:123 142:124 143:125 144:126 145:127 146:128 147:129 148:130 149:131 150:132 151:133 152:134 153:135 154:136 155:137 156:138 157:139 158:140 159:141 160:142 161:143 162:144 163:145 164:146 165:147 166:148 167:149 168:150 169:151 170:152 171:153 172:154 173:155 174:156 175:157 176:158 177:159 178:160 179:161 180:162 181:163 182:164 183:165 184:166 185:167 186:168 187:169 188:170 189:171 190:172 191:173 192:174 193:175 194:176 195:177 196:178 197:179 198:180 199:181 200:182 201:183 202:184 203:185 204:186 205:187 206:188 207:189 208:190 209:191 210:192 211:193 212:194 213:195 214:196 215:197 216:198 217:199 218:200 219:201 220:202 221:203 222:204 223:205 224:206 225:207 226:208 227:209 228:210 229:211 230:212 231:213 232:214 233:215 234:216 235:217 236:218 237:219 238:220 239:221 240:222 241:223 242:224 243:225 244:226 245:227 246:228 247:229 248:230 249:231 250:232 251:233 252:234 253:235 254:236 255:237 256:238 257:239 258:240 259:241 260:242 261:243 262:244 263:245 264:246 265:247 266:248 267:249 268:250 269:251 270:252 271:253 272:254 273:255 274:256 275:257 276:258 277:259 278:260 279:261 280:262 281:263 282:264 283:265 284:266 285:267 286:268 287:269 288:270 289:271 290:272 291:273 292:274 293:275 294:276 295:277 296:278 297:279 298:280 299:281 300:282 301:283 302:284 303:285 304:286 305:287 306:288 307:289 308:290 309:291 310:292 311:293 312:294 313:295 314:296 315:297 316:298 317:299 318:300 319:301 320:302 321:303 322:304 323:305 324:306 325:307 326:308 327:309 328:310 329:311 330:312 331:313 332:314 333:315 334:316 335:317 336:318 337:319 338:320 339:321 340:322 341:323 342:324 343:325 344:326 345:327 346:328 347:329 348:330 349:331 350:332 351:333 352:334 353:335 354:336 355:337 356:338 357:339 358:340 359:341 360:342 361:343 362:344 363:345 364:346 365:347 366:348 367:349 368:350 369:351 370:352 371:353 372:354 373:355 374:356 375:357 376:358 377:359 378:360 379:361 380:362 381:363 382:364                    # 07/24/2021 14:31:40 - INFO - utils_squad -   token_is_max_context: 18:True 19:True 20:True 21:True 22:True 23:True 24:True 25:True 26:True 27:True 28:True 29:True 30:True 31:True 32:True 33:True 34:True 35:True 36:True 37:True 38:True 39:True 40:True 41:True 42:True 43:True 44:True 45:True 46:True 47:True 48:True 49:True 50:True 51:True 52:True 53:True 54:True 55:True 56:True 57:True 58:True 59:True 60:True 61:True 62:True 63:True 64:True 65:True 66:True 67:True 68:True 69:True 70:True 71:True 72:True 73:True 74:True 75:True 76:True 77:True 78:True 79:True 80:True 81:True 82:True 83:True 84:True 85:True 86:True 87:True 88:True 89:True 90:True 91:True 92:True 93:True 94:True 95:True 96:True 97:True 98:True 99:True 100:True 101:True 102:True 103:True 104:True 105:True 106:True 107:True 108:True 109:True 110:True 111:True 112:True 113:True 114:True 115:True 116:True 117:True 118:True 119:True 120:True 121:True 122:True 123:True 124:True 125:True 126:True 127:True 128:True 129:True 130:True 131:True 132:True 133:True 134:True 135:True 136:True 137:True 138:True 139:True 140:True 141:True 142:True 143:True 144:True 145:True 146:True 147:True 148:True 149:True 150:True 151:True 152:True 153:True 154:True 155:True 156:True 157:True 158:True 159:True 160:True 161:True 162:True 163:True 164:True 165:True 166:True 167:True 168:True 169:True 170:True 171:True 172:True 173:True 174:True 175:True 176:True 177:True 178:True 179:True 180:True 181:True 182:True 183:True 184:True 185:True 186:True 187:True 188:True 189:True 190:True 191:True 192:True 193:True 194:True 195:True 196:True 197:True 198:True 199:True 200:True 201:True 202:True 203:True 204:True 205:True 206:True 207:True 208:True 209:True 210:True 211:True 212:True 213:True 214:True 215:True 216:True 217:True 218:True 219:True 220:True 221:True 222:True 223:True 224:True 225:True 226:True 227:True 228:True 229:True 230:True 231:True 232:True 233:True 234:True 235:True 236:True 237:True 238:True 239:True 240:True 241:True 242:True 243:True 244:True 245:True 246:True 247:True 248:True 249:True 250:True 251:True 252:True 253:True 254:True 255:True 256:True 257:True 258:True 259:True 260:True 261:True 262:True 263:True 264:True 265:False 266:False 267:False 268:False 269:False 270:False 271:False 272:False 273:False 274:False 275:False 276:False 277:False 278:False 279:False 280:False 281:False 282:False 283:False 284:False 285:False 286:False 287:False 288:False 289:False 290:False 291:False 292:False 293:False 294:False 295:False 296:False 297:False 298:False 299:False 300:False 301:False 302:False 303:False 304:False 305:False 306:False 307:False 308:False 309:False 310:False 311:False 312:False 313:False 314:False 315:False 316:False 317:False 318:False 319:False 320:False 321:False 322:False 323:False 324:False 325:False 326:False 327:False 328:False 329:False 330:False 331:False 332:False 333:False 334:False 335:False 336:False 337:False 338:False 339:False 340:False 341:False 342:False 343:False 344:False 345:False 346:False 347:False 348:False 349:False 350:False 351:False 352:False 353:False 354:False 355:False 356:False 357:False 358:False 359:False 360:False 361:False 362:False 363:False 364:False 365:False 366:False 367:False 368:False 369:False 370:False 371:False 372:False 373:False 374:False 375:False 376:False 377:False 378:False 379:False 380:False 381:False 382:False
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   input_ids: 101 3684 689 1400 4638 2128 7414 185 5384 5162 3791 5466 689 3221 784 720 8043 102 2128 7414 185 5384 5162 3791 8020 8024 8021 8024 3341 5632 915 5384 3172 1760 2516 2533 1836 4638 3563 4294 1036 511 1961 3221 517 1059 5401 6631 5277 3563 4294 1036 3173 4899 1920 6612 518 5018 1282 2108 4638 762 1092 511 123 121 121 129 2399 8024 2128 7414 2146 2357 3121 1726 1139 4495 3198 4638 1399 2099 8038 2128 7414 185 5384 5162 3791 8020 100 156 167 143 100 157 168 157 164 143 8021 8024 1762 3634 722 1184 3221 886 4500 2128 7414 185 1082 8020 8021 511 2128 7414 754 915 5384 3172 1139 4495 8024 1400 3341 6158 671 702 2233 857 1762 5401 1744 1909 2014 1929 5408 2270 3616 5529 2270 3589 7676 2255 4638 2157 2431 7566 1075 511 2128 7414 1282 673 2259 3198 3295 1346 680 7676 1937 1036 510 6662 3211 185 2014 4633 1350 5705 6832 8020 100 147 156 146 151 8021 5023 1501 4277 4638 7478 3633 2466 3198 6163 4899 511 123 121 121 128 2399 8024 1961 754 4482 823 2364 5529 7770 5277 704 2110 3684 689 511 3684 689 1400 8024 1961 2496 749 671 1399 1545 6573 1447 511 1961 3295 711 100 163 161 161 147 154 154 100 143 156 157 163 147 2864 3029 4212 4275 8024 100 163 161 161 147 154 154 100 143 156 157 163 147 4917 6614 1961 3221 519 3300 1184 6854 4638 3173 7481 2096 520 511 2128 7414 1762 1288 1114 1104 6612 7481 6407 3198 6432 1961 2190 3563 4294 1036 6121 689 1041 4007 4178 6411 8024 2792 809 1346 1217 1059 5401 6631 5277 3563 4294 1036 3173 4899 1920 6612 511 1961 754 3683 6612 704 6134 4385 1139 5682 8024 3295 758 3613 7674 1399 1057 1741 8024 2398 1772 1057 1741 7556 2415 3291 2897 678 1325 2237 809 3341 3297 831 2460 4638 2768 5327 113 123 119 127 125 114 8024 1369 1912 5526 1139 676 3613 2207 2904 2773 8024 1146 1166 5815 2533 680 6397 1161 2225 4862 185 4636 1046 2864 4212 510 711 3387 3597 1456 102
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   start_position: 219
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   end_position: 221
                    # 07/24/2021 14:31:40 - INFO - utils_squad -   answer: 售 货 员





                    # 07/24/2021 16:23:34 - INFO - utils_squad -   *** Example ***
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   unique_id: 1000000003
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   example_index: 0
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   doc_span_index: 3
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   tokens: [CLS] 毕 业 后 的 安 雅 · 罗 素 法 职 业 是 什 么 ？ [SEP] 马 蒂 洛 （ [UNK] a i [UNK] a t t i o l o ） 设 计 的 晚 装 。 在 最 后 两 强 中 ， 安 雅 与 另 一 名 参 赛 者 惠 妮 · 汤 姆 森 为 范 思 哲 走 秀 ， 但 评 判 认 为 她 在 台 上 不 够 惠 妮 突 出 ， 所 以 选 了 惠 妮 当 冠 军 ， 安 雅 屈 居 亚 军 (  但 就 整 体 表 现 来 说 ， 部 份 网 友 认 为 安 雅 才 是 第 十 季 名 副 其 实 的 冠 军 。 ) 安 雅 在 比 赛 拿 五 次 第  一 ， 也 胜 出 多 次 小 挑 战 。 安 雅 赛 后 再 次 与 [UNK] u s s e l l [UNK] a n o u e 合 作 ， 为 2 0 0 8 年 4 月 3 0 日 出 版 的 [UNK] i d [UNK] e e k 杂 志 拍 摄 封 面 及 内 页 照 。 其 后 她 参 加 了 [UNK] 杂 志 与 [UNK] u p r e m e 模 特 儿 公 司 合 办 的 模 特 儿 选 拔 赛 2 0 0 8 。 她 其 后 更 与 [UNK] l i t e 签 约 。 最 近 她 与 香 港 的 模 特 儿  公 司 [UNK] t y l e [UNK] n t e r n a t i o n a l [UNK] a n a g e m e n t 签 约 ， 并 在 香 港 发 展 其 模 特 儿 事 业  。 她 曾 在 很 多 香 港 的 时 装 杂 志 中 任 模 特 儿 ， 《 [UNK] e t 》 、 《 东 方 日 报 》 、 《 [UNK] l l e 》 等 。 [SEP]
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   token_to_orig_map: 18:384 19:385 20:386 21:387 22:388 23:389 24:390 25:391 26:392 27:393 28:394 29:395 30:396 31:397 32:398 33:399 34:400 35:401 36:402 37:403 38:404 39:405 40:406 41:407 42:408 43:409 44:410 45:411 46:412 47:413 48:414 49:415 50:416 51:417 52:418 53:419 54:420 55:421 56:422 57:423 58:424 59:425 60:426 61:427 62:428 63:429 64:430 65:431 66:432 67:433 68:434 69:435 70:436 71:437 72:438 73:439 74:440 75:441 76:442 77:443 78:444 79:445 80:446 81:447 82:448 83:449 84:450 85:451 86:452 87:453 88:454 89:455 90:456 91:457 92:458 93:459 94:460 95:461 96:462 97:463 98:464 99:465 100:466 101:467 102:468 103:469 104:470 105:471 106:472 107:473 108:474 109:475 110:476 111:477 112:478 113:479 114:480 115:481 116:482 117:483 118:484 119:485 120:486 121:487 122:488 123:489 124:490 125:491 126:492 127:493 128:494 129:495 130:496 131:497 132:498 133:499 134:500 135:501 136:502 137:503 138:504 139:505 140:506 141:507 142:508 143:509 144:510 145:511 146:512 147:513 148:514 149:515 150:516 151:517 152:518 153:519 154:520 155:521 156:522 157:523 158:524 159:525 160:526 161:527 162:528 163:529 164:530 165:531 166:532 167:533 168:534 169:535 170:536 171:537 172:538 173:539 174:540 175:541 176:542 177:543 178:544 179:545 180:546 181:547 182:548 183:549 184:550 185:551 186:552 187:553 188:554 189:555 190:556 191:557 192:558 193:559 194:560 195:561 196:562 197:563 198:564 199:565 200:566 201:567 202:568 203:569 204:570 205:571 206:572 207:573 208:574 209:575 210:576 211:577 212:578 213:579 214:580 215:581 216:582 217:583 218:584 219:585 220:586 221:587 222:588 223:589 224:590 225:591 226:592 227:593 228:594 229:595 230:596 231:597 232:598 233:599 234:600 235:601 236:602 237:603 238:604 239:605 240:606 241:607 242:608 243:609 244:610 245:611 246:612 247:613 248:614 249:615 250:616 251:617 252:618 253:619 254:620 255:621 256:622 257:623 258:624 259:625 260:626 261:627 262:628 263:629 264:630 265:631 266:632 267:633 268:634 269:635 270:636 271:637 272:638 273:639 274:640 275:641 276:642 277:643 278:644 279:645 280:646 281:647 282:648 283:649 284:650 285:651 286:652 287:653 288:654 289:655 290:656 291:657 292:658 293:659 294:660 295:661 296:662 297:663 298:664 299:665 300:666 301:667 302:668 303:669 304:670 305:671 306:672 307:673 308:674 309:675 310:676 311:677 312:678 313:679 314:680 315:681 316:682 317:683 318:684 319:685 320:686 321:687 322:688 323:689 324:690 325:691 326:692 327:693 328:694 329:695 330:696 331:697 332:698 333:699 334:700 335:701 336:702 337:703 338:704 339:705 340:706 341:707 342:708 343:709 344:710 345:711 346:712 347:713 348:714 349:715 350:716 351:717
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   token_is_max_context: 18:False 19:False 20:False 21:False 22:False 23:False 24:False 25:False 26:False 27:False 28:False 29:False 30:False 31:False 32:False 33:False 34:False 35:False 36:False 37:False 38:False 39:False 40:False 41:False 42:False 43:False 44:False 45:False 46:False 47:False 48:False 49:False 50:False 51:False 52:False 53:False 54:False 55:False 56:False 57:False 58:False 59:False 60:False 61:False 62:False 63:False 64:False 65:False 66:False 67:False 68:False 69:False 70:False 71:False 72:False 73:False 74:False 75:False 76:False 77:False 78:False 79:False 80:False 81:False 82:False 83:False 84:False 85:False 86:False 87:False 88:False 89:False 90:False 91:False 92:False 93:False 94:False 95:False 96:False 97:False 98:False 99:False 100:False 101:False 102:False 103:False 104:False 105:False 106:False 107:False 108:False 109:False 110:False 111:False 112:False 113:False 114:False 115:False 116:False 117:False 118:False 119:False 120:False 121:False 122:False 123:False 124:False 125:False 126:False 127:False 128:False 129:False 130:False 131:False 132:False 133:False 134:False 135:False 136:False 137:True 138:True 139:True 140:True 141:True 142:True 143:True 144:True 145:True 146:True 147:True 148:True 149:True 150:True 151:True 152:True 153:True 154:True 155:True 156:True 157:True 158:True 159:True 160:True 161:True 162:True 163:True 164:True 165:True 166:True 167:True 168:True 169:True 170:True 171:True 172:True 173:True 174:True 175:True 176:True 177:True 178:True 179:True 180:True 181:True 182:True 183:True 184:True 185:True 186:True 187:True 188:True 189:True 190:True 191:True 192:True 193:True 194:True 195:True 196:True 197:True 198:True 199:True 200:True 201:True 202:True 203:True 204:True 205:True 206:True 207:True 208:True 209:True 210:True 211:True 212:True 213:True 214:True 215:True 216:True 217:True 218:True 219:True 220:True 221:True 222:True 223:True 224:True 225:True 226:True 227:True 228:True 229:True 230:True 231:True 232:True 233:True 234:True 235:True 236:True 237:True 238:True 239:True 240:True 241:True 242:True 243:True 244:True 245:True 246:True 247:True 248:True 249:True 250:True 251:True 252:True 253:True 254:True 255:True 256:True 257:True 258:True 259:True 260:True 261:True 262:True 263:True 264:True 265:True 266:True 267:True 268:True 269:True 270:True 271:True 272:True 273:True 274:True 275:True 276:True 277:True 278:True 279:True 280:True 281:True 282:True 283:True 284:True 285:True 286:True 287:True 288:True 289:True 290:True 291:True 292:True 293:True 294:True 295:True 296:True 297:True 298:True 299:True 300:True 301:True 302:True 303:True 304:True 305:True 306:True 307:True 308:True 309:True 310:True 311:True 312:True 313:True 314:True 315:True 316:True 317:True 318:True 319:True 320:True 321:True 322:True 323:True 324:True 325:True 326:True 327:True 328:True 329:True 330:True 331:True 332:True 333:True 334:True 335:True 336:True 337:True 338:True 339:True 340:True 341:True 342:True 343:True 344:True 345:True 346:True 347:True 348:True 349:True 350:True 351:True
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   input_ids: 101 3684 689 1400 4638 2128 7414 185 5384 5162 3791 5466 689 3221 784 720 8043 102 7716 5881 3821 8020 100 143 151 100 143 162 162 151 157 154 157 8021 6392 6369 4638 3241 6163 511 1762 3297 1400 697 2487 704 8024 2128 7414 680 1369 671 1399 1346 6612 5442 2669 1984 185 3739 1990 3481 711 5745 2590 1528 6624 4899 8024 852 6397 1161 6371 711 1961 1762 1378 677 679 1916 2669 1984 4960 1139 8024 2792 809 6848 749 2669 1984 2496 1094 1092 8024 2128 7414 2235 2233 762 1092 113 852 2218 3146 860 6134 4385 3341 6432 8024 6956 819 5381 1351 6371 711 2128 7414 2798 3221 5018 1282 2108 1399 1199 1071 2141 4638 1094 1092 511 114 2128 7414 1762 3683 6612 2897 758 3613 5018 671 8024 738 5526 1139 1914 3613 2207 2904 2773 511 2128 7414 6612 1400 1086 3613 680 100 163 161 161 147 154 154 100 143 156 157 163 147 1394 868 8024 711 123 121 121 129 2399 125 3299 124 121 3189 1139 4276 4638 100 151 146 100 147 147 153 3325 2562 2864 3029 2196 7481 1350 1079 7552 4212 511 1071 1400 1961 1346 1217 749 100 3325 2562 680 100 163 158 160 147 155 147 3563 4294 1036 1062 1385 1394 1215 4638 3563 4294 1036 6848 2869 6612 123 121 121 129 511 1961 1071 1400 3291 680 100 154 151 162 147 5041 5276 511 3297 6818 1961 680 7676 3949 4638 3563 4294 1036 1062 1385 100 162 167 154 147 100 156 162 147 160 156 143 162 151 157 156 143 154 100 143 156 143 149 147 155 147 156 162 5041 5276 8024 2400 1762 7676 3949 1355 2245 1071 3563 4294 1036 752 689 511 1961 3295 1762 2523 1914 7676 3949 4638 3198 6163 3325 2562 704 818 3563 4294 1036 8024 517 100 147 162 518 510 517 691 3175 3189 2845 518 510 517 100 154 154 147 518 5023 511 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                    # 07/24/2021 16:23:34 - INFO - utils_squad -   impossible example
                    # 07/24/2021 16:23:34 - INFO - __main__ -   Saving features into cached file data\cached_train_bert-base-chinese_384
            # time.sleep(60000)
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,#第幾個文本
                    doc_span_index=doc_span_index,#第幾個滑動窗格
                    tokens=tokens,#[CLS]+Q+[SEP]+文本[SEP] 
                    token_to_orig_map=token_to_orig_map,#文本中每個字的編號
                    token_is_max_context=token_is_max_context,#文本中每個字在該滑動窗格中 左右句子是否相關
                    input_ids=input_ids,#[CLS]+Q+[SEP]+文本[SEP]  轉換為id
                    input_mask=input_mask,#[CLS]+Q+[SEP]+文本[SEP] 轉換為1
                    segment_ids=segment_ids,#[CLS]+Q+[SEP] 轉換為0  文本[SEP]轉換為1
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,#答案在[CLS]+Q+[SEP]+文本[SEP] 的開始位置 若沒有答案 則為0
                    end_position=end_position,#答案在[CLS]+Q+[SEP]+文本[SEP] 的結束位置 若沒有答案 則為0
                    is_impossible=span_is_impossible))#這滑動窗格中是否有答案
            unique_id += 1
        # time.sleep(60000)
    return features


# convert_examples_to_features(examp)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    # print("position=",position)
    for (span_index, doc_span) in enumerate(doc_spans):#所有滑動窗格中
        # print("doc_span",doc_span)
        end = doc_span.start + doc_span.length - 1
        # print("doc_span.start=",doc_span.start,"doc_span.length - 1=",doc_span.length - 1)#滑動窗格範圍
        if position < doc_span.start:#只在position出現的滑動窗格中(n個)
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start#position在該滑動窗格範圍左邊有多少字
        # print("num_left_context=",num_left_context)
        num_right_context = end - position#position在該滑動窗格範圍右邊有多少字
        # print("num_right_context=",num_right_context)

        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length#計算左邊與右邊相關性的分數
        if best_score is None or score > best_score:#如果是所有範圍分數最高，best_span_index紀錄哪個範圍
            best_score = score
            best_span_index = span_index
        # print("best_span_index=",best_span_index)
        # print("cur_span_index=",cur_span_index)   

    return cur_span_index == best_span_index#當下那個範圍


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      verbose_logging, version_2_with_negative,
                      null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[
                    0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case,
                                            verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0,
                    _NbestPrediction(
                        text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def write_predictions_extended(
        all_examples, all_features, all_results, n_best_size,
        max_answer_length, output_prediction_file, output_nbest_file,
        output_null_log_odds_file, orig_data_file, start_n_top, end_n_top,
        version_2_with_negative, tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.
        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_log_prob",
            "end_log_prob"
        ])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(
                tok_text, orig_text, tokenizer.do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw,
                            scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs