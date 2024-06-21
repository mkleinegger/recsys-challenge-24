import tensorflow as tf
import numpy as np

from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import word_tokenize


class EbnerdIterator(MINDIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts.

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.nid2index = {}
        news_title = [""]

        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, title, ab, url = line.strip("\n").split(
                    self.col_spliter
                )
                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                news_title.append(title)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]

    def init_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[
                    -self.his_size :
                ]

                impr_news = [self.nid2index[(i.split("-")[0])] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1