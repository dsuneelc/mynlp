import itertools as it
import logging
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import TfidfVectorizer

import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .utils.constants import CONTRACTIONS_MAP, UPDATED_STOPWORDS


__all__ = [
    "GetSummary",
    "StopwordRemoval",
    "DropNaRows",
    "Lemmatization",
    "ExpandContractions",
    "Clean",
    "CleanSpaces",
    "NgramPhrases",
    "CCText",
]


# nltk.data.path.append("home/nltk_data/")


def _validate_input(X):
    if isinstance(X, pd.Series):
        X = X.to_frame(name=X.name)
    return X


class GetSummary(BaseEstimator, TransformerMixin):
    """Calculates numeric summary of text columns.

    Summary includes:
        1. Word count
        2. Unique word count
        3. Stopword count
        4. URL count
        5. Mean word lenght
        6. Character count
        7. Punctuation count
        8. Hashtag count

    Parameters
    ----------
    cols :
        Columns of input dataframe to be used for summary.
    stopwords :
        List of stopwords to be used for summary. If `None` default
        list will be used.
    """

    def __init__(self, cols: Optional[List[str]] = None, stopwords: Optional[List[str]] = None):
        self.cols = cols
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        X = _validate_input(X)
        if not self.stopwords:
            self.stopwords = UPDATED_STOPWORDS
        logging.info("Creating summary for the given data.")
        for col in X.columns:
            logging.info("Creating summary for %s." % col)
            X[f"{col}_tmp"] = X[col].apply(lambda x: str(x).split(" "))
            X[f"{col}_tmpl"] = X[col].apply(lambda x: str(x).lower().split(" "))
            logging.debug("Calculating word count.")
            X[f"{col}_word_count"] = X[f"{col}_tmp"].apply(len)
            logging.debug("Calculating unique word count.")
            X[f"{col}_unique_word_count"] = X[f"{col}_tmp"].apply(lambda x: len(set(x)))
            logging.debug("Calculating stopword count.")
            X[f"{col}_stopword_count"] = X[f"{col}_tmp"].apply(
                lambda x: len([w for w in x if w in self.stopwords])
            )
            logging.debug("Calculating URL count.")
            X[f"{col}_url_count"] = X[f"{col}_tmpl"].apply(
                lambda x: len([w for w in x if w in {"http", "https"}])
            )
            logging.debug("Calculating mean word length.")
            X[f"{col}_mean_word_length"] = X[f"{col}_tmp"].apply(
                lambda x: int(np.nanmean([len(w) for w in x]))
            )
            X.drop(columns=[f"{col}_tmp", f"{col}_tmpl"], inplace=True)
            logging.debug("Calculating character count.")
            X[f"{col}_char_count"] = X[col].str.len()
            logging.debug("Calculating punctuation count.")
            X[f"{col}_punctuation_count"] = X[col].apply(
                lambda x: len([c for c in str(x) if c in string.punctuation])
            )
            logging.debug("Calculating %s's hastag count." % col)
            X[f"{col}_hastag_count"] = X[col].apply(lambda x: len([c for c in x if c == "#"]))
            logging.debug("Calculating %s's mention count." % col)
            # X[f"{col}_mention_count"] = X[col].apply(lambda x: len([c for c in x if c == "@"]))
        logging.info("Creating summary for the given data completed.")
        return X


class StopwordRemoval(BaseEstimator, TransformerMixin):
    """Removes stopwords.

    Parameters
    ----------
    column_in : str
        Input column to be used for stopword removal.
    column_out : str
        Ouput column after stopword removal.
    stopwords : Optional[List[str]]
        Stopwords to be used.
    """

    def __init__(self, column_in: str, column_out: str, stopwords: Optional[List[str]] = None):
        self.column_in = column_in
        self.column_out = column_out
        self.stopwords = stopwords if stopwords else UPDATED_STOPWORDS
        # stopwords with pattern, "in-line" will not be removed.
        self._stopword_pattern = re.compile(f"(?<!-)\\b({'|'.join(self.stopwords)})\\b(?!-)")

    def fit(self, X, y=None):
        return self

    def remove_stopwords(self, text: str) -> str:
        """Removes stopwords.

        Parameters
        ----------
        text : str
            Text data.

        Returns
        -------
        str
            Cleaned text.
        """
        if not text:
            return ""
        return re.sub(self._stopword_pattern, "", text)

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Remove stopwords from input column.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Data after stopword removal.
        """
        X = _validate_input(X)
        logging.info("Removing stopwords.")
        X[self.column_out] = X[self.column_in].apply(self.remove_stopwords)
        return X


class DropNaRows(BaseEstimator, TransformerMixin):
    """Removes rows with null values.

    Parameters
    ----------
    column : str
        Input column for NA check.
    id_column : str
        Column with unquie values to be used for dropping.

    Attributes
    ----------
    ids : List[str]
        List of ids with NA's removed.
    """

    def __init__(self, column: str, id_column: str):
        self.column = column
        self.id_column = id_column
        self.ids = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Drop NA rows.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input data.

        Returns
        -------
        pd.DataFrame
            Data with NA rows removed.
        """
        X = _validate_input(X)
        self.ids = X[X[self.column].isna()][self.id_column].tolist()
        logging.info("IDs with NA for %s column: %s" % (self.column, len(self.ids)))
        X = X[~X[self.id_column].isin(self.ids)].copy()
        return X


class Lemmatization(BaseEstimator, TransformerMixin):
    """Lemmatization using WordNetLemmatizer.

    Parameters
    ----------
    column_in : str
        Input column to be lemmatized.
    column_out : str
        Column with lemmatized text.
    """

    def __init__(self, column_in: str, column_out: str):
        self.column_in = column_in
        self.column_out = column_out
        self.__lem = WordNetLemmatizer()
        self.__tagmap = self.__get_wordnet_tags()

    def __get_wordnet_tags(self):
        tagmap = defaultdict(lambda: wordnet.NOUN)
        tagmap["J"] = wordnet.ADJ
        tagmap["V"] = wordnet.VERB
        tagmap["R"] = wordnet.ADV
        return tagmap

    def fit(self, X, y=None):
        return self

    def custom_lemmatizer(self, text: str) -> str:
        """Lemmatize input text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Lemmatized text.
        """
        text = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(text)
        lemmatized_words = [self.__lem.lemmatize(word, pos=self.__tagmap[tag[0]]) for word, tag in tagged]
        lemmatized_sentence = " ".join(lemmatized_words)
        return lemmatized_sentence

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Lemmatize input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input data for lemmatization.

        Returns
        -------
        pd.DataFrame
            Lemmatized data.
        """
        X = _validate_input(X)
        logging.info("Lemmatization of input data.")
        X[self.column_out] = X[self.column_in].apply(self.custom_lemmatizer)
        return X


class ExpandContractions(BaseEstimator, TransformerMixin):
    """Expands clitic contractions.

    Parameters
    ----------
    column_in : str
        Text column to be used for expanding contractions.
    column_out : str
        Column with expanded contractions.
    contractions_map : Optional[Dict[str, str]]
        Contractions mapping used for expansion. If `None`,
        default mappings will be used.
    """

    def __init__(self, column_in: str, column_out: str, contractions_map: Optional[Dict[str, str]] = None):
        self.column_in = column_in
        self.column_out = column_out
        self.contractions_map = contractions_map if contractions_map else CONTRACTIONS_MAP
        self.__contractions_pattern = self.__get_contractions_pattern()

    def __get_contractions_pattern(self):
        if not self.contractions_map:
            raise ValueError("Contractions mapping should be passed inorder to proceed.")
        contractions_pattern = re.compile(
            "({})".format("|".join(self.contractions_map.keys())), flags=re.IGNORECASE | re.DOTALL
        )
        return contractions_pattern

    def fit(self, X, y=None):
        return self

    def expand_contractions(self, text: str) -> str:
        """Expands contractions in the given text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with contractions expanded.
        """

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contractions = (
                self.contractions_map.get(match)
                if self.contractions_map.get(match)
                else self.contractions_map.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contractions[1:]
            return expanded_contraction

        expanded_text = self.__contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Expands contractions.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input data.

        Returns
        -------
        pd.DataFrame
            Data with contractions expanded.
        """
        X = _validate_input(X)
        logging.info("Expanding contractions.")
        X[self.column_out] = X[self.column_in].apply(self.expand_contractions)
        return X


class Clean(BaseEstimator, TransformerMixin):
    """Cleans text data.

    Parameters
    ----------
    column_in : str
        Text column for cleaning.
    column_out : str
        Cleaned text column.
    """

    def __init__(self, column_in: str, column_out: str):
        self.column_in = column_in
        self.column_out = column_out

    def fit(self, X, y=None):
        return self

    def clean(self, text: str) -> str:
        """Cleans input text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Cleaned text.
        """
        if text:
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r"[^A-Za-Z0-9(),!?\'\`]", " ", text)
            text = re.sub(r"[0-9]\w+|[0-9]", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\xa0", " ", text)
            text = " ".join([word for word in word_tokenize(text) if len(word) > 1])
            text = text.strip.lower()
            if len(text) > 1:
                return text
        return None

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Cleans text data.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input data.

        Returns
        -------
        pd.DataFrame
            Cleaned text data.
        """
        X = _validate_input(X)
        logging.info("Cleaning input text.")
        X[self.column_out] = X[self.column_in].apply(self.clean)
        return X


class CleanSpaces(BaseEstimator, TransformerMixin):
    """Cleans extra spaces.

    Parameters
    ----------
    column_in : str
        Text column for cleaning.
    column_out : str
        Cleaned text column.
    """

    def __init__(self, column_in: str, column_out: str):
        self.column_in = column_in
        self.column_out = column_out

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Removes extra spaces.

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series]
            Input text data.

        Returns
        -------
        pd.DataFrame
            Data with extra spaces removed.
        """
        X = _validate_input(X)
        logging.info("Removing extra spaces.")
        X[self.column_out] = X[self.column_in].apply(lambda x: re.sub(r"\s+", " ", x))
        X[self.column_out] = X[self.column_out].str.strip()
        return X


class NgramPhrases:
    """Ngram phrases along with metrics.

    Parameters
    ----------
    corpus :
        Text data.
    ngram :
        Ngram size.

    Attributes
    ----------
    ngram_df : pandas.DataFrame
        Ngram dataframe with relevant metrics.
    """

    def __init__(self, corpus: pd.Series, ngram: Optional[int] = 2):
        self.corpus = corpus
        self.ngram = ngram
        self.ngram_df = pd.DataFrame()

    @staticmethod
    def get_df(corpus: pd.Series, n: Optional[int] = 2) -> pd.DataFrame:
        """Calculates document frequency.

        Parameters
        ----------
        corpus : pd.Series
            Input text.
        n : Optional[int], {1, 2, 3}
            ngram size, by default 2

        Returns
        -------
        pd.DataFrame
            Document frequency of ngrams.

        Raises
        ------
        NotImplementedError
            Document frequency is only for uni, bi and tri-gram(s).
        """
        if n > 3 or n < 1:
            raise NotImplementedError
        ngram_list = []
        for text in corpus:
            unigrams = nltk.word_tokenize(text)
            if n == 1:
                ngrams = unigrams
            elif n == 2:
                ngrams = nltk.bigrams(unigrams)
            else:
                ngrams = nltk.trigrams(unigrams)
            ngram_list.append(list(dict.fromkeys(ngrams)))
        ngram_list = it.chain.from_iterable(ngram_list)
        ngram_cnt = list(Counter(ngram_list).items())
        return pd.DataFrame(ngram_cnt, columns=["Ngram", "Document_frequency"])

    @staticmethod
    def get_tfidf(corpus: pd.Series, n: Optional[int] = 2) -> pd.DataFrame:
        """Term frquency - inverse document frequency calculation.

        Parameters
        ----------
        corpus : pd.Series
            Input text data.
        n : Optional[int]
            ngram size, by default 2

        Returns
        -------
        pd.DataFrame
            Ngrams with tf-idf scores.
        """
        tfidf = TfidfVectorizer(ngram_range=(n, n), token_pattern="(?u)\\b[\\w+]+\\b")
        logging.info("TF-IDF, fitting data.")
        tfidf.fit(corpus)
        logging.info("TF-IDF, building vocab and its IDF.")
        idf_dict = [(word, tfidf.idf_[idx]) for word, idx in tfidf.vocabulary_.items()]
        tfidf_df = pd.DataFrame(idf_dict, columns=["Ngram", "tfidf"])
        tfidf_df.sort_values(by=["tfidf"], inplace=True)
        tfidf_df.index = np.arange(len(tfidf_df)) + 1
        tfidf_df = tfidf_df.rename_axis("tfidf_rank").reset_index()
        if n > 1:
            tfidf_df["Ngram"] = tfidf_df["Ngram"].apply(lambda x: tuple(x.split()))
        return tfidf_df

    def get_phrases(self, to_excel: Optional[str] = None) -> None:
        """Calculates different n-gram metrics.

        Frequency, document frequency, tf-idf, chi-square
        metrics are calculated.

        Parameters
        ----------
        to_excel : Optional[str]
            Excel filename with file path, by default None

        Raises
        ------
        ValueError
            Only Uni, bi and tri-gram(s) metrics are supported.
        """
        if self.ngram > 3 or self.ngram < 1:
            raise ValueError("Only uni-grams, bi-grams and tri-grams are supported.")
        unigrams = list(it.chain.from_iterable(self.corpus.str.split()))
        if self.ngram == 1:
            word_count = Counter(unigrams)
            ngram_freq_df = (
                pd.DataFrame.from_dict(word_count, orient="index", columns=["Freq"])
                .rename_axis("Ngram")
                .reset_index()
            )
        if self.ngram == 2:
            ngram_measures = nltk.collocations.BigramAssocMeasures()
            ngram_finder = nltk.collocations.BigramCollocationFinder.from_words(unigrams)
        if self.ngram == 3:
            ngram_measures = nltk.collocations.TrigramAssocMeasures()
            ngram_finder = nltk.collocations.TrigramCollocationFinder.from_words(unigrams)
        if self.ngram == 2 or self.ngram == 3:
            ngram_freq = ngram_finder.ngram_fd.items()
            ngram_freq_df = pd.DataFrame(list(ngram_freq), columns=["Ngram", "Freq"])
            ngram_chi_df = pd.DataFrame(
                list(ngram_finder.score_ngrams(ngram_measures.chi_sq)), columns=["Ngram", "Chi_sq"]
            )
        logging.info("Calculating TF-IDF values for ngrams.")
        tfidf_df = NgramPhrases.get_tfidf(self.corpus, self.ngram)
        logging.info("Calculating document frequency values for ngrams.")
        doc_freq = NgramPhrases.get_df(self.corpus, self.ngram)
        self.ngram_df = ngram_freq_df.merge(tfidf_df, left_on="Ngram", right_on="Ngram").merge(
            doc_freq, left_on="Ngram"
        )
        if self.ngram > 1:
            self.ngram_df = self.ngram_df.merge(ngram_chi_df, left_on="Ngram")
        if to_excel:
            logging.info("Writing data to %s excel." % to_excel)
            self.ngram_df.to_excel(to_excel, index=False)
        return None


class CCText(nltk.Text):
    def __init__(self, tokens, name=None):
        super().__init__(tokens, name)
        self.full_text = None

    def custom_concordance_text(
        self, word: str, width: Optional[int] = 79, lines: Optional[int] = 25
    ) -> List[str]:
        """Concordance for ngram text.

        Parameters
        ----------
        word : str
            Word to search.
        width : Optional[int]
            Context width, by default 79.
        lines : Optional[int]
            Number of lines to return, by default 25.

        Returns
        -------
        List[str]
            List of sentences with context around the `word`.
        """
        if not self.full_text:
            self.full_text = " ".join(self.tokens)
        total_len = len(self.full_text)
        context_sents = []
        pattern = r"(?<!-)\b" + word + r"\b(?!-)"
        for line_no, match in enumerate(re.finditer(pattern, self.full_text), 1):
            if lines and (line_no < lines):
                break
            start = int(match.start() - width / 2)
            if start < 0:
                start = 0
            end = int(match.end() + width / 2)
            if end > total_len:
                end = total_len
            context_sents.append(" ".join(self.full_text[start:end].split()[1:-1]))
        return context_sents

    def custom_concordance(
        self, word: str, context_word_count: Optional[int] = 10, lines: Optional[int] = 25
    ) -> List[str]:
        """Concordance for Ngramm text.

        Parameters
        ----------
        word : str
            Word to search.
        context_word_count : Optional[int]
            Number of words to consider around input word, by default 10.
        lines : Optional[int]
            Number of lines to return, by default 25.

        Returns
        -------
        List[str]
            List of sentences with context around the `word`.
        """
        if not self.full_text:
            self.full_text = " ".join(self.tokens)
        total_len = len(self.tokens)
        context_sents = []
        pattern = "\\b" + word + "\\b"
        window_start, window_end = 0, 0
        for line_no, match in enumerate(re.finditer(pattern, self.full_text), 1):
            if lines and (line_no > lines):
                break
            window_start = len(self.full_text[: match.start()].split()) - context_word_count // 2
            if window_start < 0:
                window_start = 0
            elif window_start < window_end:
                window_start = window_end
            window_end = len(self.full_text[: match.end()].split()) + context_word_count // 2
            if window_end > total_len:
                window_end = total_len
            context_sent = " ".join(self.tokens[window_start:window_end])
            if context_sent:
                context_sents.append(context_sent)
        return context_sents

    def context_word_frequencies(
        self, word: str, context_word_count: Optional[int] = 10, lines: Optional[int] = 25
    ) -> pd.DataFrame:
        """Word frequency for the context words.

        Parameters
        ----------
        word : str
            Word to search.
        context_word_count : Optional[int]
            Number of words to consider around the input word, by default 10.
        lines : Optional[int]
            Number of lines to return, by default 25.

        Returns
        -------
        pandas.DataFrame
            Sorted word frequencies.
        """
        context_word_list = [
            sentence.split() for sentence in self.custom_concordance(word, context_word_count, lines)
        ]
        context_words = it.chain.from_iterable(context_word_list)
        # context_words = sum(context_word_list, [])
        word_freq = Counter(context_words)
        if not word_freq:
            return pd.DataFrame()
        return (
            pd.DataFrame.from_dict(word_freq, orient="index", columns=["frequency"])
            .rename_axis("context_word")
            .reset_index()
            .sort_values(by=["frequency"], ascending=False)
        )
