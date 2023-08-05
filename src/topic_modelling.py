
from __future__ import annotations

import os
from typing import Any

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import Nmf, LdaModel
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from gsdmm import MovieGroupProcess


def load_trope_corpus(media_trope_fp: str) -> pd.DataFrame:
    """
    Load dataframe for disability trope data.

    :param str media_trope_fp: Filepath for trope description/example data
    :return: DataFrame of trope data'
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        media_trope_fp,
        sep="|",
        index_col=None,
    )

def lemmatize_token(token: str) -> str:
    """
    Perform lemmatization (using WordNet lemmatizer) on the current token.

    :param str token: Token to lemmatize
    :return: Token lemma, or token itself if no lemma is returned
    :rtype: str
    """
    lemma = wn.morphy(token)
    if lemma:
        return lemma
    else:
        return token
    
def preprocess_into_tokens(
        trope_description: str,
        minimum_token_length: int = 4
    )-> list[str]:
    """
    Tokenize input string, lemmatize tokens, and remove stopwords and tokens
    no longer than the minimum token length provided.
    
    :param str trope_description: Input string consisting of trope description/example
    :param int minimum_token_length: Minimum number of letters of tokens to keep
    :return: List of input string tokens after preprocessing
    :rtype: list[str]
    """
    stop_words = stopwords.words("english")
    tokens = nltk.pos_tag(trope_description.split())
    tokens = [
        lemmatize_token(token).lower()
        for token, pos in tokens
        if lemmatize_token(token).lower() not in stop_words
        and len(token) >= minimum_token_length
        and token.isalpha()
        and "NN" in pos
        and pos != "NNP"
    ]
    return tokens

def create_dictionary_and_bow(
        tokenized_descriptions: list[list[str]],
        no_below: int = 5,
        no_above: float = 0.5,
) -> tuple(Dictionary, list[list[Any]]):
    """
    Create dictionary and bag-of-words (BoW) representations for the
    tokenized documents.

    :param list[list[str]] tokenized_descriptions: Preprocessed tokens for entire text data
    :param int no_below: Minimum number of documents that tokens must appear in to be kept
    :param float no_above: Maximum fraction of documents that tokens must appear in to be kept
    :return: Tuple with dictionary representation as first element and BoW as second element
    :rtype: tuple(Dictionary, list[list[Any]])
    """
    dictionary = Dictionary(tokenized_descriptions)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify()

    bow_descriptions = [
        dictionary.doc2bow(desc)
        for desc in tokenized_descriptions
    ]
    print(bow_descriptions)

    return (dictionary, bow_descriptions)

def create_lda_model(
        dictionary: Dictionary,
        bow_descriptions: list[list[Any]],
        output_fp: str,
        num_topics: int = 25,
    ) -> None:
        """
        Generate and save top words for topics generated through Latent Dirichlet Allocation (LDA).

        :param Dictionary dictionary: Dictionary representation of text corpus
        :param list[list[Any]] bow_descriptions: BoW representation of text corpus
        :param str output_fp: Filepath to save topics to
        :param int num_topics: Number of topics to generate
        """
        lda_model = LdaModel(
            bow_descriptions,
            num_topics=num_topics,
            passes=num_topics,
            eval_every=10,
            id2word=dictionary,
            random_state=1,
            terations=500,
        )
        write_top_tokens_per_topic(lda_model, output_fp, num_topics)
    
def create_nmf_model(
        dictionary,
        bow_descriptions,
        output_fp: str,
        num_topics: int = 25,
) -> None:
        """
        Generate and save top words for topics generated through Non-negative Matrix Factorization (NMF).

        :param Dictionary dictionary: Dictionary representation of text corpus
        :param list[list[Any]] bow_descriptions: BoW representation of text corpus
        :param str output_fp: Filepath to save topics to
        :param int num_topics: Number of topics to generate
        """
        nmf_model = Nmf(bow_descriptions,
               num_topics=num_topics,
               passes=num_topics,
               eval_every=10,
               id2word=dictionary,
               random_state=1,
               )
        write_top_tokens_per_topic(nmf_model, output_fp, num_topics)

def create_gdsmm_model(
        tokenized_descriptions: list[list[str]],
        output_fp: str,
        num_topics: int = 25,
        num_words: int = 10,
) -> None:
        """
        Generate and save top words for topics generated through Gibbs Sampling Multinomial Mixture (GSDMM).

        :param list[list[str]] tokenized_descriptions: Preprocessed tokens for each document in the text corpus
        :param str output_fp: Filepath to save topics to
        :param int num_topics: Number of topics to generate
        :param int num_words: Number of words to save for each topic
        """
        no_of_words = len(set(word for desc in tokenized_descriptions for word in desc))
        mgp = MovieGroupProcess(K=num_topics, alpha=0.1, beta=0.1, n_iters=num_topics)
        mgp.fit(tokenized_descriptions, no_of_words)

        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        topic_file = open(output_fp, "w")

        doc_count = np.array(mgp.cluster_doc_count)
        top_topics = doc_count.argsort()[-num_topics:][::-1]

        for topic in top_topics:
            top_words_and_distributions = sorted(mgp.cluster_word_distribution[topic].items(), key=lambda k: k[1], reverse=True)[:num_words]
            topic_file.write(
                "{}|{}\n".format(list(top_topics).index(topic), "|".join(pair[0] for pair in top_words_and_distributions))
            )

def write_top_tokens_per_topic(
        model: LdaModel | Nmf,
        topic_output_fp: str,
        num_topics: int = 25,
        num_words: int = 10,
) -> None:
        """
        Write topics and top words per topics generated by Gensim models as a text file.

        :param LdaModel | Nmf model: Either an LdaModel or an Nmf model trained on text data
        :topic_output_fp: Filepath to save data to
        :param int num_topics: Number of topics to save
        :param int num_words: Number of words to save for each topic
        """
        os.makedirs(os.path.dirname(topic_output_fp), exist_ok=True)
        topic_file = open(topic_output_fp, "w")

        for index, topic in model.show_topics(
            num_topics=num_topics,
            num_words=num_words,
            formatted=False
            ):
                topic_file.write(
                    "{}|{}\n".format(index, "|".join(word[0] for word in topic))
                )

        topic_file.close()

#def create_bertopic_model(
        #documents
#):
    #bertopic_model = BERTopic()
    #topics, probs = bertopic_model.fit_transform(documents)
    #print(bertopic_model.get_topic_info().head(21))


if __name__ == "__main__":
    disability_media_tropes = load_trope_corpus("data/media_disability_tropes")

    disability_media_tropes["media_trope_description"] = disability_media_tropes["media_trope_description"].str.strip()
    disability_media_tropes["tokenized_description"] = disability_media_tropes["media_trope_description"].apply(preprocess_into_tokens)
    
    topic_output_tmpl = "data/topics/{model}/{category}_topics"
    for category in ("All", "Anime", "Film", "VideoGame"):
        print(f"Generating {category} topics...")
        if category != "All":
            cat_disability_tropes = disability_media_tropes[disability_media_tropes["category"] == category]
        else:
            cat_disability_tropes = disability_media_tropes.copy()

        tokenized_descriptions = cat_disability_tropes["tokenized_description"].tolist()
        dictionary, bow_descriptions = create_dictionary_and_bow(tokenized_descriptions)
        num_topics = 10

        print("Performing topic modelling via LDA...")
        lda_output_fp = topic_output_tmpl.format(category=category.lower(), model="lda")
        create_lda_model(dictionary, bow_descriptions, num_topics=num_topics, output_fp=lda_output_fp)
        print()

        print("Performing topic modelling via NMF...")
        nmf_output_fp = topic_output_tmpl.format(category=category.lower(), model="nmf")
        nmf_model = create_nmf_model(dictionary, bow_descriptions, num_topics=num_topics, output_fp=nmf_output_fp)
        print()
        
        #print("Performing topic modelling via BERTopic...")
        #create_bertopic_model(cat_disability_tropes["media_trope_description"].tolist())

        print("Performing topic modelling via GDSMM...")
        gdsmm_output_fp = topic_output_tmpl.format(category=category.lower(), model="gdsmm")
        create_gdsmm_model(tokenized_descriptions, num_topics=10, output_fp=gdsmm_output_fp)
        print()

