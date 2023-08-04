
from __future__ import annotations

import os

import nltk
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


def load_media_trope_corpus(media_trope_fp: str) -> pd.DataFrame:
    """
    Load dataframe of disability tropes in media.

    :param str media_trope_fp: Filepath for disability trope data.
    """
    return pd.read_csv(
        media_trope_fp,
        sep="|",
        index_col=None,
    )

def lemmatize_token(token: str) -> str:
    lemma = wn.morphy(token)
    if lemma:
        return lemma
    else:
        return token
    
def preprocess_into_tokens(trope_description: str, minimum_token_length: int = 4) -> list[str]:

    stop_words = stopwords.words("english")
    tokens = nltk.pos_tag(trope_description.split())
    tokens = [
        lemmatize_token(token).lower()
        for token, pos in tokens
        if lemmatize_token(token).lower() not in stop_words
        and len(token) >= minimum_token_length
        and token.isalpha()
        and pos != "NNP"
    ]
    return tokens

def create_topic_model(
        tokenized_descriptions: list[list[str]],
        num_topics: int = 10,
        no_below=5,
        no_above=0.7,
        ):

    dictionary = Dictionary(tokenized_descriptions)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify()

    bow_descriptions = [
        dictionary.doc2bow(desc)
        for desc in tokenized_descriptions
    ]
    lda_model = LdaModel(
        bow_descriptions,
        num_topics=num_topics,
        passes=num_topics,
        eval_every=10,
        id2word=dictionary,
        random_state=1,
        iterations=500,
    )
    return lda_model

def write_top_tokens_per_topic(
        lda_model: LdaModel,
        topic_output_tmpl: str,
        category: str,
        num_topics: int = 10,
        num_words: int = 10,
) -> None:
    topic_output_fp = topic_output_tmpl.format(category=category)
    os.makedirs(os.path.dirname(topic_output_fp), exist_ok=True)
    topic_file = open(topic_output_fp, "w")

    for index, topic in lda_model.show_topics(
        num_topics=num_topics,
        num_words=num_words,
        formatted=False
        ):
            topic_file.write(
                "{}|{}\n".format(index, "|".join(word[0] for word in topic))
            )

    topic_file.close()

if __name__ == "__main__":
    disability_media_tropes = load_media_trope_corpus("data/media_disability_tropes")

    disability_media_tropes["media_trope_description"] = disability_media_tropes["media_trope_description"].str.strip()
    disability_media_tropes["tokenized_description"] = disability_media_tropes["media_trope_description"].apply(preprocess_into_tokens)
    
    topic_output_tmpl = "data/topics/{category}_topics"
    lda_model = create_topic_model(disability_media_tropes["tokenized_description"].tolist())
    write_top_tokens_per_topic(lda_model, topic_output_tmpl, "all")

    for category in ("Film", "VideoGame"):
        print(f"Generating {category} topics...")
        cat_did_tropes = disability_media_tropes[disability_media_tropes["category"] == category]
        lda_model = create_topic_model(cat_did_tropes["tokenized_description"].tolist())
        write_top_tokens_per_topic(lda_model, topic_output_tmpl, category.lower())
        


