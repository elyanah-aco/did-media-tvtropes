from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random

from const import (
    DISABILITY_TROPE_DIRECTORY_URLS,
    VIDEO_GAME_URLS,
    FILM_URLS,
    ANIME_URLS,
)


@retry(
    wait=wait_random(min=2, max=5),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def parse_page_as_soup(url: str) -> BeautifulSoup:
    """
    Parse URL contents as a BeautifulSoup object.

    :param str url: URL to get data for
    :return: BeautifulSoup of URL
    :rtype: BeautifulSoup
    """
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    return soup


def get_directory_tropes(trope_directory_urls: list[str]) -> pd.DataFrame:
    """
    Get tropes URLs and descriptions for all tropes from a specific category page.

    :param list[str] trope_directory_urls: List of trope directory URLs
    :return: DataFrame of entries for trope in trope directories
    :rtype: pd.DataFrame
    """
    did_tropes: dict[str, str] = {
        "trope_name": [],
        "trope_url": [],
        "trope_description": [],
    }

    for dir_url in trope_directory_urls:
        soup = parse_page_as_soup(dir_url)

        trope_listings = [
            list for list in soup.select("li:not(.plus)") if list.select("a.twikilink")
        ]
        for trope in trope_listings:
            trope_url = "https://tvtropes.org" + trope.find("a")["href"]
            trope_page_soup = parse_page_as_soup(trope_url)

            # Descriptions are found in titles (h1) and example header (h2)/folders
            start_of_trope_description = trope_page_soup.find("h1")
            end_of_trope_description = trope_page_soup.find("h2")
            if not end_of_trope_description:
                end_of_trope_description = trope_page_soup.select("div.folderlabel")[0]
            trope_name = start_of_trope_description.text.strip()

            trope_description = [
                par_tag.text.encode("ascii", "ignore").decode()
                for par_tag in start_of_trope_description.find_all_next("p")
                if par_tag in end_of_trope_description.find_all_previous("p")
                and len(par_tag.text.split())
                >= 10  # To further remove non-description paragraph texts
            ]
            trope_description = " ".join(trope_description)

            did_tropes["trope_name"].append(trope_name.strip())
            did_tropes["trope_url"].append(trope_url)
            did_tropes["trope_description"].append(
                trope_description.replace("\n", "").strip()
            )

        disability_tropes_db = pd.DataFrame().from_dict(did_tropes)
        return disability_tropes_db


def get_media_urls(
    media_directory_urls: list[str], media_category: str
) -> pd.DataFrame():
    """
    Create a DataFrame of media entries with URLs and years from media directories.

    :param list[str] media_directory_urls: List of media directory URLs
    :param str category: Media category to get entries for; is used to exclude
        entries in page that don't belong to the specific category
    :return: DataFrame of media entries
    :rtype: pd.DataFrame
    """
    media_urls_dict: dict[str, list[Any]] = {
        "media_name": [],
        "media_year": [],
        "media_url": [],
    }

    for dir_url in media_directory_urls:
        soup = parse_page_as_soup(dir_url)
        folders = [
            folder
            for folder in soup.select("div.folderlabel")
            if re.search("\d", folder.text) and re.search("folder\d", folder["onclick"])
        ]
        for folder in folders:
            folder_id = re.findall("folder\d+", folder["onclick"])[0]
            year = re.findall("\d+", folder.text)[0]
            year_folder = soup.find("div", {"class": "folder", "id": folder_id})
            for media_entry in year_folder.select("a.twikilink"):
                if media_category in media_entry["href"]:
                    media_urls_dict["media_name"].append(media_entry.text.strip())
                    media_urls_dict["media_year"].append(year)
                    media_urls_dict["media_url"].append(
                        "https://tvtropes.org" + media_entry["href"]
                    )

    media_entries_db = pd.DataFrame().from_dict(media_urls_dict)
    media_entries_db["category"] = media_category
    return media_entries_db


def get_tropes_in_media_page(media_url: str) -> pd.DataFrame:
    """
    Gets all tropes within a certain media entry.

    :param str media_url: URL for media entry's TvTropes page
    :return: DataFrame of all tropes for the current entry
    :rtype: pd.DataFrame
    """
    media_page_entries: dict[str, list[Any]] = {
        "media_url": [],
        "trope_name": [],
        "media_trope_description": [],
    }

    soup = parse_page_as_soup(media_url)
    start_of_trope_listings = soup.find("h2")
    if not start_of_trope_listings:
        start_of_trope_listings = soup.find("h1")

    trope_listings = [
        list
        for list in start_of_trope_listings.find_all_next("li")
        if list.select("a.twikilink")
    ]

    cleaned_tropes: list[str] = []
    for trope in trope_listings:
        if re.search(
            "Tropes [A-Z]", trope.find("a")["href"]
        ):  # Entry tropes are in another page
            separate_trope_page_url = "https://tvtropes.org" + trope.find("a")["href"]
            separate_page_soup = parse_page_as_soup(separate_trope_page_url)
            separate_page_tropes = [
                list
                for list in separate_page_soup.findAll("li")
                if list.select("a.twikilink")
            ]
            for separate_trope in separate_page_tropes:
                add_cleaned_trope_entry(separate_trope, cleaned_tropes)
        else:
            add_cleaned_trope_entry(trope, cleaned_tropes)

    for cleaned_trope in cleaned_tropes:
        if ":" in cleaned_trope:
            trope_name, media_trope_description = cleaned_trope.split(":", maxsplit=1)
            media_page_entries["media_url"].append(media_url)
            media_page_entries["trope_name"].append(trope_name.strip())
            media_page_entries["media_trope_description"].append(
                media_trope_description.replace("\n", "").strip()
            )

    media_page_db = pd.DataFrame().from_dict(media_page_entries)
    return media_page_db


def add_cleaned_trope_entry(
    trope_tag: Tag, cleaned_tropes_list: list[str]
) -> list[str]:
    """
    Appends bulleted entry descriptions to the trope's main description.

    :param Tag trope_tag: HTML tag of the current trope entry/description
    :param list[str] cleaned_tropes_list: List of all currently cleaned trope descriptions
    :return: cleaned_tropes_list appended by the new cleaned trope entry
    :rtype: list[str]
    """
    is_a_main_entry = re.search("a class", str(trope_tag)[:13])
    if is_a_main_entry:
        cleaned_tropes_list.append(trope_tag.text)
    elif not is_a_main_entry and cleaned_tropes_list:
        cleaned_tropes_list[-1] += trope_tag.text
    else:
        pass


if __name__ == "__main__":
    disability_tropes_db = get_directory_tropes(DISABILITY_TROPE_DIRECTORY_URLS)
    disability_tropes_db.to_csv("data/disability_tropes", sep="|", index=False)
    print("Disability tropes saved.")

    video_games_db = get_media_urls(VIDEO_GAME_URLS, "VideoGame")
    film_db = get_media_urls(FILM_URLS, "Film")
    anime_db = get_media_urls(ANIME_URLS, "Anime")
    media_db = pd.concat([video_games_db, film_db, anime_db], ignore_index=True)

    media_tropes_list: list[pd.DataFrame] = []
    for media_url in media_db["media_url"]:
        media_tropes = get_tropes_in_media_page(media_url)
        media_tropes_list.append(media_tropes)
    media_tropes_db = pd.concat(media_tropes_list, ignore_index=True)
    media_tropes_db = media_tropes_db.merge(media_db, how="inner", on="media_url")

    disability_tropes_in_media = disability_tropes_db.merge(
        media_tropes_db, how="left", on="trope_name"
    )
    disability_tropes_in_media = (
        disability_tropes_in_media.query("media_url.notna()")
        .sort_values("media_year")
        .drop_duplicates(
            ["trope_name", "media_trope_description", "media_url"]
        )[  # Duplicates occur when entries in one franchise link to the same page
            [
                "trope_name",
                "media_trope_description",
                "media_name",
                "media_year",
                "category",
                "media_url",
            ]
        ]
    )
    disability_tropes_in_media.to_csv(
        "data/media_disability_tropes", sep="|", index=False
    )
    print("Entries for disability tropes in media saved.")
