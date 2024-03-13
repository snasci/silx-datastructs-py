from typing import Optional
from pydantic import BaseModel


class Author(BaseModel):
    surname: str
    givennames: str
    affiliations: Optional[list[str]] = None


class Reference(BaseModel):
    pmid: Optional[str]
    articleid: str
    title: Optional[str]
    authors: list[Author]


class Paragraph(BaseModel):
    content: str
    refs: list[Reference]


class Section(BaseModel):
    title: str
    paragraphs: list[Paragraph]


class Figure(BaseModel):
    img: str
    caption: str
    label: str
    encoding: str


class Table(BaseModel):
    content: str


class SetIDError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class BaseDocument(BaseModel):
    queryid: str
    src: str
    srcid: str


class Article(BaseDocument):
    queryid: str
    src: str
    srcid: str
    full_title: str
    citation: str
    journal: str
    tags: list[str]
    tagtext: str
    author_list: list[Author]
    abstract: str
    publication_date: str
    sections: list[Section]
    references: list[Reference]
    figures: list[Figure]
    tables: list[Table]


class ArticleUpdate(BaseModel):
    queryid: str
    src: str
    srcid: str
    full_title: str
    journal: str
    tags: list[str]
    author_list: list[Author]
    abstract: str
    publication_date: str
    sections: list[Section]
    references: list[Reference]
    figures: list[Figure]
    tables: list[Table]
