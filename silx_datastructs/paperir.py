from pydantic import BaseModel
from datetime import date
from typing import Optional, Any

from .article import Author
from .dag import ProbabilityStatement


class Publication(BaseModel):
    citation: str
    context: str
    pmid: Optional[int]


class StudyInfo(BaseModel):
    title: Optional[str]
    study_date: date
    study_type: str
    status: str = "Completed"
    enrollment: Optional[int] = None
    short_title: Optional[str] = None
    abstract: Optional[str] = None
    sponsor: Optional[str] = None
    n_arms: Optional[int] = 1
    publication_list: Optional[Publication] = None


class StudySponsor(BaseModel):
    name: str
    status: str
    agency_class: str


class StudyDesign(BaseModel):
    allocation: Optional[str]
    intervention_model: Optional[str]
    observation_model: Optional[str]
    purpose: Optional[str]
    masking: Optional[str]


class PaperIR(BaseModel):
    id: str
    study_info: StudyInfo
    study_design: StudyDesign
    authors: list[StudySponsor] | list[Author]
    source_index: list[Any]
    dag: list[ProbabilityStatement]
