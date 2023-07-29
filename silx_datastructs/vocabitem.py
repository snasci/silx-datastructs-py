from pydantic import BaseModel


class VocabularyItem(BaseModel):
    id: str
    src_db: str
    token: str
    species: str
    var_type: str | None
    domain: list[str] | None
    description: str | None
    parents: list[str] | None
    children: list[str] | None
