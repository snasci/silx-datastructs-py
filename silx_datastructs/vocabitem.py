from pydantic import BaseModel


class VocabularyItem(BaseModel):
    id: str
    src_db: str
    text: str
    norm_text: str
    description: str
    outnode_id: list[str] | None
    innode_id: list[str] | None
    species: str
