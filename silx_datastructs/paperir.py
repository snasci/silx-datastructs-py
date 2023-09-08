from pydantic import BaseModel


class Paper(BaseModel):
    id: int
    paper_type: str
    data_source: str
    n_studies: int


class Population(BaseModel):
    species: str
    sampling: str
    n: int | None
    inclusion: list[str] | None = None
    exclusion: list[str] | None = None


class Entity(BaseModel):
    token: str
    var_type: str | None
    domain: str | None
    unit: str | None
    context: str | None
    measure: str | None
    distribution: str | None


TValue = float | int | bool | dict[str, float]


class EntityState(BaseModel):
    entity: Entity
    state: str | None


class SID(BaseModel):
    causes: list[EntityState]
    outcome: EntityState


class Datum(BaseModel):
    n: int | None
    value: TValue
    variance: float | tuple[float, float] | None
    variance_type: str | None
    sid: SID


class PaperIR(BaseModel):
    paper: Paper
    population: Population
    entities: list[Entity]
    data: list[Datum]
