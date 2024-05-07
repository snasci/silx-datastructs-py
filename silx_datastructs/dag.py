from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, validator


class NodeType(Enum):
    BASELINE = 1
    INTERVENTION = 2
    CONDITION = 3
    OUTCOME = 4
    NULL = 5


class MetaDataBase(BaseModel):
    source: Optional[str] = None


class InterventionMetaData(MetaDataBase):
    entity_description: Optional[str] = None
    experiment_description: Optional[str] = None


class OutcomeMetaData(MetaDataBase):
    title: Optional[str] = None
    description: Optional[str] = None
    time_frame: Optional[str] = None
    population: Optional[str] = None
    experiment_description: Optional[str] = None


class BaselineMetaData(BaseModel):
    result_description: Optional[str] = None


class ConditionMetaData(MetaDataBase):
    definition: str


class StrHashableBaseModel(BaseModel):
    def __hash__(self) -> int:
        return hash(str(self))


class DAGEntity(StrHashableBaseModel):
    name: str
    cui: Optional[str] = None
    tree_numbers: Optional[list[str]] = None
    definition: Optional[str] = None

    @staticmethod
    def clean_str(s: str) -> str:
        return s.replace(":", "").lower()

    def __str__(self) -> str:
        return self.name

    @validator("name", pre=True, always=True)
    def clean_for_db(cls, val):
        if isinstance(val, str):
            return DAGEntity.clean_str(val)
        return val


class CatRange(BaseModel):
    low: float | None
    high: float | None
    low_inclusive: bool = True
    high_inclusive: bool = True


class UnitEntity(BaseModel):
    name: str
    dimensions: list[dict[str, Any]] = []
    uri: Optional[str] = None

    @staticmethod
    def from_quantulum3(q3_entity):
        return UnitEntity(
            name=q3_entity.name,
            dimensions=q3_entity.dimensions,
            uri=q3_entity.uri,
        )


class Unit(BaseModel):
    name: str
    entity: UnitEntity
    surfaces: list[str] = []
    uri: Optional[str] = None
    symbols: list[str] = []
    dimensions: list[dict[str, Any]] = []
    original_dimensions: Optional[list[dict[str, Any]]] = None
    lang: str = "en_US"

    @staticmethod
    def from_quantulum3(q3_unit):
        return Unit(
            name=q3_unit.name,
            entity=UnitEntity.from_quantulum3(q3_unit.entity),
            surfaces=q3_unit.surfaces,
            uri=q3_unit.uri,
            symbols=q3_unit.symbols,
            dimensions=q3_unit.dimensions,
            original_dimensions=q3_unit.original_dimensions,
            lang=q3_unit.lang,
        )


class Realization(StrHashableBaseModel):
    value: float | str | int | CatRange | bool
    unit: Optional[Unit] = None

    def __str__(self) -> str:
        s = f"{self.value}"
        if self.unit is not None:
            s += f"{self.unit.name}"
        return s


class DistributionParams(BaseModel):
    param: Realization
    dispersion: float | tuple[float, float] | None = None
    param_type: Optional[str] = None
    dispersion_type: Optional[str] = None
    n: Optional[int] = None

    def __str__(self) -> str:
        nm = None
        if self.param.unit is not None:
            nm = self.param.unit.name
        s = f"N(μ={self.param.value} {nm}, σ={self.dispersion})"
        return s


class DAGNode(StrHashableBaseModel):
    entity: DAGEntity
    realization: Optional[Realization] = None
    source: Optional[str] = None
    dist_params: Optional[DistributionParams] = None

    def __str__(self):
        s = f"{self.entity.name}"
        if self.realization is not None:
            s += f" = {self.realization}"

        return s


class ConditionNode(DAGNode):
    meta: Optional[ConditionMetaData] = None
    group_type: Optional[str] = None


def generate_placebo_condition(source: str, group_type: str) -> ConditionNode:
    return ConditionNode(
        entity=DAGEntity(
            name="placebo",
            cui="C0032042",
            tree_numbers=["D26.660", "E02.785"],
            definition="""Any dummy medication or treatment.
            Although placebos originally were medicinal
            preparations having no specific pharmacological
            activity against a targeted condition, the concept
            has been extended to include treatments or procedures,
            especially those administered to control groups in
            clinical trials in order to provide baseline measurements
            for the experimental protocol.""",
        ),
        realization=Realization(value=True),
        source=source,
        group_type=group_type,
    )


class OutcomeNode(DAGNode):
    meta: Optional[OutcomeMetaData] = None


class InterventionNode(DAGNode):
    context: Optional[str] = None
    meta: Optional[InterventionMetaData] = None


def generate_placebo_intervention(
    source: str, meta: Optional[InterventionMetaData] = None
) -> InterventionNode:
    return InterventionNode(
        entity=DAGEntity(
            name="placebo",
            cui="C0032042",
            tree_numbers=["D26.660", "E02.785"],
            definition="""Any dummy medication or treatment.
            Although placebos originally were medicinal
            preparations having no specific pharmacological
            activity against a targeted condition, the concept
            has been extended to include treatments or procedures,
            especially those administered to control groups in
            clinical trials in order to provide baseline measurements
            for the experimental protocol.""",
        ),
        realization=Realization(value=True),
        source=source,
        meta=meta,
    )


class BaselineNode(DAGNode):
    data: Optional[DistributionParams] = None
    meta: Optional[BaselineMetaData] = None


def get_node_type(entity: DAGNode) -> NodeType:
    if isinstance(entity, BaselineNode):
        return NodeType.BASELINE
    elif isinstance(entity, InterventionNode):
        return NodeType.INTERVENTION
    elif isinstance(entity, ConditionNode):
        return NodeType.CONDITION
    elif isinstance(entity, OutcomeNode):
        return NodeType.OUTCOME
    else:
        raise TypeError(f"Invalid input type {entity} type: {type(entity)}")


def remove_local_repeats(nodelist: list[DAGNode] | None) -> list[DAGNode] | None:
    if nodelist is None:
        return nodelist
    list_without_duplicates = []
    for item in nodelist:
        if item not in list_without_duplicates:
            list_without_duplicates.append(item)
    return list_without_duplicates


def remove_repeats_in_other(
    nodelist: list[DAGNode], reflist: list[DAGNode]
) -> list[DAGNode]:
    if nodelist is None:
        return nodelist
    list_without_duplicates = []
    for item in nodelist:
        if item not in list_without_duplicates and item not in reflist:
            list_without_duplicates.append(item)
    return list_without_duplicates


class ProbabilityStatement(BaseModel):
    baseline_entities: Optional[list[BaselineNode]] = None
    condition_entities: Optional[list[ConditionNode]] = None
    intervention_entities: Optional[list[InterventionNode]] = None
    outcome_entity: Optional[OutcomeNode] = None
    data: Optional[DistributionParams] = None

    def __str__(self) -> str:
        s = f"P([{self.outcome_entity}]"
        if (
            self.baseline_entities is not None
            or self.intervention_entities is not None
            or self.condition_entities is not None
        ):
            s += "|"
            if self.condition_entities is not None:
                for e in self.condition_entities:
                    s += f"[{e}], "
            if self.intervention_entities is not None:
                for e in self.intervention_entities:
                    s += f"[{e}], "
            if self.baseline_entities is not None:
                for e in self.baseline_entities:
                    s += f"[{e}], "
            # remove trailing comma
            s = s[:-2]
        s += ")"
        if self.data is not None:
            s += f" = {self.data}"
        return s

    def condition_entities_set(self) -> set[str]:
        ce = []
        if self.condition_entities is not None:
            ce = self.condition_entities
        return set([str(e) for e in ce])

    def baseline_entities_set(self) -> set[str]:
        be = []
        if self.baseline_entities is not None:
            be = self.baseline_entities
        return set([str(e) for e in be])

    def intervention_entities_set(self) -> set[str]:
        ie = []
        if self.intervention_entities is not None:
            ie = self.intervention_entities
        return set([str(e) for e in ie])

    def outcome_entity_set(self) -> set[str]:
        if self.outcome_entity is None:
            return set([])
        return set([str(self.outcome_entity)])

    def data_set(self) -> set[str]:
        if self.data is None:
            return set([])
        return set([str(self.data)])

    def equal_vec(self, prob_stmnt) -> tuple[bool]:
        c_v: bool = self.condition_entities_set() == prob_stmnt.condition_entities_set()
        b_v: bool = self.baseline_entities_set() == prob_stmnt.baseline_entities_set()
        i_v: bool = (
            self.intervention_entities_set() == prob_stmnt.intervention_entities_set()
        )
        o_v: bool = self.outcome_entity_set() == prob_stmnt.outcome_entity_set()
        d_v: bool = self.data_set() == prob_stmnt.data_set()
        eq_v = (c_v, b_v, i_v, o_v, d_v)
        return eq_v

    def remove_placebo_repeats(self):
        self.baseline_entities = remove_local_repeats(self.baseline_entities)
        self.condition_entities = remove_local_repeats(self.condition_entities)
        self.intervention_entities = remove_local_repeats(self.intervention_entities)
