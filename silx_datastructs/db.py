import json
from pydantic import BaseModel

from .dag import DAGEntity, NodeType

EDGE_SEPARATOR = "<||>"
DIGRAPH_NODE_SEPARATOR = "->"


def hyper_edge_string_to_entities(hyper_str: str) -> list[str]:
    edges: list[str] = hyper_str.split(EDGE_SEPARATOR)
    nodes: list[str] = []
    for edge in edges:
        s, e = edge.split(DIGRAPH_NODE_SEPARATOR)
        nodes.append(s)
        nodes.append(e)
    return nodes


class NodeMetaData(BaseModel):
    name: str
    canonical_unit: str
    entity_type: str
    description: str | None
    name_embedding: list[float]
    description_embedding: list[float] | None


class KeyBase(BaseModel):
    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def causes_targets(self):
        raise NotImplementedError()


class NodeKey(KeyBase):
    entity: DAGEntity
    node_type: NodeType

    def print(self) -> str:
        return f"{self.entity}"

    def __str__(self) -> str:
        return self.model_dump_json()

    def causes_targets(self) -> tuple[None, str]:
        return None, str(self.entity)


UNIVERSAL_CONTEXT_KEY = NodeKey(
    entity=DAGEntity(name="universalcontext"), node_type=NodeType.NULL
)


class NodeDomain(BaseModel):
    entity: DAGEntity
    domain: set[str]


class EdgeKey(KeyBase):
    src: NodeKey
    dst: NodeKey

    def __len__(self) -> int:
        return 1

    def print(self) -> str:
        return f"{self.src}{DIGRAPH_NODE_SEPARATOR}{self.dst}"

    def __str__(self) -> str:
        return self.model_dump_json()

    def as_tuple(self) -> tuple[str, str]:
        return self.src.entity.name, self.dst.entity.name

    def causes_targets(self) -> tuple[list[str], str]:
        return [self.src.entity.name], self.dst.entity.name


class HyperEdgeKey(KeyBase):
    edges: list[EdgeKey]
    _n: int = 0

    def __len__(self) -> int:
        return len(self.edges)

    def __hash__(self) -> int:
        return super().__hash__()

    def print(self) -> str:
        self.edges.sort(key=lambda x: str(x.src) + str(x.dst))
        return EDGE_SEPARATOR.join(map(str, self.edges))

    def __str__(self) -> str:
        d = self.model_dump()
        # sort keys for comparisons
        d["edges"].sort(
            key=lambda e: str(e["src"]["entity"]["name"])
            + str(e["dst"]["entity"]["name"])
        )
        return json.dumps(d)

    def causes_targets(self) -> tuple[list[str], list[str]]:
        causes = [e.src.entity.name for e in self.edges]
        targets = [e.dst.entity.name for e in self.edges]
        return causes, targets


class HyperEdgeKeyStringList(BaseModel):
    hyper_edges: list[str]


def edge_to_hyper_edge_lookup(
    hyper_edges: list[HyperEdgeKey],
) -> dict[EdgeKey, list[HyperEdgeKey]]:
    hyper_lookup: dict[EdgeKey, list[HyperEdgeKey]] = {}
    for hyper_edge in hyper_edges:
        for edge in hyper_edge.edges:
            if edge in hyper_lookup:
                hyper_lookup[edge].append(hyper_edge)
            else:
                hyper_lookup[edge] = [hyper_edge]

    return hyper_lookup
