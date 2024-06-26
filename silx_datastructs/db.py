import json
from pydantic import BaseModel

from .dag import DAGEntity, NodeType

EDGE_SEPARATOR = "<||>"
DIGRAPH_NODE_SEPARATOR = "->"


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

    def to_redis_key(self) -> str:
        json_str = str(self)
        redis_str = json_str.replace(":", _COLON_REPLACEMENT)
        return redis_str


_COLON_REPLACEMENT = "-|-"


class NodeKey(KeyBase):
    entity: DAGEntity
    node_type: NodeType

    def print(self) -> str:
        return f"{self.entity}"

    def __str__(self) -> str:
        return self.model_dump_json()

    def causes_targets(self) -> tuple[None, str]:
        return None, str(self.entity)

    @staticmethod
    def from_redis_key(k: str) -> "NodeKey":
        json_str = k.replace(_COLON_REPLACEMENT, ":")
        nk = NodeKey.model_validate_json(json_str)
        return nk


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

    def contains_node_by_name(self, query: str) -> NodeKey | None:
        if self.src.entity.name == query:
            return self.src
        elif self.dst.entity.name == query:
            return self.dst
        else:
            return None

    def rename_node(self, old_name: str, new_name: str) -> bool:
        renamed = False
        if self.src.entity.name == old_name:
            self.src.entity.name = new_name
            renamed = True
        if self.dst.entity.name == old_name:
            self.dst.entity.name = new_name
            renamed = True
        return renamed

    def node_names(self) -> list[str]:
        return [self.src.entity.name, self.dst.entity.name]

    @staticmethod
    def from_redis_key(k: str) -> "EdgeKey":
        json_str = k.replace(_COLON_REPLACEMENT, ":")
        ek = EdgeKey.model_validate_json(json_str)
        return ek


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

    def contains_node_by_name(self, query: str) -> NodeKey | None:
        for ek in self.edges:
            r = ek.contains_node_by_name(query)
            if r is not None:
                return r
        return None

    def rename_node(self, old_name: str, new_name: str) -> bool:
        renamed = False
        for ek in self.edges:
            r = ek.rename_node(old_name, new_name)
            renamed = renamed or r
        return renamed

    def node_names(self) -> list[str]:
        names: list[str] = []
        for edge in self.edges:
            names.extend(edge.node_names())
        return names

    @staticmethod
    def from_redis_key(k: str) -> "HyperEdgeKey":
        json_str = k.replace(_COLON_REPLACEMENT, ":")
        hek = HyperEdgeKey.model_validate_json(json_str)
        return hek


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
