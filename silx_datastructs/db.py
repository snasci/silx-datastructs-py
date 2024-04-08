from pydantic import BaseModel

from .dag import DAGEntity, NodeType

EDGE_SEPARATOR = "<::>"
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

    def __eq__(self, __value: object) -> bool:
        return str(self) == str(__value)

    def causes_targets(self):
        raise NotImplementedError()


class NodeKey(KeyBase):
    entity: DAGEntity
    node_type: NodeType

    def __str__(self) -> str:
        return f"[{str(self.entity)}]"

    def causes_targets(self) -> tuple[None, str]:
        return None, str(self.entity)


def node_key_from_string(s: str) -> NodeKey:
    # get rid of brackets
    name = s[1:-1]
    return NodeKey(name=name)


class EdgeKey(KeyBase):
    src: NodeKey
    dst: NodeKey

    def __len__(self) -> int:
        return 1

    def __str__(self) -> str:
        return f"{self.src}{DIGRAPH_NODE_SEPARATOR}{self.dst}"

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self) -> NodeKey:
        if self._n == 0:
            member = self.src
            self._n += 1
            return member
        elif self._n == 1:
            member = self.dst
            self._n += 1
            return member
        raise StopIteration

    def as_tuple(self) -> tuple[str, str]:
        return self.src.name, self.dst.name

    def causes_targets(self) -> tuple[list[str], str]:
        return [self.src.name], self.dst.name


def edge_key_from_string(s: str) -> EdgeKey:
    ssrc, sdst = s.split(DIGRAPH_NODE_SEPARATOR)
    src = NodeKey(name=node_key_from_string(ssrc))
    dst = NodeKey(name=node_key_from_string(sdst))
    return EdgeKey(src=src, dst=dst)


class HyperEdgeKey(KeyBase):
    edges: list[EdgeKey]
    _n: int = 0

    def __len__(self) -> int:
        return len(self.edges)

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        self.edges.sort(key=lambda x: str(x.src) + str(x.dst))
        return EDGE_SEPARATOR.join(map(str, self.edges))

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self) -> EdgeKey:
        if self._n < len(self.edges):
            member = self.edges[self._n]
            self._n += 1
            return member
        raise StopIteration

    def causes_targets(self) -> tuple[list[str], list[str]]:
        causes = [e.src.name for e in self.edges]
        targets = [e.dst.name for e in self.edges]
        return causes, targets


def hyperedge_key_from_string(s: str) -> HyperEdgeKey:
    edge_strings = s.split(EDGE_SEPARATOR)
    edges = [edge_key_from_string(s) for s in edge_strings]
    return HyperEdgeKey(edges=edges)


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
