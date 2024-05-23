import unittest

from silx_datastructs import dag
from silx_datastructs import db


class TestDB(unittest.TestCase):
    e1 = dag.DAGEntity(name="a")
    e2 = dag.DAGEntity(name="b")
    e3 = dag.DAGEntity(name="c")

    n1 = db.NodeKey(entity=e1, node_type=dag.NodeType.CONDITION)
    n2 = db.NodeKey(entity=e2, node_type=dag.NodeType.INTERVENTION)
    n3 = db.NodeKey(entity=e3, node_type=dag.NodeType.OUTCOME)

    ed1 = db.EdgeKey(src=n1, dst=n2)
    ed2 = db.EdgeKey(src=n2, dst=n3)

    he1 = db.HyperEdgeKey(edges=[ed1, ed2])
    he2 = db.HyperEdgeKey(edges=[ed2, ed1])

    def test_node_key(self):
        self.assertNotEqual(self.n1, self.n2)

        rk = self.n1.to_redis_key()
        self.assertNotIn(rk, ":")

        recovered = db.NodeKey.from_redis_key(rk)
        self.assertEqual(self.n1, recovered)

    def test_edge_key(self):
        self.assertNotEqual(self.ed1, self.ed2)

        rk = self.ed1.to_redis_key()
        self.assertNotIn(rk, ":")

        recovered = db.EdgeKey.from_redis_key(rk)
        self.assertEqual(self.ed1, recovered)

    def test_hyper_edge_key(self):
        self.assertEqual(self.he1, self.he2)

        rk = self.he1.to_redis_key()
        self.assertNotIn(rk, ":")

        recovered = db.HyperEdgeKey.from_redis_key(rk)
        self.assertEqual(recovered, self.he1)
        self.assertEqual(recovered, self.he2)


if __name__ == "__main__":
    unittest.main()
