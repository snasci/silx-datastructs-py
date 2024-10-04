import unittest
from unittest.main import main

import numpy as np

from silx_datastructs import db
from silx_datastructs import dag
from silx_datastructs import ranges
from silx_datastructs import distributions


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
        self.assertIsInstance(self.ed1.contains_node_by_name("a"), db.NodeKey)
        self.assertIsNone(self.ed1.contains_node_by_name("c"))
        self.ed1.rename_node("a", "n")
        self.assertIsInstance(self.ed1.contains_node_by_name("n"), db.NodeKey)
        self.assertIsInstance(self.ed1.contains_node_by_name("b"), db.NodeKey)
        self.ed1.rename_node("n", "a")

    def test_hyper_edge_key(self):
        self.assertEqual(self.he1, self.he2)

        rk = self.he1.to_redis_key()
        self.assertNotIn(rk, ":")

        recovered = db.HyperEdgeKey.from_redis_key(rk)
        self.assertEqual(recovered, self.he1)
        self.assertEqual(recovered, self.he2)

        self.assertIsInstance(self.he1.contains_node_by_name("a"), db.NodeKey)
        self.assertIsInstance(self.he1.contains_node_by_name("b"), db.NodeKey)
        self.assertIsInstance(self.he1.contains_node_by_name("c"), db.NodeKey)
        self.assertIsNone(self.he1.contains_node_by_name("z"))

        self.he1.rename_node("a", "y")
        self.assertIsInstance(self.he1.contains_node_by_name("y"), db.NodeKey)

    def test_gdb_hyper(self):
        s = "data:(2.6>3.3),(2.6>4.1),(2.6>5.1),(2.6>5.1),(2.6>6.1),(2.6>6.1),(2.6>6.1),(2.6>7.1),(2.6>7.1),(2.6>7.1),(2.6>7.1),(2.6>7.1),(2.6>7.1),(2.6>7.1),(2.6>8.1),(2.6>9.1),(2.6>10.1),(2.6>11.1),(2.6>12.1),(2.6>13.1),(2.6>14.1),(2.6>15.1),(2.6>16.1),(2.6>17.1),(2.6>18.1),(2.6>19.1),(2.6>20.1),(2.6>21.1),(2.6>22.1),(2.6>23.1),(2.6>24.1),(2.6>25.1),(2.6>26.1),(2.6>27.1),(2.6>28.1),(2.6>29.1),(2.6>30.1),(2.6>31.1),(2.6>32.1),(2.6>33.1),(2.6>34.1),(2.6>35.1),(2.6>36.1),(2.6>37.1),(2.6>38.1),(2.6>39.1),(2.6>40.1),(2.6>41.1),(2.6>41.1),(2.6>42.1),(2.6>42.1),(2.6>42.1),(2.6>43.1),(2.6>43.1),(2.6>44.1),(2.6>44.1),(2.6>44.1),(2.6>44.1),(2.6>45.1),(2.6>45.1),(2.6>45.1),(2.6>45.1),(2.6>45.1),(2.6>46.1),(2.6>46.1),(2.6>47.1),(2.6>47.1),(2.6>48.1),(2.6>48.1),(2.6>48.1),(2.6>49.1),(2.6>49.1),(2.6>49.1),(2.6>50.1),(2.6>50.1),(2.6>50.1),(3.3>51.4),(4.1>51.4),(5.1>51.4),(5.1>51.4),(6.1>51.4),(6.1>51.4),(6.1>51.4),(7.1>51.4),(7.1>51.4),(7.1>51.4),(7.1>51.4),(7.1>51.4),(7.1>51.4),(7.1>51.4),(8.1>51.4),(9.1>51.4),(10.1>51.4),(11.1>51.4),(12.1>51.4),(13.1>51.4),(14.1>51.4),(15.1>51.4),(16.1>51.4),(17.1>51.4),(18.1>51.4),(19.1>51.4),(20.1>51.4),(21.1>51.4),(22.1>51.4),(23.1>51.4),(24.1>51.4),(25.1>51.4),(26.1>51.4),(27.1>51.4),(28.1>51.4),(29.1>51.4),(30.1>51.4),(31.1>51.4),(32.1>51.4),(33.1>51.4),(34.1>51.4),(35.1>51.4),(36.1>51.4),(37.1>51.4),(38.1>51.4),(39.1>51.4),(40.1>51.4),(41.1>51.4),(41.1>51.4),(42.1>51.4),(42.1>51.4),(42.1>51.4),(43.1>51.4),(43.1>51.4),(44.1>51.4),(44.1>51.4),(44.1>51.4),(44.1>51.4),(45.1>51.4),(45.1>51.4),(45.1>51.4),(45.1>51.4),(45.1>51.4),(46.1>51.4),(46.1>51.4),(47.1>51.4),(47.1>51.4),(48.1>51.4),(48.1>51.4),(48.1>51.4),(49.1>51.4),(49.1>51.4),(49.1>51.4),(50.1>51.4),(50.1>51.4),(50.1>51.4),(52.2>51.4)"
        handler = db.GDBHyperEdgeHandler(s)
        self.assertIsInstance(handler.nodes(), set)


class TestDistributions(unittest.TestCase):
    def test_single_count_prob(self):
        good = distributions.SingleCountProbability(
            name="a", numerator=10, denominator=40
        )
        bad = distributions.SingleCountProbability(name="b", numerator=4, denominator=2)
        self.assertEqual(good.p(), 0.25)
        self.assertTrue(good.is_valid())
        self.assertFalse(bad.is_valid())

    def test_count_distribution(self):
        d1 = distributions.SingleCountProbability(
            name="male", numerator=4, denominator=10
        )
        d2 = distributions.SingleCountProbability(
            name="female", numerator=6, denominator=10
        )
        d3 = distributions.SingleCountProbability(
            name="bad", numerator=6, denominator=14
        )
        good = distributions.CountDistribution(probabilities=[d1, d2])
        bad1 = distributions.CountDistribution(probabilities=[d1, d2, d3])
        bad2 = distributions.CountDistribution(probabilities=[d1, d1])

        good.check()
        self.assertRaises(ValueError, bad1.check)
        self.assertRaises(ValueError, bad2.check)

        n1 = good.name_lookup("male")
        self.assertEqual(n1, d1)

        generated = good.generate()
        self.assertEqual(len(generated), 10)

        self.assertRaises(ValueError, bad1.generate)
        self.assertRaises(ValueError, bad2.generate)

        d4 = distributions.SingleCountProbability(
            name="true", numerator=4, denominator=10
        )
        d5 = distributions.SingleCountProbability(
            name="false", numerator=6, denominator=10
        )
        good2 = distributions.CountDistribution(probabilities=[d4, d5])
        arr = good2.generate()
        self.assertIsInstance(arr[0], bool)
        self.assertEqual(sum(arr), 4)

    def test_normal_distribution(self):
        good = distributions.NormalDistribution(mu=4, sigma=2, N=400)
        bad = distributions.NormalDistribution(mu=-2, sigma=0, N=0)

        good.check()
        self.assertRaises(ValueError, bad.check)

        generated = good.generate()
        m = np.mean(generated)
        self.assertEqual(len(generated), 400)
        self.assertAlmostEqual(m, 4.0, places=0)


class TestRanges(unittest.TestCase):
    def test_fragments(self) -> None:
        l1 = ranges.LeftRangeFragment(number=3, inclusive=False)
        l2 = ranges.LeftRangeFragment(number=5.6, inclusive=True)
        self.assertEqual(str(l1), "(3.0")
        self.assertEqual(str(l2), "[5.6")

        r1 = ranges.RightRangeFragment(number=6, inclusive=False)
        r2 = ranges.RightRangeFragment(number=8.6, inclusive=True)
        self.assertEqual(str(r1), "6.0)")
        self.assertEqual(str(r2), "8.6]")

    def test_range_statement(self) -> None:
        r1 = ranges.RangeStatement(
            left=ranges.LeftRangeFragment(number=4, inclusive=False), right=None
        )
        self.assertEqual(str(r1), "(4.0, _)")
        self.assertTrue(r1.in_range(5))
        self.assertFalse(r1.in_range(-1))
        self.assertEqual(r1.distance(2), 2)
        self.assertEqual(r1.distance(5), 0)

        r2 = ranges.RangeStatement(
            left=None,
            right=ranges.RightRangeFragment(number=4, inclusive=True),
        )
        self.assertEqual(str(r2), "(_, 4.0]")
        self.assertTrue(r2.in_range(4))
        self.assertFalse(r2.in_range(4.1))
        self.assertEqual(r2.distance(2), 0)
        self.assertEqual(r2.distance(5), 1)

        r3 = ranges.RangeStatement(
            left=ranges.LeftRangeFragment(number=4, inclusive=False),
            right=ranges.RightRangeFragment(number=7.4, inclusive=True),
        )
        self.assertEqual(str(r3), "(4.0, 7.4]")
        self.assertTrue(r3.in_range(4.1))
        self.assertFalse(r3.in_range(7.41))
        self.assertEqual(r3.distance(2), 2)
        self.assertEqual(r3.distance(5), 0)


if __name__ == "__main__":
    unittest.main()
