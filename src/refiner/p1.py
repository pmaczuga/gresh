from src.gresh import Gresh
from src.refiner.prod import Production


class P1(Production):
    r"""
    ```
        v3               v3
       / \              /|\
      /   \     =>     / | \
     /     \          /  |  \
    v1-----v2        v1--h--v2
    ```

    Conditions:
    - Triangle is marked to be refined (`:refined` property is set to `true`)
    - Breaks *longest edge*, if either is true:
        - It is on the boundary (`:boundary` property is set to `true`), **OR**
        - It's vertices are not hanging nodes **AND** other two egdes are not same
        length and on the boundary
    """

    def transform(self, g: Gresh, center: int) -> bool:
        mapping = self.check(g, center)
        if mapping is None:
            return False

        v3, v1, v2 = mapping
        B1 = g.is_on_boundary(v1, v2)

        g.remove_edge(v1, v2)
        new_coords = g.new_vertex_coords(v1, v2)
        h = g.add_vertex(new_coords)
        if not B1:
            g.set_hanging(h, v1, v2)

        g.add_edge(h, v1, boundary=B1)
        g.add_edge(h, v2, boundary=B1)
        g.add_edge(h, v3, boundary=False)

        g.add_pure_interior(v1, h, v3)
        g.add_pure_interior(h, v2, v3)

        g.remove_node(center)

        return True

    def check(self, g: Gresh, center: int) -> list[int] | None:
        if not g.is_interior(center):
            return None
        elif not g.should_refine(center):
            return None

        vA, vB, vC = g.interior_connectivity(center)
        hA = g.get_hanging_node_between(vB, vC)
        hB = g.get_hanging_node_between(vA, vC)
        hC = g.get_hanging_node_between(vA, vB)

        if len(list(filter(lambda x: x is None, [hA, hB, hC]))) != 3:
            return None  # Return if we have a hanging node

        if not g.has_edge(vA, vB) or not g.has_edge(vB, vC) or not g.has_edge(vC, vA):
            return None

        check_conditions = []

        for v1, v2, v3 in [[vA, vB, vC], [vB, vC, vA], [vC, vA, vB]]:
            B1 = g.is_on_boundary(v1, v2)
            B2 = g.is_on_boundary(v2, v3)
            B3 = g.is_on_boundary(v3, v1)
            L1 = g.distance(v1, v2)
            L2 = g.distance(v2, v3)
            L3 = g.distance(v3, v1)
            HN1 = g.is_hanging(v1)
            HN2 = g.is_hanging(v2)

            if (
                (L1 >= L2)
                and (L1 >= L3)
                and (
                    B1
                    or (
                        not B1
                        and (not HN1 and not HN2)
                        and not ((B2 and L2 == L1) or (B3 and L3 == L1))
                    )
                )
            ):
                check_conditions.append([v3, v1, v2])
        if not len(check_conditions) == 0:
            # Here we can undraw
            return check_conditions[0]

        return None
