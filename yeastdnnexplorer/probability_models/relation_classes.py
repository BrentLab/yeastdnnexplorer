class Relation:
    """Base class for relations between TF indices."""

    def evaluate(self, bound_vec: list[int]):
        raise NotImplementedError


class And(Relation):
    """Class for representing the logical AND of multiple conditions Allows nesed
    conditions, i.e. And(1, Or(2, 3))"""

    def __init__(self, *conditions):
        """
        :param conditions: List of conditions to be evaluated
        :type conditions: List[float | Relation]
        """
        self.conditions = conditions

    def evaluate(self, bound_vec):
        """
        Returns true if the And() condition evaluates to true Evaluates nested
        conditions as needed.

        :param bound_vec: Vector of TF indices (0 or 1) indicating which TFs are bound
            for the gene in question
        :type bound_vec: List[float]

        """
        if type(bound_vec) is not list or not all(
            isinstance(x, float) for x in bound_vec
        ):
            raise ValueError("bound_vec must be a list of floats")

        if not self.conditions:
            return True

        # Each condition can be an index or another Relation (And/Or)
        return all(
            c.evaluate(bound_vec) if isinstance(c, Relation) else bound_vec[c]
            for c in self.conditions
        )

    def __str__(self):
        return f"AND({', '.join(str(c) for c in self.conditions)})"


class Or(Relation):
    def __init__(self, *conditions):
        """
        :param conditions: List of conditions to be evaluated
        :type conditions: List[int | Relation]
        """
        self.conditions = conditions

    def evaluate(self, bound_vec):
        """
        Returns true if the Or() condition evaluates to true Evaluates nested conditions
        as needed.

        :param bound_vec: Vector of TF indices (0 or 1) indicating which TFs are bound
            for the gene in question
        :type bound_vec: List[int]

        """
        if type(bound_vec) is not list or not all(
            isinstance(x, float) for x in bound_vec
        ):
            raise ValueError("bound_vec must be a list of floats")

        if not self.conditions:
            return True

        # Each condition can be an index or another Relation (And/Or)
        return any(
            c.evaluate(bound_vec) if isinstance(c, Relation) else bound_vec[c]
            for c in self.conditions
        )

    def __str__(self):
        return f"OR({', '.join(str(c) for c in self.conditions)})"


# EXAMPLE USAGE:
# Defining a complex condition:
# "index 2 should only have its score adjusted if it is activated and if
#  (index 5 and 7) or 3 is activated"
condition = And(
    2,  # Index 2 must be activated
    Or(
        And(5, 7),  # Both indices 5 and 7 must be activated
        3,  # Or index 3 must be activated
    ),
)
