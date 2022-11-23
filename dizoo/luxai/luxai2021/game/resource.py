"""
Implements /src/Resource/index.ts
"""


class Resource:
    """
    Enum implementation
    """

    class Types:
        WOOD = 'wood'
        COAL = 'coal'
        URANIUM = 'uranium'

    def __init__(self, resource_type, amount) -> None:
        """

        :param resource_type:
        :param amount:
        """
        self.type = resource_type
        self.amount = amount
