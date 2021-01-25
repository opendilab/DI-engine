class PropertyItem:
    pass


class PropertyData:

    def __init__(self, item: PropertyItem, value, given: bool):
        self.__item = item
        self.__given = not not given
        self.__value = value

    @property
    def value(self):
        if self.__given:
            return self.__value
        else:
            raise ValueError('Value not given.')

    @value.setter
    def value(self, new_value):
        self.__given = True
        self.__value = new_value
