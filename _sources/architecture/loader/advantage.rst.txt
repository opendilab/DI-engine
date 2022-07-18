Advantages
================

In the introduction part, we can see the convenience of compositing the logic together.

So based on these features, some more advanced practice can be developed.

Transform and validation once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loader can be used in data validation, especially configuration data. For example,

.. code:: json

    {
        "name": "tom",
        "age": 23,
        "gender": "male"
    }

In the json data above, age and gender should be limited, name should not be too long. So we can build the loader to validate.

.. code:: python

    l = dict_(
        name=check_only(item("name") >> length(1, 64)),
        age=check_only(item("age") >> interval(0, None)),
        gender=check_only(item("gender") >> enum("male", "female", case_sensitive=False)),
    )

    parsed_info = l(personal_info)

Also, we can validate data's constraints. For example

.. code:: json

    {
        "a": 1,
        "b": 2,
        "sum": 3
    }

We can validate this kind of data with the following loader.

.. code:: python

    l = check_only(Loader(lcmp(norm(item('a')) + norm(item('b')), '==', norm(item('sum'))))))

Based on this, if we need to simplify the json (by removing `sum`), and then swap the `a` and `b`, we can build the loader to validate and transform one.

.. code:: python

    sl = l & dict_(a=item('b'), b=item('a'))
    sl({'a': 1, 'b': 2, 'sum': 3})  # {'a': 2, 'b': 1}
    sl({'a': 1, 'b': 2, 'sum': 4})  # ValueError, 1 + 2 != 4

Support for equivalent forms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base on `or` feature, equivalent forms of the configuration can be easily supported. For example, this is form 1

.. code:: json

    {
        "a": 1,
        "b": 2
    }

and this is form 2,

.. code:: json

    {
        "sum": 3,
        "diff": -1
    }

The actual `a` and `b` should be from 0 to 10.

we can standardize the 2 forms by the following loader

.. code:: python

    common_loader = dict_(a=item('a'), b=item('b'))
    ss_loader = dict_(
        a=(norm(item('sum')) + norm(item('diff'))) / 2,
        b=(norm(item('sum')) - norm(item('diff'))) / 2,
    )
    validator = (item('a') >> interval(0, 10)) && (item('b') >> interval(0, 10))
    loader = (common_loader | ss_loader) >> check_only(validator)

So the calculation result should be

.. code:: python

    loader({'a': 1, 'b': 2})        # {'a': 1, 'b': 2}
    loader({'sum': 3, 'diff': -1})  # {'a': 1, 'b': 2}
    loader({'a': 1, 'b': -1})       # ValueError, b invalid
    loader({'sum': 0, 'diff': -2})  # ValueError, a invalid
    loader({'a': 1, 'c': 2})        # KeyError

Not only this, let's suggest that format 3 appeared at some time, like this (it is a simple string)

.. code:: json

    "1,2"

So we can migrate the loader like this

.. code:: python

    common_loader = dict_(a=item('a'), b=item('b'))
    ss_loader = dict_(
        a=(norm(item('sum')) + norm(item('diff'))) / 2,
        b=(norm(item('sum')) - norm(item('diff'))) / 2,
    )
    validator = (item('a') >> interval(0, 10)) && (item('b') >> interval(0, 10))
    loader_v0 = (common_loader | ss_loader) >> check_only(validator)

    @Loader
    def splitter(string):
        a, b = (int(item) for item in string.split(','))
        return dict(a=a, b=b)

    loader_v1 = (common_loader | ss_loader | splitter) >> check_only(validator)

The calculation result should be

.. code:: python

    loader_v1({'a': 1, 'b': 2})        # {'a': 1, 'b': 2}
    loader_v1({'sum': 3, 'diff': -1})  # {'a': 1, 'b': 2}
    loader_v1({'a': 1, 'b': -1})       # ValueError, b invalid
    loader_v1({'sum': 0, 'diff': -2})  # ValueError, a invalid
    loader_v1({'a': 1, 'c': 2})        # KeyError
    loader_v1('1,2')                   # {'a': 1, 'b': 2}

Afterwards, the inner data structure is changed, `a` or `b` are no longer used, but `first` and `second` instead, like this

.. code:: json

    {
        "first": 1,
        "second": 2
    }

we can migrate loader like this

.. code:: python

    common_loader = dict_(a=item('a'), b=item('b'))
    ss_loader = dict_(
        a=(norm(item('sum')) + norm(item('diff'))) / 2,
        b=(norm(item('sum')) - norm(item('diff'))) / 2,
    )
    validator = (item('a') >> interval(0, 10)) && (item('b') >> interval(0, 10))
    loader_v0 = (common_loader | ss_loader) >> check_only(validator)

    @Loader
    def splitter(string):
        a, b = (int(item) for item in string.split(','))
        return dict(a=a, b=b)

    loader_v1 = (common_loader | ss_loader | splitter) >> check_only(validator)

    new_loader_v0 = loader_v1 >> dict(first=item('a'), second=item('b'))

The calculation result should be

.. code:: python

    new_loader_v0({'a': 1, 'b': 2})        # {'first': 1, 'second': 2}
    new_loader_v0({'sum': 3, 'diff': -1})  # {'first': 1, 'second': 2}
    new_loader_v0({'a': 1, 'b': -1})       # ValueError, b invalid
    new_loader_v0({'sum': 0, 'diff': -2})  # ValueError, a invalid
    new_loader_v0({'a': 1, 'c': 2})        # KeyError
    new_loader_v0('1,2')                   # {'first': 1, 'second': 2}


