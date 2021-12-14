.. _header-n50:

Loader Overview
===============

.. _header-n52:

Overview
------------

Loader is a light, independent, and flexible data assembling module in our framework.

Loader Module \ **though is originally for data configurations, conversions, and verifications**\, had been found to have more potential for features and usages. Therefore Loader Module have been set to be an independent module.  

.. _header-n55:

Examples
------------

Loader's basic design idea is to operate common, small-scaled conversions and verifications. It uses methods such as assembly of logic units to form complex verifiable conversion structures.

Followings are some specific simple cases：

-  String Enumerate Verification

.. code:: python

   colors = enum('red', 'green', 'blue')  # Enum loader built
   colors('red')   # 'red'
   colors('Red')   # ValueError
   colors('None')  # ValueError

   ucolors = enum('red', 'green', 'blue', case_sensitive=False)  # Case-insensitive enum loader build
   ucolors('red')   # 'red'
   ucolors('Red')   # 'red'
   ucolors('None')  # ValueError

-  Type Verification and Conversion

.. code:: python

   itype = is_type(float)  # Verifiable Type loader built
   itype(1.2)    # 1.2
   itype(1)      # TypeError
   itype('str')  # TypeError

   otype = to_type(float)  # Convertable Type loader built
   otype(1.2)    # 1.2
   otype(1)      # 1
   otype('1.5')  # 1.5
   otype('inf')  # inf （Yes, python support this inf i.e. float('inf') or math.inf）
   otype('str')  # ValueError

-  Interval Verification

.. code:: python

   it = interval(1, 5, right_ok=False)  # Verify whether in interval [1, 5)
   it(1)    # 1
   it(3.8)  # 3.8
   it(5)    # ValueError
   it(-1)   # ValueError

   it = interval(None, 10)  # Verify whether in interval (-inf, 10]
   it(-100)  # -100
   it(10)    # 10
   it(11)    # ValueError

-  Simple Math

.. code:: python

   pl = plus(1)  # Tautology，add to 1
   pl(1)    # 2
   pl(1.2)  # 2.2

   mi = minus(2)  # Tautology，sub by 1
   mi(1)  # -1
   mi(3)  # 1

   cp = mcmp(1, "<", keep(), "<=", 10)  # Judge whether larger than 1, not larger than 10
   cp(1)   # ValueError
   cp(2)   # 2
   cp(10)  # 10
   cp(11)  # ValueError

-  List Element Verification

.. code:: python

   cl = collection(is_type(int))  # Judge whether is list of int type element
   cl([1, 2, 3])     # [1, 2, 3]
   cl([1, 2, None])  # TypeError

   co = contains(2)  # Judge whether has element 2
   co([1, 2, 3])  # [1, 2, 3]
   co([1, 3, 3])  # ValueError

-  Dict Mapping Verification

.. code:: python

   itt = item('a')  # Inspect and extract charater 'a'
   itt({'a': 1, 'b': 2})  # 1
   itt({'a': 2})          # 2
   itt({'aa': 2})         # KeyError

   dt = dict_(  # Build data of Dict format
       a=item('b'),
       b=item('a'),
   )
   dt({'a': 1, 'b': 2})   # {'a': 2, 'b': 1}
   dt({'a': 2, 'bb': 3})  # KeyError

-  Assembling Logic Units (and, or, pass-on)

.. code:: python

   iit = is_type(int) & interval(1, 5, right_ok=False)  # Whether is an int type value with interval [1, 5)
   iit(1)    # 1
   iit(4)    # 4
   iit(5)    # ValueError
   iit(-1)   # ValueError
   iit(3.8)  # TypeError

   iit = interval(None, -1) | interval(1, None)  # Whether is with interval (-inf, -1] | [1, +inf)
   iit(-2)  # -2
   iit(-1)  # -1
   iit(0)   # ValueError
   iit(1)   # 1
   iit(2)   # 2

   iit = to_type(float) >> (interval(None, -1) | interval(1, None))  # Whether is with interval (-inf, -1] | [1, +inf) after converting to float type value
   iit(1)     # 1.0
   iit('-1')  # -1.0
   iit('0')   # ValueError (Note that it is ValueError here, not TypeError)

-  Basic Modules

.. code:: python

   kp = keep()  # Tautology, preserve old value
   kp(1)     # 1
   kp(None)  # None

   r = raw(233)  # Tautology, constant
   r(1)     # 233
   r(None)  # 233

   r = optional(is_type(int) | is_type(float))  # Option type, equivalent to any of {int, float, None} types
   r(1)      # 1
   r(1.2)    # 1.2
   r(None)   # None
   r('str')  # TypeError

   ck = check_only(to_type(float) >> plus(2))  # Revert converted value
   ck(1)    # 1
   ck(2.2)  # 2.2

   ckx = to_type(float) >> plus(2)  # if without reversion
   ck(1)    # 3.0
   ck(2.2)  # 4.2

-  norm mechanism (to support moderately complex mathematical operations)

.. code:: python

   mt = norm(keep()) * (norm(keep()) + 1) - 10 / norm(keep())  # Calculate x * (x + 1) - 10 / x
   mt(1)    # -8
   mt(3.5)  # 12.8929

   tt = Loader(mt) >> interval(None, 10)  # Judge whether x * (x + 1) - 10 / x is in interval (-inf, 10]
   tt(1)    # -8
   tt(3.5)  # ValueError

-  normfunc mechanism (used to support highly complex mathematical calculations or uncontrollable calculation logic)
  
.. code:: python

   def _calculate(x, y):
       return x ** (1 / y)

   @normfunc
   def _calculate2(x, y):
       return x / (1 + y)

   nf = normfunc(_calculate)(norm(item('a')), norm(item('b')))  # Calculate  a ** (1 / b)
   nf({'a': 3, 'b': 7})  # 1.1699

   nf2 = _calculate2(norm(item('a')) - 1, norm(item('b')))  # Calculate (a - 1) / (1 + b)
   nf2({'a': 3, 'b': 7})  # 0.25

.. _header-n98:

FAQ
--------

.. _header-n100:

Q：What is the difference and relationship between loader and norm?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：The difference between the two is:

-  Loader focuses on the construction, conversion and verification of logic

-  Norm focuses on mathematical calculations, especially the construction of math calculation logic

\ ``Loader``\  can convert any type (including norm) to loader, \ ``norm``\ can convert the loader type to norm. Please note that in usage practices, in order to avoid ambiguity，\ **when loader and norm are used together, please pay attention to adding**\ ``norm`` or ``Loader``\ **fn to identify, to prevent the system from using unexpected computing overloads.**


Q：Loader sometimes feels too long to write, and it is all repetitive. Is there a better format?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：Simply, you can consider reuse. E.g

.. code:: python

   l1 = dict_(
       a=item('a') >> item('b') >> item('c'),
       b=item('a') >> item('b') >> item('d'),
   )

which can be simplified to

.. code:: python

   find_ab = item('a') >> item('b')
   l1 = dict_(
       a=find_ab >> item('c'),
       b=find_ab >> item('d'),
   )

**The principle of logical combination of loader is computing a new loader based on the two old loaders.** Therefore reusing is appropriate.

In operations, it is also recommended that users make full use of this feature to implement more elegant codes.