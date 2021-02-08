.. _header-n50:

Loader Overview
===============

.. _header-n52:

概述
----

Loader为一个轻量化、自由、灵活的数据组合框架组件。

Loader组件\ **原计划用于配置数据的转化与验证**\ ，然而实际开发中探索出了其更多的用法以及更大的功能潜力，故作为一个独立模块存在。

.. _header-n55:

示例
----

Loader的基本设计思路是将一些常见的小规模转化与验证模块，通过逻辑组合等方式，形成复杂的验证转换结构。

以下是一些具体的简单例子：

-  字符串枚举验证

.. code:: python

   colors = enum('red', 'green', 'blue')  # 构建一个枚举loader
   colors('red')   # 'red'
   colors('Red')   # ValueError
   colors('None')  # ValueError

   ucolors = enum('red', 'green', 'blue', case_sensitive=False)  # 构建一个大小写不敏感的枚举loader
   ucolors('red')   # 'red'
   ucolors('Red')   # 'red'
   ucolors('None')  # ValueError

-  类型判断与转换

.. code:: python

   itype = is_type(float)  # 构建一个类型验证loader
   itype(1.2)    # 1.2
   itype(1)      # TypeError
   itype('str')  # TypeError

   otype = to_type(float)  # 构建一个类型转换loader
   otype(1.2)    # 1.2
   otype(1)      # 1
   otype('1.5')  # 1.5
   otype('inf')  # inf （没错，python真支持这个，即float('inf')或math.inf）
   otype('str')  # ValueError

-  区间验证

.. code:: python

   it = interval(1, 5, right_ok=False)  # 验证是否在区间[1, 5)
   it(1)    # 1
   it(3.8)  # 3.8
   it(5)    # ValueError
   it(-1)   # ValueError

   it = interval(None, 10)  # 验证是否在区间(-inf, 10]
   it(-100)  # -100
   it(10)    # 10
   it(11)    # ValueError

-  简单数学运算

.. code:: python

   pl = plus(1)  # 恒真，加一
   pl(1)    # 2
   pl(1.2)  # 2.2

   mi = minus(2)  # 恒真，减二
   mi(1)  # -1
   mi(3)  # 1

   cp = mcmp(1, "<", keep(), "<=", 10)  # 判断是否大于1且不大于10
   cp(1)   # ValueError
   cp(2)   # 2
   cp(10)  # 10
   cp(11)  # ValueError

-  列表元组

.. code:: python

   cl = collection(is_type(int))  # 判断是否为int构成的list
   cl([1, 2, 3])     # [1, 2, 3]
   cl([1, 2, None])  # TypeError

   co = contains(2)  # 判断是否包含2
   co([1, 2, 3])  # [1, 2, 3]
   co([1, 3, 3])  # ValueError

-  字典映射

.. code:: python

   itt = item('a')  # 检查并提取'a'字段
   itt({'a': 1, 'b': 2})  # 1
   itt({'a': 2})          # 2
   itt({'aa': 2})         # KeyError

   dt = dict_(  # 构建dict格式数据
       a=item('b'),
       b=item('a'),
   )
   dt({'a': 1, 'b': 2})   # {'a': 2, 'b': 1}
   dt({'a': 2, 'bb': 3})  # KeyError

-  逻辑组合（与、或、传递）

.. code:: python

   iit = is_type(int) & interval(1, 5, right_ok=False)  # 验证是否为在[1, 5)的整数
   iit(1)    # 1
   iit(4)    # 4
   iit(5)    # ValueError
   iit(-1)   # ValueError
   iit(3.8)  # TypeError

   iit = interval(None, -1) | interval(1, None)  # 验证是否为(-inf, -1] | [1, +inf)
   iit(-2)  # -2
   iit(-1)  # -1
   iit(0)   # ValueError
   iit(1)   # 1
   iit(2)   # 2

   iit = to_type(float) >> (interval(None, -1) | interval(1, None))  # 是否在转为float后在区间(-inf, -1] | [1, +inf)上
   iit(1)     # 1.0
   iit('-1')  # -1.0
   iit('0')   # ValueError （注意，这里是ValueError，不是TypeError）

-  基础组件

.. code:: python

   kp = keep()  # 恒真，且保留原值
   kp(1)     # 1
   kp(None)  # None

   r = raw(233)  # 恒真，常量
   r(1)     # 233
   r(None)  # 233

   r = optional(is_type(int) | is_type(float))  # 可选类型，等价于int, float, None三选一
   r(1)      # 1
   r(1.2)    # 1.2
   r(None)   # None
   r('str')  # TypeError

   ck = check_only(to_type(float) >> plus(2))  # 转换值还原
   ck(1)    # 1
   ck(2.2)  # 2.2

   ckx = to_type(float) >> plus(2)  # 如果不加还原的效果
   ck(1)    # 3.0
   ck(2.2)  # 4.2

-  norm机制（用于支持中等复杂数学运算）

.. code:: python

   mt = norm(keep()) * (norm(keep()) + 1) - 10 / norm(keep())  # 计算x * (x + 1) - 10 / x
   mt(1)    # -8
   mt(3.5)  # 12.8929

   tt = Loader(mt) >> interval(None, 10)  # 判断x * (x + 1) - 10 / x是否在区间(-inf, 10]上
   tt(1)    # -8
   tt(3.5)  # ValueError

-  normfunc机制（用于支持高复杂数学计算或不可控计算逻辑）

.. code:: python

   def _calculate(x, y):
       return x ** (1 / y)

   @normfunc
   def _calculate2(x, y):
       return x / (1 + y)

   nf = normfunc(_calculate)(norm(item('a')), norm(item('b')))  # 计算 a ** (1 / b)
   nf({'a': 3, 'b': 7})  # 1.1699

   nf2 = _calculate2(norm(item('a')) - 1, norm(item('b')))  # 计算(a - 1) / (1 + b)
   nf2({'a': 3, 'b': 7})  # 0.25

.. _header-n98:

常见问题
--------

.. _header-n100:

Q：loader和norm的区别和关系是什么？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：这两者的差异在于：

-  loader专注于逻辑的构建、转换与验证

-  norm专注于数学计算，尤其指数学计算逻辑的构建

其中，通过\ ``Loader``\ 函数可以将任意类型（包括norm）转为loader，而\ ``norm``\ 函数可以将loader类型转为norm。请注意在实际使用的时候，为了避免歧义，\ **当loader和norm放在一块使用的时候，请注意加上\ ``norm``\ 或者\ ``Loader``\ 函数以明确身份，以避免系统使用预期外的运算重载。**


Q：感觉loader有些时候写起来太长了，而且都是重复的，有没有更好的解决方案？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A：很简单，可以考虑重用。例如

.. code:: python

   l1 = dict_(
       a=item('a') >> item('b') >> item('c'),
       b=item('a') >> item('b') >> item('d'),
   )

这样的写法，可以简化为

.. code:: python

   find_ab = item('a') >> item('b')
   l1 = dict_(
       a=find_ab >> item('c'),
       b=find_ab >> item('d'),
   )

**loader进行逻辑组合的原理，是基于两个loader，计算得出一个新的loader** ，故可以支持重用。

实际上操作中，也更建议各位使用者充分利用这一特性，使得代码更加优美。
