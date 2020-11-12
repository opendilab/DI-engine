utils.log_helper
===================

log_helper
-----------------


build_logger
~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.build_logger


build_logger_naive
~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.utils.log_helper.build_logger_naive



get_default_logger
~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.get_default_logger




TextLogger
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.TextLogger
    :members: __init__, _create_logger, info, bug



TensorBoardLogger
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.TensorBoardLogger
    :members: __init__, add_scalar, add_text, add_scalars, add_histogram, add_figure, add_image, add_scalar_list, register_var, scalar_var_names

VariableRecord
~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.VariableRecord
    :members: __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text


AlphastarVarRecord
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.AlphastarVarRecord
    :members: register_var, _get_vars_text_1darray, _get_vars_tb_format_1darray

AverageMeter
~~~~~~~~~~~~~
.. automodule:: nervex.utils.log_helper.AverageMeter
    :members: __init__, reset, update

DistributionTimeImage
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.utils.log_helper.DistributionTimeImage
    :members: __init__, add_one_time_step, get_image


pretty_print
~~~~~~~~~~~~~~~~~~~

.. automodule:: nervex.utils.log_helper.pretty_print

