armor
===================

armor
-----------------

BaseArmor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor.BaseArmor
    :members: __init__, forward, mode, state_dict, load_state_dict, reset

Armor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor.Armor
    :members: __init__, __getattr__


armor_plugin
----------------------

IArmorPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.IArmorPlugin
    :members: register


IArmorStatefulPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.IArmorStatefulPlugin
    :members: __init__, reset

GradHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.GradHelper
    :members: register

HiddenStateHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.HiddenStateHelper
    :members: register

ArgmaxSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.ArgmaxSampleHelper
    :members: register

MultinomialSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.MultinomialSampleHelper
    :members: register

EpsGreedySampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.EpsGreedySampleHelper
    :members: register

ActionNoiseHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.ActionNoiseHelper
    :members: register, __init__, add_noise, reset

TargetNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.TargetNetworkHelper
    :members: register, __init__, update, reset

TeacherNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.armor.armor_plugin.TeacherNetworkHelper
    :members: register


register_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.armor.armor_plugin.register_plugin


add_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.armor.armor_plugin.add_plugin
