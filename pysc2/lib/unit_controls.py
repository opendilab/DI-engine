import numpy as np
import collections

class UnitFlags(collections.namedtuple("UnitFlags", [
    "is_selected", "is_on_screen", "is_blip",
    "is_powered", "is_flying", "is_burrowed"])):
  """The set of unit flags"""
  __slots__ = ()

  @property
  def np_array(self):
    return np.array([self.is_selected,
        self.is_on_screen,
        self.is_blip,
        self.is_powered,
        self.is_flying,
        self.is_burrowed], dtype=np.bool_)

class UnitFloatAttr(collections.namedtuple("UnitFloats", [
    "pos_x", "pos_y", "pos_z",
    "facing", "radius", "build_progress",
    "detect_range", "radar_range",
    "health", "health_max",
    "energy", "energy_max", "weapon_cooldown"])):
  """The set of unit flags"""
  __slots__ = ()

  @property
  def np_array(self):
    return np.array([
        self.pos_x,
        self.pos_y,
        self.pos_z,
        self.facing,
        self.radius,
        self.build_progress,
        self.detect_range,
        self.radar_range,
        self.health,
        self.health_max,
        self.energy,
        self.energy_max,
        self.weapon_cooldown], dtype=float32)

class UnitIntAttr(collections.namedtuple("UnitInt64", [
    "tag", "unit_type", "owner", 
    "display_type", "alliance", "cloak",
    "mineral_contents", "vespene_contents", 
    "add_on_tag", "engaged_target_tag",
    "cargo_space_taken", "cargo_space_max", 
    "assigned_harvesters", "ideal_harvesters"])):
  """The set of unit attr"""
  __slots__ = ()

  @property
  def np_array(self):
    return np.array([
        self.tag,
        self.unit_type,
        self.owner,
        self.display_type,
        self.alliance,
        self.cloak,
        self.mineral_contents,
        self.vespene_contents,
        self.add_on_tag,
        self.engaged_target_tag,
        self.cargo_space_taken,
        self.cargo_space_max,
        self.assigned_harvesters,
        self.ideal_harvesters], dtype=int64)

class UnitOrders(collections.namedtuple("UnitOrders", [
    "ability_id", "target_tag", "target_pos_x", 
    "target_pos_y", "target_pos_z", "progress"])):
  """the set of unit orders  """
  __slots__ = ()

  @property
  def np_array(self):
    return np.array([
        self.ability_id,
        self.target_tag,
        self.target_pos_x,
        self.target_pos_y,
        self.target_pos_z,
        self.progress
    ], dtype=float)

class PassengerUnit(collections.namedtuple("PassengerUnit", [
    "tag", "health", "health_max", "shield", "shield_max",
    "energy", "energy_max", "unit_type"])):
  __slots__ = ()

  @property
  def np_array(self):
    return np.array([
        self.tag,
        self.health,
        self.health_max,
        self.shield,
        self.shield_max,
        self.energy,
        self.energy_max,
        self.unit_type
    ], dtype=float)


class Unit(collections.namedtuple("unit", [
    "bool_attr", "int_attr", "float_attr", 
    "orders", "passengers", "buff_ids"])):
  """The definition of unit"""
  __slots__ = ()

  @property
  def tag(self):
    return self.int_attr.tag

  @property
  def unit_type(self):
    return self.int_attr.unit_type

  @property
  def bool_attrs(self):
    return self.bool_attr

  @property
  def int_attrs(self):
    return self.int_attr

  @property
  def float_attrs(self):
    return self.float_attr

  def __new__(cls, u=None):
    if u is None:
      return Exception("input unit is none")
    uflag = assemble_bool(u)
    ufloat = assemble_float(u)
    uint = assemble_int(u)
    uorders = assemble_orders(u)
    upassengers = assemble_passenger_unit(u)
    ubuff_ids = assemble_buff_ids(u)
    return super(Unit, cls).__new__(cls,
        bool_attr=uflag,
        int_attr=uint,
        float_attr=ufloat,
        orders=uorders,
        passengers=upassengers,
        buff_ids=ubuff_ids)

def assemble_bool(u):
  return UnitFlags(is_selected=u.is_selected, 
      is_on_screen=u.is_on_screen,
      is_blip=u.is_blip,
      is_powered=u.is_powered,
      is_flying=u.is_flying,
      is_burrowed=u.is_burrowed)

def assemble_float(u):
  return UnitFloatAttr(pos_x=u.pos.x,
      pos_y=u.pos.y,
      pos_z=u.pos.z,
      facing=u.facing,
      radius=u.radius,
      build_progress=u.build_progress,
      detect_range=u.detect_range,
      radar_range=u.radar_range,
      health=u.health,
      health_max=u.health_max,
      energy=u.energy,
      energy_max=u.energy_max,
      weapon_cooldown=u.weapon_cooldown)

def assemble_int(u):
  return UnitIntAttr(tag=u.tag,
      unit_type=u.unit_type,
      owner=u.owner,
      display_type=u.display_type,
      alliance=u.alliance,
      cloak=u.cloak,
      mineral_contents=u.mineral_contents,
      vespene_contents=u.vespene_contents,
      add_on_tag=u.add_on_tag,
      engaged_target_tag=u.engaged_target_tag,
      cargo_space_taken=u.cargo_space_taken,
      cargo_space_max=u.cargo_space_max,
      assigned_harvesters=u.assigned_harvesters,
      ideal_harvesters=u.ideal_harvesters)

def assemble_orders(u):
  orders = []
  for o in u.orders:
    uo = UnitOrders(ability_id=o.ability_id,
        target_tag=o.target_unit_tag,
        target_pos_x=o.target_world_space_pos.x,
        target_pos_y=o.target_world_space_pos.y,
        target_pos_z=o.target_world_space_pos.z,
        progress=o.progress)
    orders.append(uo)
  return orders

def assemble_passenger_unit(u):
  passengers = []
  for p in u.passengers:
    pu = PassengerUnit(tag=p.tag,
        health=p.health,
        health_max=p.health_max,
        shield=p.shield,
        shield_max=p.shield_max,
        energy=p.energy,
        energy_max=p.energy_max,
        unit_type=p.unit_type)
    passengers.append(pu)
  return passengers

def assemble_buff_ids(u):
  buff_ids = []
  for val in u.buff_ids:
    buff_ids.append(val)
  return buff_ids

