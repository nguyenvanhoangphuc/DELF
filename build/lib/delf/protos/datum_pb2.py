# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: delf/protos/datum.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='delf/protos/datum.proto',
  package='delf.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x17\x64\x65lf/protos/datum.proto\x12\x0b\x64\x65lf.protos\"\x1d\n\nDatumShape\x12\x0f\n\x03\x64im\x18\x01 \x03(\x03\x42\x02\x10\x01\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1f\n\nUint32List\x12\x11\n\x05value\x18\x01 \x03(\rB\x02\x10\x01\"\xa0\x01\n\nDatumProto\x12&\n\x05shape\x18\x01 \x01(\x0b\x32\x17.delf.protos.DatumShape\x12,\n\nfloat_list\x18\x02 \x01(\x0b\x32\x16.delf.protos.FloatListH\x00\x12.\n\x0buint32_list\x18\x03 \x01(\x0b\x32\x17.delf.protos.Uint32ListH\x00\x42\x0c\n\nkind_oneof\"a\n\x0e\x44\x61tumPairProto\x12&\n\x05\x66irst\x18\x01 \x01(\x0b\x32\x17.delf.protos.DatumProto\x12\'\n\x06second\x18\x02 \x01(\x0b\x32\x17.delf.protos.DatumProto')
)




_DATUMSHAPE = _descriptor.Descriptor(
  name='DatumShape',
  full_name='delf.protos.DatumShape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dim', full_name='delf.protos.DatumShape.dim', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=69,
)


_FLOATLIST = _descriptor.Descriptor(
  name='FloatList',
  full_name='delf.protos.FloatList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='delf.protos.FloatList.value', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=101,
)


_UINT32LIST = _descriptor.Descriptor(
  name='Uint32List',
  full_name='delf.protos.Uint32List',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='delf.protos.Uint32List.value', index=0,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=103,
  serialized_end=134,
)


_DATUMPROTO = _descriptor.Descriptor(
  name='DatumProto',
  full_name='delf.protos.DatumProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='delf.protos.DatumProto.shape', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_list', full_name='delf.protos.DatumProto.float_list', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='uint32_list', full_name='delf.protos.DatumProto.uint32_list', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind_oneof', full_name='delf.protos.DatumProto.kind_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=137,
  serialized_end=297,
)


_DATUMPAIRPROTO = _descriptor.Descriptor(
  name='DatumPairProto',
  full_name='delf.protos.DatumPairProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='first', full_name='delf.protos.DatumPairProto.first', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second', full_name='delf.protos.DatumPairProto.second', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=299,
  serialized_end=396,
)

_DATUMPROTO.fields_by_name['shape'].message_type = _DATUMSHAPE
_DATUMPROTO.fields_by_name['float_list'].message_type = _FLOATLIST
_DATUMPROTO.fields_by_name['uint32_list'].message_type = _UINT32LIST
_DATUMPROTO.oneofs_by_name['kind_oneof'].fields.append(
  _DATUMPROTO.fields_by_name['float_list'])
_DATUMPROTO.fields_by_name['float_list'].containing_oneof = _DATUMPROTO.oneofs_by_name['kind_oneof']
_DATUMPROTO.oneofs_by_name['kind_oneof'].fields.append(
  _DATUMPROTO.fields_by_name['uint32_list'])
_DATUMPROTO.fields_by_name['uint32_list'].containing_oneof = _DATUMPROTO.oneofs_by_name['kind_oneof']
_DATUMPAIRPROTO.fields_by_name['first'].message_type = _DATUMPROTO
_DATUMPAIRPROTO.fields_by_name['second'].message_type = _DATUMPROTO
DESCRIPTOR.message_types_by_name['DatumShape'] = _DATUMSHAPE
DESCRIPTOR.message_types_by_name['FloatList'] = _FLOATLIST
DESCRIPTOR.message_types_by_name['Uint32List'] = _UINT32LIST
DESCRIPTOR.message_types_by_name['DatumProto'] = _DATUMPROTO
DESCRIPTOR.message_types_by_name['DatumPairProto'] = _DATUMPAIRPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DatumShape = _reflection.GeneratedProtocolMessageType('DatumShape', (_message.Message,), dict(
  DESCRIPTOR = _DATUMSHAPE,
  __module__ = 'delf.protos.datum_pb2'
  # @@protoc_insertion_point(class_scope:delf.protos.DatumShape)
  ))
_sym_db.RegisterMessage(DatumShape)

FloatList = _reflection.GeneratedProtocolMessageType('FloatList', (_message.Message,), dict(
  DESCRIPTOR = _FLOATLIST,
  __module__ = 'delf.protos.datum_pb2'
  # @@protoc_insertion_point(class_scope:delf.protos.FloatList)
  ))
_sym_db.RegisterMessage(FloatList)

Uint32List = _reflection.GeneratedProtocolMessageType('Uint32List', (_message.Message,), dict(
  DESCRIPTOR = _UINT32LIST,
  __module__ = 'delf.protos.datum_pb2'
  # @@protoc_insertion_point(class_scope:delf.protos.Uint32List)
  ))
_sym_db.RegisterMessage(Uint32List)

DatumProto = _reflection.GeneratedProtocolMessageType('DatumProto', (_message.Message,), dict(
  DESCRIPTOR = _DATUMPROTO,
  __module__ = 'delf.protos.datum_pb2'
  # @@protoc_insertion_point(class_scope:delf.protos.DatumProto)
  ))
_sym_db.RegisterMessage(DatumProto)

DatumPairProto = _reflection.GeneratedProtocolMessageType('DatumPairProto', (_message.Message,), dict(
  DESCRIPTOR = _DATUMPAIRPROTO,
  __module__ = 'delf.protos.datum_pb2'
  # @@protoc_insertion_point(class_scope:delf.protos.DatumPairProto)
  ))
_sym_db.RegisterMessage(DatumPairProto)


_DATUMSHAPE.fields_by_name['dim'].has_options = True
_DATUMSHAPE.fields_by_name['dim']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_FLOATLIST.fields_by_name['value'].has_options = True
_FLOATLIST.fields_by_name['value']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_UINT32LIST.fields_by_name['value'].has_options = True
_UINT32LIST.fields_by_name['value']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
