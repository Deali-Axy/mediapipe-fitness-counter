from google.protobuf import wrappers_pb2 as _wrappers_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InputFrame(_message.Message):
    __slots__ = ["base64buffer"]
    BASE64BUFFER_FIELD_NUMBER: _ClassVar[int]
    base64buffer: str
    def __init__(self, base64buffer: _Optional[str] = ...) -> None: ...

class OutputFrame(_message.Message):
    __slots__ = ["base64buffer"]
    BASE64BUFFER_FIELD_NUMBER: _ClassVar[int]
    base64buffer: str
    def __init__(self, base64buffer: _Optional[str] = ...) -> None: ...
