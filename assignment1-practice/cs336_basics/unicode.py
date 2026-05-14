print(repr(chr(0)))
print("test " + chr(0))

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
 return "".join([bytes([b]).decode("utf-8") for b in bytestring])

test_str="The cat sat on the 帽子"
uft8_encoded_string = test_str.encode("utf-8")
print(list(uft8_encoded_string))
print(decode_utf8_bytes_to_str_wrong(uft8_encoded_string))