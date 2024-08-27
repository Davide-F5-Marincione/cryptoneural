import ctypes
import os

# Load the compiled shared library
if os.name == "nt":
    # Windows
    lib = ctypes.CDLL('./encrypt.dll')
else:
    # Linux/MacOS
    lib = ctypes.CDLL('./encrypt.so')

# Define the function prototype for des
lib.des.restype = ctypes.c_uint64  # Return type is uint64_t
lib.des.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]  # Argument types

# Call the function
input_value = 0xAAAAAAAAAAAAAAAA  # Example input value
key = 0x0000000000000000  # Example key
rounds = 16  # Example number of rounds

result = lib.des(input_value, key, rounds)
print(f"Result of des({hex(input_value)}, {hex(key)}, {rounds}) = {hex(result)}")
