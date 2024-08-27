import ctypes
import os

# Load the compiled shared library
if os.name == "nt":
    # Windows
    lib = ctypes.CDLL('./encrypt.dll')
else:
    # Linux/MacOS
    lib = ctypes.CDLL('./encrypt.so')

# --------------------------------------------------------------------------------------- DES
# Define the function prototype for des
lib.des.restype = ctypes.c_uint64  # Return type is uint64_t
lib.des.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]  # Argument types

# Call the function
input_value = 0xAAAAAAAAAAAAAAAA  # Example input value
key = 0x0000000000000000  # Example key
rounds = 16  # Example number of rounds

result = lib.des(input_value, key, rounds)
print(f"Result of des({hex(input_value)}, {hex(key)}, {rounds}) = {hex(result)}")



# --------------------------------------------------------------------------------------- AES
# Define the key and plaintext as byte arrays
key = (ctypes.c_ubyte * 16)(*b'kkkkeeeeyyyy....')  # Equivalent to unsigned char key[16] in C #16byte or 128bit
plaintext = (ctypes.c_ubyte * 16)(*b'abcdef1234567890')  # Equivalent to unsigned char plaintext[16] in C
ciphertext = (ctypes.c_ubyte * 16)()  # Allocate space for the ciphertext output (16, 24 or 32 to decide if 128, 192 or 256bit key)

# Define the keySize enum value for SIZE_16
SIZE_16 = 16  # Assuming SIZE_16 corresponds to 16 (16, 24 or 32 to decide if 128, 192 or 256bit key)

# Define the function prototype for aes_encrypt
lib.aes_encrypt.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int)
lib.aes_encrypt.restype = None

# Call the aes_encrypt function
lib.aes_encrypt(plaintext, ciphertext, key, SIZE_16, 2)  # The last parameter is assumed to be some mode or rounds, set as 2 for now

# Print the ciphertext in HEX format
print("Ciphertext (HEX format):")
print(" ".join(f"{b:02x}" for b in ciphertext))