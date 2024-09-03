import numpy as np
import ctypes
import os

class BasicCrypto:
    def __init__(self, rounds, seed=42, nbytes=4):
        self.s_box = np.asarray([
            0xa3, 0xd7, 0x09, 0x83, 0xf8, 0x48, 0xf6, 0xf4, 
            0xb3, 0x21, 0x15, 0x78, 0x99, 0xb1, 0xaf, 0xf9, 
            0xe7, 0x2d, 0x4d, 0x8a, 0xce, 0x4c, 0xca, 0x2e, 
            0x52, 0x95, 0xd9, 0x1e, 0x4e, 0x38, 0x44, 0x28, 
            0x0a, 0xdf, 0x02, 0xa0, 0x17, 0xf1, 0x60, 0x68, 
            0x12, 0xb7, 0x7a, 0xc3, 0xe9, 0xfa, 0x3d, 0x53, 
            0x96, 0x84, 0x6b, 0xba, 0xf2, 0x63, 0x9a, 0x19, 
            0x7c, 0xae, 0xe5, 0xf5, 0xf7, 0x16, 0x6a, 0xa2, 
            0x39, 0xb6, 0x7b, 0x0f, 0xc1, 0x93, 0x81, 0x1b, 
            0xee, 0xb4, 0x1a, 0xea, 0xd0, 0x91, 0x2f, 0xb8, 
            0x55, 0xb9, 0xda, 0x85, 0x3f, 0x41, 0xbf, 0xe0, 
            0x5a, 0x58, 0x80, 0x5f, 0x66, 0x0b, 0xd8, 0x90, 
            0x35, 0xd5, 0xc0, 0xa7, 0x33, 0x06, 0x65, 0x69, 
            0x45, 0x00, 0x94, 0x56, 0x6d, 0x98, 0x9b, 0x76, 
            0x97, 0xfc, 0xb2, 0xc2, 0xb0, 0xfe, 0xdb, 0x20, 
            0xe1, 0xeb, 0xd6, 0xe4, 0xdd, 0x47, 0x4a, 0x1d, 
            0x42, 0xed, 0x9e, 0x6e, 0x49, 0x3c, 0xcd, 0x43, 
            0x27, 0xd2, 0x07, 0xd4, 0xde, 0xc7, 0x67, 0x18, 
            0x89, 0xcb, 0x30, 0x1f, 0x8d, 0xc6, 0x8f, 0xaa, 
            0xc8, 0x74, 0xdc, 0xc9, 0x5d, 0x5c, 0x31, 0xa4, 
            0x70, 0x88, 0x61, 0x2c, 0x9f, 0x0d, 0x2b, 0x87, 
            0x50, 0x82, 0x54, 0x64, 0x26, 0x7d, 0x03, 0x40, 
            0x34, 0x4b, 0x1c, 0x73, 0xd1, 0xc4, 0xfd, 0x3b, 
            0xcc, 0xfb, 0x7f, 0xab, 0xe6, 0x3e, 0x5b, 0xa5, 
            0xad, 0x04, 0x23, 0x9c, 0x14, 0x51, 0x22, 0xf0, 
            0x29, 0x79, 0x71, 0x7e, 0xff, 0x8c, 0x0e, 0xe2, 
            0x0c, 0xef, 0xbc, 0x72, 0x75, 0x6f, 0x37, 0xa1, 
            0xec, 0xd3, 0x8e, 0x62, 0x8b, 0x86, 0x10, 0xe8, 
            0x08, 0x77, 0x11, 0xbe, 0x92, 0x4f, 0x24, 0xc5, 
            0x32, 0x36, 0x9d, 0xcf, 0xf3, 0xa6, 0xbb, 0xac, 
            0x5e, 0x6c, 0xa9, 0x13, 0x57, 0x25, 0xb5, 0xe3, 
            0xbd, 0xa8, 0x3a, 0x01, 0x05, 0x59, 0x2a, 0x46], dtype=np.uint8)
        
        np.random.seed(seed)

        self.rounds = rounds
        self.nbytes = nbytes

        self.shifts = np.asarray([(nbytes - i - 1) * 8 for i in range(nbytes)], dtype=np.uint8)[None]
        self.seeds = np.random.randint(0, 256, size=(1, nbytes), dtype=np.uint8)

        perm_list = np.random.permutation(nbytes * 8)
        self.permutation_matrix = np.zeros((nbytes * 8,nbytes * 8), dtype=np.uint8)

        for i in range(nbytes * 8):
            self.permutation_matrix[i, perm_list[i]] = 1

    def round(self, i):
        #S-BOX
        i = np.bitwise_xor(i, self.seeds) 
        res = self.s_box[i]

        #P-BOX
        bits = np.unpackbits(res).reshape(i.shape[0], self.nbytes*8) @ self.permutation_matrix
        y = np.packbits(bits).reshape(i.shape[0], self.nbytes)
        return y
    
    def sample(self, i):
        for _ in range(self.rounds):
            i = self.round(i)
        return i



class DES:
    def __init__(self, rounds, seed=42, nbytes=8):
        if nbytes != 8:
            raise ValueError("DES only works with 64 bits")
        
        self.rounds = rounds
        self.nbytes = nbytes
        np.random.seed(seed)
        self.seed = np.random.randint(0, 2**64, dtype=np.uint64)

        # Load the compiled shared library
        if os.name == "nt":
            # Windows
            self.lib = ctypes.CDLL('./encrypt.dll')
        else:
            # Linux/MacOS
            self.lib = ctypes.CDLL('./encrypt.so')

        # Define the function prototype for des
        self.lib.des.restype = ctypes.c_uint64  # Return type is uint64_t
        self.lib.des.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]  # Argument types

    def sample(self, i):
        return np.asarray([list(self.lib.des(v.item(), self.seed, self.rounds).to_bytes(8, "big")) for v in i.view(np.uint64)], dtype=np.uint8)
    

class AES:
    def __init__(self, rounds, seed=42, nbytes=16, keysize=16):
        if nbytes != 16:
            raise ValueError("AES only works with 128 bits")
        
        self.rounds = rounds
        self.nbytes = nbytes
        np.random.seed(seed)
        self.key = (ctypes.c_ubyte * keysize)(*np.random.randint(0, 256, size=keysize, dtype=np.uint8))

        # Load the compiled shared library
        if os.name == "nt":
            # Windows
            self.lib = ctypes.CDLL('./encrypt.dll')
        else:
            # Linux/MacOS
            self.lib = ctypes.CDLL('./encrypt.so')

        # Define the keySize enum value for SIZE_16
        self.key_size = keysize  # Assuming SIZE_16 corresponds to 16 (16, 24 or 32 to decide if 128, 192 or 256bit key)

        # Define the function prototype for aes_encrypt
        self.lib.aes_encrypt.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int)
        self.lib.aes_encrypt.restype = None

    def sample(self, i):
        ciphertext = (ctypes.c_ubyte * 16)()
        res = []
        for v in i:
            plaintext = (ctypes.c_ubyte * 16)(*v) 
            print(*plaintext)
            print(*self.key)
            self.lib.aes_encrypt(plaintext, ciphertext, self.key, self.key_size, self.rounds)
            res.append(ciphertext)

        return np.asarray(res, dtype=np.uint8)
    
class ASCON:
    def __init__(self, rounds, seed=42, nbytes=8):
        if nbytes != 8:
            raise ValueError("ASCON only works with 128 bits")
        
        self.rounds = rounds
        self.nbytes = nbytes
        np.random.seed(seed)
        self.key = (ctypes.c_ubyte * 16)(*np.random.randint(0, 256, size=16, dtype=np.uint8))

        # Load the compiled shared library
        if os.name == "nt":
            # Windows
            self.lib = ctypes.CDLL('./encrypt.dll')
        else:
            # Linux/MacOS
            self.lib = ctypes.CDLL('./encrypt.so')


        # Define the function prototype for aes_encrypt
        self.lib.ascon128_encrypt.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int)
        self.lib.ascon128_encrypt.restype = None

    def sample(self, i):
        ciphertext = (ctypes.c_ubyte * 8)()
        res = []
        for v in i:
            plaintext = (ctypes.c_ubyte * 8)(*v) 
            self.lib.ascon128_encrypt(ciphertext, plaintext, self.key, self.rounds)
            res.append(ciphertext)

        return np.asarray(res, dtype=np.uint8)
        

if __name__ == "__main__":
    # Example usage

    toy = BasicCrypto(2, nbytes=4)
    print(toy.sample(np.array([[1,0,0,0]], dtype=np.uint8)))

    toy = BasicCrypto(2, nbytes=8)
    print(toy.sample(np.array([[1,0,0,0,0,0,0,0]], dtype=np.uint8)))

    toy = BasicCrypto(2, nbytes=16)
    print(toy.sample(np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)))


    des = DES(2)
    print(des.sample(np.array([[1,0,0,0,0,0,0,0]], dtype=np.uint8)))

    aes = AES(2)
    print(aes.sample(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]], dtype=np.uint8)))

    ascon = ASCON(2)
    print(ascon.sample(np.array([[1,2,3,4,5,6,7,8]], dtype=np.uint8)))