import hashlib

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def count_functions(filepath):
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if 'void crc_' in line or 'uint8_t crc_' in line or 'uint16_t crc_' in line or 'uint32_t crc_' in line:
                if '(' in line:
                    count += 1
    return count

print("=== Original ardupilot_crc.cpp ===")
print(f"File hash: {file_hash('ardupilot_crc.cpp')}")
print(f"Line count: {sum(1 for _ in open('ardupilot_crc.cpp'))}")
print(f"Function count: {count_functions('ardupilot_crc.cpp')}")

print("\n=== STUNIR generated.cpp ===")
print(f"File hash: {file_hash('stunir_runs/ardupilot_full/generated.cpp')}")
print(f"Line count: {sum(1 for _ in open('stunir_runs/ardupilot_full/generated.cpp'))}")
print(f"Function count: {count_functions('stunir_runs/ardupilot_full/generated.cpp')}")

print("\n=== Function signatures comparison ===")
print("\nOriginal functions:")
with open('ardupilot_crc.cpp', 'r') as f:
    for line in f:
        if 'crc_' in line and '(' in line and not line.strip().startswith('//'):
            print(f"  {line.strip()}")

print("\nGenerated functions:")
with open('stunir_runs/ardupilot_full/generated.cpp', 'r') as f:
    for line in f:
        if 'crc_' in line and '(' in line:
            print(f"  {line.strip()}")