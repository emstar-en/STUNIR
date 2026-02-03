import json
import unittest

from tools.ir_bundle_v1 import normalize_json_value, dcbor_encode, make_ir_bundle_bytes, sha256_hex


class TestIrBundleV1Vectors(unittest.TestCase):
    def test_vectors(self):
        with open('tests/test_ir_bundle_v1_vectors.json', 'r', encoding='utf-8') as f:
            tv = json.load(f)

        for vec in tv['vectors']:
            cir_units = normalize_json_value(vec['cir_units'])

            cir_bytes = dcbor_encode(cir_units)
            self.assertEqual(sha256_hex(cir_bytes), vec['expected']['cir_sha256'])

            bundle_bytes = make_ir_bundle_bytes(cir_units)
            self.assertEqual(sha256_hex(bundle_bytes), vec['expected']['ir_bundle_sha256'])
            self.assertEqual(bundle_bytes.hex(), vec['expected']['ir_bundle_bytes_hex'])


if __name__ == '__main__':
    unittest.main()
