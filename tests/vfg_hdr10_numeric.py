"""CPU reference checks for the HDR10 VFG P010/RGB10A2 conversion.

These are deterministic mathematical tests (ITU-R BT.2020 NCL), NOT a GPU
test or proof of GPU numerical equivalence. Device validation requires RTX.
"""
import pathlib
import random
import unittest

KR, KG, KB = 0.2627, 0.6780, 0.0593


def clip(x):
    return max(0.0, min(1.0, x))


def q10(x):
    return round(clip(x) * 1023)


def range10(v, center, scale, lo, hi):
    return max(lo, min(hi, round(center + scale * v)))


def reference_rgb_to_yuv(r, g, b):
    y = KR * r + KG * g + KB * b
    return (range10(y, 64, 876, 64, 940),
            range10((b-y)/(2*(1-KB)), 512, 896, 64, 960),
            range10((r-y)/(2*(1-KR)), 512, 896, 64, 960))


def p010_to_rgb10a2(y, cb, cr):
    y = (y - 64) / 876
    u = (cb - 512) / 896
    v = (cr - 512) / 896
    r = q10(y + 1.4746 * v)
    g = q10(y - 0.16455312684 * u - 0.57135312684 * v)
    b = q10(y + 1.8814 * u)
    return r | (g << 10) | (b << 20) | (3 << 30)


def rgb10a2_to_yuv(packed):
    r, g, b = ((packed >> shift & 1023) / 1023 for shift in (0, 10, 20))
    y = KR*r + KG*g + KB*b
    return y, (b-y)/1.8814, (r-y)/1.4746


def rgb10a2_to_p010_block(packed4):
    vals = [rgb10a2_to_yuv(p) for p in packed4]
    ys = [range10(v[0], 64, 876, 64, 940) for v in vals]
    cb = range10(sum(v[1] for v in vals)/4, 512, 896, 64, 960)
    cr = range10(sum(v[2] for v in vals)/4, 512, 896, 64, 960)
    return ys, cb, cr


class ConversionTest(unittest.TestCase):
    def test_packed_rgb_bits_and_neutral(self):
        self.assertEqual(p010_to_rgb10a2(64,512,512), 0xC0000000)
        self.assertEqual(p010_to_rgb10a2(940,512,512), 0xFFFFFFFF)
        for y in [64,65,128,502,850,940]:
            words = [p010_to_rgb10a2(y,512,512)]*4
            ys,cb,cr = rgb10a2_to_p010_block(words)
            self.assertLessEqual(abs(ys[0]-y),1)
            self.assertEqual((cb,cr),(512,512))

    def test_bt2020_matrix_anchors(self):
        self.assertAlmostEqual(2*(1-KR),1.4746,places=7)
        self.assertAlmostEqual(2*(1-KB),1.8814,places=7)
        self.assertAlmostEqual(2*KB*(1-KB)/KG,0.16455312684,delta=1e-9)
        self.assertAlmostEqual(2*KR*(1-KR)/KG,0.57135312684,delta=1e-9)
        for rgb in [(0,0,0),(1,1,1),(1,0,0),(0,1,0),(0,0,1)]:
            packed = p010_to_rgb10a2(*reference_rgb_to_yuv(*rgb))
            decoded = [(packed >> shift & 1023) / 1023 for shift in (0,10,20)]
            self.assertLess(max(abs(a-b) for a,b in zip(rgb,decoded)),0.003)

    def test_fifty_thousand_in_gamut_roundtrips(self):
        rng = random.Random(20201116)
        maxima = [0,0,0]
        for _ in range(50000):
            src = reference_rgb_to_yuv(*(rng.randrange(1024)/1023 for i in range(3)))
            dst_luma,cb,cr = rgb10a2_to_p010_block([p010_to_rgb10a2(*src)]*4)
            dst = (dst_luma[0],cb,cr)
            for i in range(3):
                maxima[i] = max(maxima[i],abs(src[i]-dst[i]))
        self.assertLessEqual(max(maxima),1, f"max code errors Y,Cb,Cr={maxima}")
        print("50k in-gamut max 10-bit code errors [Y,Cb,Cr]:",maxima)

    def test_2x2_chroma_reconstruction(self):
        rgb = [(1,0,0),(0,1,0),(0,0,1),(1,0,0)]
        words = [p010_to_rgb10a2(*reference_rgb_to_yuv(*c)) for c in rgb]
        ys,cb,cr = rgb10a2_to_p010_block(words)
        self.assertEqual(len(ys),4)
        self.assertTrue(all(64 <= y <= 940 for y in ys))
        self.assertTrue(64 <= cb <= 960 and 64 <= cr <= 960)

    def test_kernel_source_constants(self):
        kernel = pathlib.Path(__file__).resolve().parents[1] / "NVEncCore" / "NVEncFilterCrop.cu"
        text = kernel.read_text(encoding="utf-8-sig")
        for fragment in [
            "runNvvfxP010ToRGB10A2", "runNvvfxRGB10A2ToP010",
            "yp + 1.4746f * v", "yp + 1.8814f * u",
            "yp - 0.16455312684f * u - 0.57135312684f * v",
            "(3u << 30)", "yRow[x] >> 6", "outUV[bx * 2 + 1]"
        ]:
            self.assertIn(fragment,text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
