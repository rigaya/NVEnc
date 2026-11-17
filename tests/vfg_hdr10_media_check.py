"""Verify native VFG output frame count, cadence, color signaling and HDR SEI.

Requires ffprobe (FFmpeg) on PATH. Example:
python tests/vfg_hdr10_media_check.py --source AV1_HDR.mkv \
    --output HDR_VFG_Medium_600.mkv --mode hdr10 --input-frames 600

Checking frame metadata does NOT prove the CUDA kernels' numerical accuracy.
"""
import argparse
from fractions import Fraction
import json
import shutil
import subprocess
import sys


def probe(path, executable="ffprobe"):
    binary = shutil.which(executable) or executable
    cmd = [binary, "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_streams", "-of", "json", path]
    data = json.loads(subprocess.check_output(cmd, text=True, encoding="utf-8"))
    if not data.get("streams"):
        raise AssertionError(f"Missing video stream: {path}")
    stream = data["streams"][0]
    frames = json.loads(subprocess.check_output([
        binary, "-v", "error", "-select_streams", "v:0",
        "-read_intervals", "0%+#8", "-show_frames",
        "-show_entries", "frame=side_data_list", "-of", "json", path
    ], text=True, encoding="utf-8"))
    sides = stream.get("side_data_list", [])[:]
    for frame in frames.get("frames", []):
        sides.extend(frame.get("side_data_list", []))
    return stream, sides


def seis(sides):
    found = {}
    for item in sides:
        name = item.get("side_data_type", "").lower()
        if "mastering display" in name:
            found["mastering"] = item
        if "content light" in name:
            found["content_light"] = item
    return found


def verify(source, output, source_sei, output_sei, mode, input_frames, multiplier):
    formats = {"hdr10": ("bt2020nc", "bt2020", "smpte2084"),
               "sdr": ("bt709", "bt709", "bt709")}
    matrix, primaries, transfer = formats[mode]
    for key, expected in (("color_space", matrix),
                          ("color_primaries", primaries),
                          ("color_transfer", transfer)):
        actual = output.get(key)
        assert actual == expected, f"{key}: {actual!r} != {expected!r}"
    pix_fmt = output.get("pix_fmt", "")
    assert ("10" in pix_fmt) == (mode == "hdr10"), f"unexpected pix_fmt: {pix_fmt}"
    count = int(output.get("nb_read_frames", "0"))
    expected_count = input_frames * multiplier - (multiplier - 1)
    assert count == expected_count, f"frame count: {count} != {expected_count}"
    rate_in = Fraction(source["avg_frame_rate"])
    rate_out = Fraction(output["avg_frame_rate"])
    assert rate_in > 0 and abs(rate_out/rate_in - multiplier) < Fraction(1, 10000), \
        f"rate ratio: {rate_out}/{rate_in} != {multiplier}"
    print(f"PASS {mode} frames={count}, fps={float(rate_out):.5f}, pix_fmt={pix_fmt}, VUI={matrix}/{primaries}/{transfer}")
    if mode == "hdr10":
        src_sei, dst_sei = seis(source_sei), seis(output_sei)
        common_fields = {
            "mastering": ("red_x", "red_y", "green_x", "green_y", "blue_x", "blue_y",
                          "white_point_x", "white_point_y", "min_luminance", "max_luminance"),
            "content_light": ("max_content", "max_average", "max_content_light_level",
                              "max_frame_average_light_level"),
        }
        for category, fields in common_fields.items():
            if category not in src_sei:
                print(f"UNVERIFIED {category}: not detected in source")
                continue
            assert category in dst_sei, f"lost HDR {category} metadata"
            for field in fields:
                if field in src_sei[category]:
                    assert field in dst_sei[category], f"missing HDR metadata field {category}.{field}"
                    a = src_sei[category][field]
                    b = dst_sei[category][field]
                    try:
                        same = abs(Fraction(str(a)) - Fraction(str(b))) <= Fraction(1, 10000)
                    except (ValueError, ZeroDivisionError):
                        same = str(a) == str(b)
                    assert same, f"changed HDR {category}.{field}: {a} -> {b}"
            print(f"PASS HDR {category} metadata retained")


def self_test():
    src = {"avg_frame_rate": "30000/1001"}
    dst = {"avg_frame_rate": "60000/1001", "pix_fmt": "yuv420p10le",
           "color_space": "bt2020nc", "color_primaries": "bt2020",
           "color_transfer": "smpte2084", "nb_read_frames": "11"}
    verify(src, dst, [], [], "hdr10", 6, 2)
    for changed in ({"color_transfer": "bt709"}, {"nb_read_frames": "12"},
                    {"pix_fmt": "yuv420p"}, {"avg_frame_rate": "30000/1001"}):
        invalid = {**dst, **changed}
        try:
            verify(src, invalid, [], [], "hdr10", 6, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"metadata verifier accepted invalid output: {changed}")
    dst_sdr = {**dst, "pix_fmt": "yuv420p", "color_space": "bt709",
               "color_primaries": "bt709", "color_transfer": "bt709"}
    verify(src, dst_sdr, [], [], "sdr", 6, 2)
    print("PASS metadata validator self-tests")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--source")
    p.add_argument("--output")
    p.add_argument("--mode", choices=["hdr10", "sdr"])
    p.add_argument("--input-frames", type=int)
    p.add_argument("--multiplier", type=int, default=2)
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if not all([args.source, args.output, args.mode, args.input_frames]):
                p.error("--source --output --mode --input-frames are required")
            src, src_side = probe(args.source, args.ffprobe)
            dst, dst_side = probe(args.output, args.ffprobe)
            verify(src, dst, src_side, dst_side, args.mode, args.input_frames, args.multiplier)
    except (AssertionError, OSError, subprocess.CalledProcessError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
