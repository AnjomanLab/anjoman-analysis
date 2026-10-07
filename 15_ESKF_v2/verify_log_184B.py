#!/usr/bin/env python3
"""
verify_log.py — Decode and verify Anjoman binary log files.

Schema is auto-detected from the header's `version` field:
  - v1: 68-byte records (single UWB peer)
  - v2: 184-byte records (three UWB peers)
"""

import struct
import zlib
import sys
import pandas as pd


# ==============================================================================
# Record formats (little-endian)
# ==============================================================================

# ---- Version 1 (68 bytes) ----
RECORD_V1_FORMAT = "<I14fHBBf"
RECORD_V1_SIZE   = struct.calcsize(RECORD_V1_FORMAT)

# ---- Version 2 (184 bytes) ----
# Peer block: peerId(B) ldeErr(B) stdNoise(H) fpAmpl1(H) fpAmpl2(H)
#             cirPwr(H) rawDist(f) cleanDist(f) rssi(f) fpPower(f) respTemp(f)
PEER_BLOCK_FORMAT = "<BBHHHHfffff"
PEER_BLOCK_SIZE   = struct.calcsize(PEER_BLOCK_FORMAT)

# Full record:
#   t_ms                 I     4
#   x,y,heading,
#   v_cmd,omega_cmd,
#   rpm_l,rpm_r,
#   gyro_z,
#   imu_temp_c,imu_accel_x       10f  40
#   vbat,current_a,duty_l,duty_r  4f  16
#   target_rpm_l,target_rpm_r     2f   8
#   uwb[3]                       3×peer
#   rxpacc                       H     2
#   eskf_var_x/y/theta/bias      4f  16
#   tdma_frame_id                I     4
#   slot,sync_lost,maneuver_id,state  4B  4
RECORD_V2_FORMAT = (
    "<I"
    "10f"
    "4f"
    "2f"
    + "BBHHHHfffff" * 3
    + "H"
    "4f"
    "I"
    "4B"
)
RECORD_V2_SIZE = struct.calcsize(RECORD_V2_FORMAT)


# ==============================================================================
# Diagnostics: report sizes at import time
# ==============================================================================
print(f"[verify_log] PEER_BLOCK_SIZE = {PEER_BLOCK_SIZE} (expected 30)")
print(f"[verify_log] RECORD_V1_SIZE  = {RECORD_V1_SIZE} (expected 68)")
print(f"[verify_log] RECORD_V2_SIZE  = {RECORD_V2_SIZE} (expected 184)")

if PEER_BLOCK_SIZE != 30:
    print(f"  → format: {PEER_BLOCK_FORMAT!r}")
    print(f"  → sizes per field: "
          f"{[struct.calcsize(c) for c in ['B','B','H','H','H','H','f','f','f','f','f']]}")
    sys.exit(f"FATAL: PEER_BLOCK_SIZE is {PEER_BLOCK_SIZE}, expected 30")

if RECORD_V2_SIZE != 184:
    sys.exit(f"FATAL: RECORD_V2_SIZE is {RECORD_V2_SIZE}, expected 184")


# ==============================================================================
# Column names
# ==============================================================================
BASE_V1_COLS = [
    "t_ms",
    "x", "y", "heading",
    "v_cmd", "omega_cmd",
    "rpm_l", "rpm_r",
    "gyro_z",
    "vbat", "current_a",
    "uwb_raw", "uwb_clean",
    "uwb_rssi", "uwb_fp_power",
    "uwb_std_noise",
    "uwb_peer_id", "uwb_lde_err",
    "uwb_temp",
]


def v2_columns():
    cols = [
        "t_ms",
        "x", "y", "heading",
        "v_cmd", "omega_cmd",
        "rpm_l", "rpm_r",
        "gyro_z",
        "imu_temp_c", "imu_accel_x",
        "vbat", "current_a",
        "duty_l", "duty_r",
        "target_rpm_l", "target_rpm_r",
    ]
    for i in (1, 2, 3):
        cols += [
            f"uwb{i}_peer_id",
            f"uwb{i}_lde_err",
            f"uwb{i}_std_noise",
            f"uwb{i}_fp_ampl1",
            f"uwb{i}_fp_ampl2",
            f"uwb{i}_cir_pwr",
            f"uwb{i}_raw",
            f"uwb{i}_clean",
            f"uwb{i}_rssi",
            f"uwb{i}_fp_power",
            f"uwb{i}_resp_temp",
        ]
    cols += [
        "rxpacc",
        "eskf_var_x", "eskf_var_y",
        "eskf_var_theta", "eskf_var_bias",
        "tdma_frame_id",
        "tdma_slot_index", "tdma_sync_lost",
        "maneuver_id", "robot_state",
    ]
    return cols


# ==============================================================================
# Main decoder
# ==============================================================================
def verify_and_unpack(filename):
    with open(filename, "rb") as f:
        data = f.read()

    if len(data) < 22:
        print(f"ERROR: file too small ({len(data)} bytes)")
        return

    # ---- Header ----
    magic, version, robot_id, maneuver_id, count, rec_size = struct.unpack(
        "<4sHBB I H", data[:14]
    )
    print(f"\nHeader: Magic={magic.decode()}, Version={version}, "
          f"RobotID={robot_id}, ManeuverID={maneuver_id}, "
          f"Records={count}, RecSize={rec_size}")

    if magic != b"ANJM":
        print("ERROR: Invalid Header Magic")
        return

    # ---- Pick schema ----
    if version == 1:
        expected_size = RECORD_V1_SIZE
        record_format = RECORD_V1_FORMAT
        cols          = BASE_V1_COLS
    elif version == 2:
        expected_size = RECORD_V2_SIZE
        record_format = RECORD_V2_FORMAT
        cols          = v2_columns()
    else:
        print(f"ERROR: Unsupported schema version {version}")
        return

    if rec_size != expected_size:
        print(f"ERROR: Record size mismatch: header says {rec_size}, "
              f"decoder expects {expected_size}")
        return

    # ---- Payload + footer ----
    payload     = data[14:-8]
    footer_data = data[-8:]

    if len(payload) < count * rec_size:
        print(f"ERROR: payload size ({len(payload)}) < "
              f"count × rec_size ({count * rec_size})")
        return

    expected_crc, end_magic = struct.unpack("<I4s", footer_data)

    if end_magic != b"MJNA":
        print("ERROR: Invalid Footer Magic")
        return

    computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
    ok = (expected_crc == computed_crc)
    print(f"CRC Check: Expected=0x{expected_crc:08X}, "
          f"Computed=0x{computed_crc:08X} -> "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- Unpack ----
    rows = []
    for i in range(count):
        chunk = payload[i * rec_size : (i + 1) * rec_size]
        if len(chunk) != rec_size:
            print(f"WARNING: short chunk at index {i}, stopping")
            break
        rows.append(struct.unpack(record_format, chunk))

    if not rows:
        print("ERROR: no records decoded")
        return

    # ---- Sanity check ----
    n_fields = len(rows[0])
    if n_fields != len(cols):
        print(f"WARNING: field count mismatch: "
              f"record has {n_fields} fields, "
              f"columns defined {len(cols)}")
        n = min(n_fields, len(cols))
        cols = cols[:n]
        rows = [r[:n] for r in rows]

    df = pd.DataFrame(rows, columns=cols)

    csv_name = filename.replace(".bin", ".csv")
    df.to_csv(csv_name, index=False)
    print(f"\nSUCCESS: Saved {len(df)} verified records to {csv_name}")

    print(f"\nDuration: "
          f"{(df['t_ms'].iloc[-1] - df['t_ms'].iloc[0]) / 1000.0:.2f} s")
    print(f"Sample rate: "
          f"{len(df) / ((df['t_ms'].iloc[-1] - df['t_ms'].iloc[0]) / 1000.0):.2f} Hz")

    print(f"\nFirst 2 rows:")
    print(df.head(2).to_string(index=False))
    print(f"\nLast 2 rows:")
    print(df.tail(2).to_string(index=False))


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_log.py <filename.bin>")
        sys.exit(1)
    verify_and_unpack(sys.argv[1])
