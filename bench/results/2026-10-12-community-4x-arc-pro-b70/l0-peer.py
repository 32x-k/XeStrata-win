#!/usr/bin/env python3
"""Ask Level Zero whether the Arc Pro B70s on this machine can access each other's VRAM (P2P).

zeDeviceCanAccessPeer is the authoritative answer: it reports whether a device can address a
peer's memory directly.  Read-only queries, no compute, safe to run while the engine serves.

Usage: l0-peer.py
"""
import ctypes

ze = ctypes.CDLL("libze_loader.so.1")
U32, U64 = ctypes.c_uint32, ctypes.c_uint64
PV = ctypes.c_void_p


def check(rc, what):
    if rc != 0:
        print(f"  {what}: Level Zero error {rc}")
        return False
    return True


ze.zeInit.argtypes = [U32]
ze.zeDriverGet.argtypes = [ctypes.POINTER(U32), ctypes.POINTER(PV)]
ze.zeDeviceGet.argtypes = [PV, ctypes.POINTER(U32), ctypes.POINTER(PV)]
ze.zeDeviceCanAccessPeer.argtypes = [PV, PV, ctypes.POINTER(ctypes.c_uint8)]
ze.zeDeviceGetProperties.argtypes = [PV, PV]
ze.zeDeviceGetP2PProperties.argtypes = [PV, PV, PV]

if not check(ze.zeInit(0), "zeInit"):
    raise SystemExit(1)

ndrv = U32(0)
ze.zeDriverGet(ctypes.byref(ndrv), None)
drivers = (PV * ndrv.value)()
ze.zeDriverGet(ctypes.byref(ndrv), drivers)

ndev = U32(0)
ze.zeDeviceGet(drivers[0], ctypes.byref(ndev), None)
devs = (PV * ndev.value)()
ze.zeDeviceGet(drivers[0], ctypes.byref(ndev), devs)
print(f"  devices visible to Level Zero: {ndev.value}")

# a small struct for the fields we want out of ze_device_properties_t
class Props(ctypes.Structure):
    _fields_ = [("stype", U32), ("pNext", PV), ("type", U32), ("vendorId", U32),
                ("deviceId", U32), ("flags", U32), ("coreClockRate", U32),
                ("maxMemAllocSize", U64), ("maxHardwareContexts", U32),
                ("maxCommandQueuePriority", U32), ("numThreadsPerEU", U32),
                ("physicalEUSimdWidth", U32), ("numEUsPerSubslice", U32),
                ("numSubslicesPerSlice", U32), ("numSlices", U32),
                ("timerResolution", U64), ("timestampValidBits", U32),
                ("kernelTimestampValidBits", U32), ("uuid", ctypes.c_uint8 * 16),
                ("name", ctypes.c_char * 256)]

names = []
for d in devs:
    p = Props()
    ze.zeDeviceGetProperties(d, ctypes.byref(p))
    names.append(p.name.decode(errors="replace"))

class P2P(ctypes.Structure):
    _fields_ = [("stype", U32), ("pNext", PV), ("flags", U32),
                ("maxBandwidth", U32), ("latency", U32)]

print("  pairwise zeDeviceCanAccessPeer (1 = the first device may read the second's memory):")
pairs = [(i, j) for i in range(ndev.value) for j in range(ndev.value) if i != j]
reachable = 0
for i, j in pairs:
    v = ctypes.c_uint8(0)
    rc = ze.zeDeviceCanAccessPeer(devs[i], devs[j], ctypes.byref(v))
    ok = (rc == 0 and v.value == 1)
    reachable += ok
    band, lat = "", ""
    if ok:
        p2p = P2P()
        if ze.zeDeviceGetP2PProperties(devs[i], devs[j], ctypes.byref(p2p)) == 0:
            band = f"  maxBandwidth {p2p.maxBandwidth} MB/s  latency {p2p.latency} ns"
    print(f"    dev{i} -> dev{j}: {'YES' if ok else 'no '} (ze rc={rc}){band}")
print(f"  reachable ordered pairs: {reachable} of {len(pairs)}")
