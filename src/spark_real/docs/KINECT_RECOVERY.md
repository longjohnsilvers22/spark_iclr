# Kinect Recovery

How to bring an Azure Kinect back up when `/api/status` shows
`kinect_connected: false` or the SDK fails to open it.

## TL;DR for future agents

**Software recovery works in MOST cases.** Don't tell the user "you
need to unplug the Kinect" until you have:

1. Run `sudo ~/spark/scripts/kinect_authorize_reset.sh`
   (it does two phases of recovery and reports per-Kinect status)
2. Run the pyk4a probe to confirm the actual SDK state
3. Confirmed via dmesg that the specific failure is `error -71` on
   USB address-set, which means the depth MCU's USB processor itself
   is hung - that's the ONLY case software can't fix

Even when one Kinect genuinely needs a 12V cycle, the OTHER Kinect is
almost always recoverable. **The SPARK pipeline runs on one Kinect.**

## Symptoms

- `k4a_device_open` fails with `LIBUSB_ERROR_IO` on `libusb_get_bos_descriptor`
- `k4a_device_open` fails with `find_libusb_device() libusb device(s)
  are all unavalable`
- `/api/status` shows one or both of `kinect_connected`,
  `kinect2_connected` as `false`
- `available_cameras` is missing `birdview` or `sideview`
- SAM3 returns 0 detections, or detections with confidence < 0.2
- SAM3 returns 3D positions with `z` far below the table (~-0.17 m)

If perception is dead, **do not chase motion/IK bugs first**. Check
`/api/status` and `/api/detect` output before touching the planner or
the executor.

## Failure signatures (read `lsusb -d 045e:` and `dmesg`)

The recovery path depends on what's enumerated. There are three
distinct signatures:

### Signature A: stale enumeration, all hardware fine

`lsusb -d 045e:` shows everything (2× `097a` SS hub, 2× `097b` HS hub,
2× `097d` 4K cam, 2× `097c` depth cam, 2× `097e` mic) but
`k4a_device_open` still fails with `LIBUSB_ERROR_IO`.

→ The recovery script's **phase 1** (per-hub sysfs `authorized` cycle)
fixes this. Sometimes phase 2 (root-hub re-auth) is also needed.

### Signature B: one Kinect's depth MCU hung (the case that fools people)

`lsusb -d 045e:` shows both `097a` SS hubs, both `097d` 4K cams, but
ONLY ONE `097c` depth camera. The Kinect missing its `097c` has its
depth MCU's USB processor stuck in enumeration:

```
$ dmesg | grep "2-2.2"
usb 2-2.2: Device not responding to setup address.
usb 2-2.2: device not accepting address NN, error -71
usb 2-2-port2: attempt power cycle
usb 2-2-port2: unable to enumerate USB device
```

`error -71` is `EPROTO` - the device is responding to enumeration
probes but failing the USB address-set handshake. This is a state in
the Kinect's depth MCU that USB-level resets cannot clear.

→ **The recovery script reports this case explicitly** and points to
the specific SS hub that needs a 12V cycle. The OTHER Kinect comes
back to working state via phase 1 + phase 2 of the script.

### Signature C: SS hub completely missing

`lsusb -d 045e:` shows fewer than two `097a` SuperSpeed hubs total.
One Kinect's entire SS side is gone (depth + 4K + everything past the
hub). This means the Kinect's SS PHY is hard-locked.

→ Only a 12V cycle on that Kinect recovers it. Software cannot help.
The OTHER Kinect is still recoverable via the script.

## Recovery ladder

### 1. Run the recovery script

```bash
sudo ~/spark/scripts/kinect_authorize_reset.sh
```

This is now the one-stop recovery. It:
- Discovers Kinect hubs dynamically by USB ID (no hardcoded buses)
- Phase 1: per-hub `authorized=0/1` toggle (kernel detach/reattach)
- Phase 2: if any depth cam still missing, `usb2/authorized` root-hub
  re-auth - broader re-enumeration of the entire usb2 bus
- Reports per-Kinect status at the end: which Kinects are OK and
  which one (if any) needs a 12V cycle, with dmesg evidence

Exit codes:
- `0` - all Kinects recovered (both depth cams enumerated)
- `2` - no Kinect hubs enumerated at all, both unplugged or bricked
- `3` - one Kinect recovered, the other needs a 12V cycle (the
  script tells you which SS hub maps to the bad one)

### 2. Probe with pyk4a

```bash
~/miniconda3/envs/spark_conda/bin/python -c "
import pyk4a
print('count:', pyk4a.connected_device_count())
for i in range(pyk4a.connected_device_count()):
    k = pyk4a.PyK4A(device_id=i)
    try: k.open(); print(f'  {i}: OK serial={k.serial}'); k.close()
    except Exception as e: print(f'  {i}: FAIL')
"
```

Expected serials on this host:
- `000000000000` (sideview, master)
- `000000000000` (birdview)

If at least one Kinect opens, the SPARK pipeline can run.

### 3. Physical 12V barrel cycle (only if script exit code is 3)

If the script exit code 3, ONE Kinect's depth MCU is hung at the
hardware level. Unplug that Kinect's 12V barrel for ~5 s, plug it
back in, re-run the script. After the 12V cycle the script's phase 1
will bring it back cleanly.

### 4. Restart the SPARK server

```bash
# SIGTERM (let the SDK call k4a_device_close cleanly)
pkill -TERM -f spark_real.server
# Wait for it to exit, THEN start again
```

**Never `kill -9` the server while Kinects are open.** SIGKILL
bypasses the SDK's cleanup, leaves the depth MCU mid-stream, and
locks the Kinect at firmware level - even software recovery may
not bring it back without a 12V cycle.

## Things that don't work (don't try them)

These were tested and either accomplish nothing or actively make
things worse:

- `usbreset /dev/bus/usb/...` - fails with "No such device" because
  the BOS descriptor read is what failed
- `echo 0 > /sys/bus/usb/devices/usb2/authorized` alone - works as
  phase 2 of the script, but useless on its own without first doing
  per-hub `authorized` toggles
- `echo 0000:00:14.0 > /sys/bus/pci/drivers/xhci_hcd/unbind` then
  `bind` - **DAMAGES the working Kinect's color MCU**. Triggered a
  `usb_cmd_write(80000001) ... colormcu_set_multi_device_mode`
  stuck state on 2026-05-20 that no software recovery could clear.
  Stick to per-hub and root-hub sysfs `authorized` toggles only.
- `echo <hub> > /sys/bus/usb/drivers/usb/unbind` for a specific
  Kinect hub - leaves the hub in a broken state where downstream
  children disappear. The root-hub `usb2/authorized` toggle can
  recover from this, but don't go there in the first place.
- `uhubctl -a cycle` on a single Kinect's HS hub - only cuts USB
  Vbus, doesn't reach the depth MCU which is powered from 12V
- Multiple phase-1 cycles in a row - if the first one didn't fix it,
  more won't either. Escalate to phase 2 (root-hub re-auth) or 12V.

## Preventing the problem (the udev rule)

The host kernel crashes during SPARK sessions were traced to
`uvcvideo` and `libusb` both holding the Kinect 4K color interface.
Prevented by:

```
# /etc/udev/rules.d/99-k4a-unbind-uvcvideo.rules
ACTION=="add", SUBSYSTEM=="usb", DEVTYPE=="usb_interface",
    ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097d",
    DRIVER=="uvcvideo",
    RUN+="/bin/sh -c 'echo %k > /sys/bus/usb/drivers/uvcvideo/unbind'"
```

This unbinds `uvcvideo` from the Kinect color device at plug time so
only the Azure Kinect SDK (libusb) holds it. If host crashes during
SPARK sessions resume, verify this rule is still in place.

## DISPLAY requirement (libdepthengine)

The Kinect depth engine uses OpenGL and needs a display. Always
launch the SPARK server with:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  python -m spark_real.server --robot franka --auto-unlock --port 8888
```

Without these, depth opens fail with error 204 that *looks* like an
MCU lockup but isn't.

## See also

- `scripts/kinect_software_reset.sh` - uhubctl Vbus cycle. Mostly
  superseded by `kinect_authorize_reset.sh`, but kept for the case
  where uhubctl's deeper hub-level Vbus cut helps.
- `scripts/kinect_authorize_reset.sh` - the primary recovery tool
- `/etc/udev/rules.d/99-k4a-unbind-uvcvideo.rules` - host-crash prevention
