#pragma once

#include "rgy_version.h"

#if ENABLE_PROBE_BUILD
#include <cuda.h>
#include "rgy_probe_cli.h"

int nvenc_probe_inventory();
int nvenc_probe_devices();
rgy_probe::Json nvenc_probe_device_identity(int index, CUdevice device);
rgy_probe::Json nvenc_probe_cuda_failure(const char *reason, const char *api, CUresult error);
#endif
