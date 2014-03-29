#include <Windows.h>
#include <tchar.h>
#include <algorithm>
#include "NVEncParam.h"
#include "nv_util.h"

int _tmain(int argc, TCHAR **argv) {
	NVEncParam nvParam;

	TCHAR buf[1024];
	getEnviromentInfo(buf, _countof(buf));

	auto nvEncCaps = nvParam.GetNVEncCapability(0);

	size_t max_length = 0;
	std::for_each(nvEncCaps.begin(), nvEncCaps.end(), [&max_length](const NVEncCap& x) { max_length = (std::max)(max_length, _tcslen(x.name)); });
	
	_ftprintf(stderr, "%d - %s\n", argc, argv[0]);

	_ftprintf(stderr, "%s\n", buf);

	for (auto cap : nvEncCaps) {
		_ftprintf(stderr, _T("%s"), cap.name);
		for (size_t i = _tcslen(cap.name); i <= max_length; i++)
			_ftprintf(stderr, _T(" "));
		_ftprintf(stderr, _T("%d\n"), cap.value);
	}

	return 0;
}

