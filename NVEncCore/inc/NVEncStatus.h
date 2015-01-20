//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <Windows.h>
#include <stdio.h>
#include <stdint.h>
#include <tchar.h>
#include <string>
#include <process.h>
#pragma comment(lib, "winmm.lib")
#include "nvEncodeAPI.h"
#include "cpu_info.h"


#ifndef MIN3
#define MIN3(a,b,c) (min((a), min((b), (c))))
#endif
#ifndef MAX3
#define MAX3(a,b,c) (max((a), max((b), (c))))
#endif

typedef struct EncodeStatusData {
	uint64_t outFileSize;      //出力ファイルサイズ
	uint32_t tmStart;          //エンコード開始時刻
	uint32_t tmLastUpdate;     //最終更新時刻
	uint32_t frameTotal;       //入力予定の全フレーム数
	uint32_t frameOut;         //出力したフレーム数
	uint32_t frameOutIDR;      //出力したIDRフレーム
	uint32_t frameOutI;        //出力したIフレーム
	uint32_t frameOutP;        //出力したPフレーム
	uint32_t frameOutB;        //出力したBフレーム
	uint64_t frameOutISize;    //出力したIフレームのサイズ
	uint64_t frameOutPSize;    //出力したPフレームのサイズ
	uint64_t frameOutBSize;    //出力したBフレームのサイズ
	uint32_t frameOutIQPSum;   //出力したIフレームの平均QP
	uint32_t frameOutPQPSum;   //出力したPフレームの平均QP
	uint32_t frameOutBQPSum;   //出力したBフレームの平均QP
	uint32_t frameIn;          //エンコーダに入力したフレーム数 (drop含まず)
	uint32_t frameDrop;        //ドロップしたフレーム数
	double encodeFps;          //エンコード速度
	double bitrateKbps;        //ビットレート
} EncodeStatusData;

class EncodeStatus {
public:
	EncodeStatus() {
		ZeroMemory(&m_sData, sizeof(m_sData));

		m_sData.tmLastUpdate = timeGetTime();

		DWORD mode = 0;
		m_bStdErrWriteToConsole = 0 != GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode); //stderrの出力先がコンソールかどうか
	}
	~EncodeStatus() { };

	virtual void SetStart() {
		m_sData.tmStart = timeGetTime();
		GetProcessTime(GetCurrentProcess(), &m_sStartTime);
	}
	virtual void AddOutputInfo(const NV_ENC_LOCK_BITSTREAM *bitstream) {
		const NV_ENC_PIC_TYPE picType = bitstream->pictureType;
		const uint32_t outputBytes = bitstream->bitstreamSizeInBytes;
		const uint32_t frameAvgQP = bitstream->frameAvgQP;
		m_sData.outFileSize    += outputBytes;
		m_sData.frameOut       += 1;
		m_sData.frameOutIDR    += (NV_ENC_PIC_TYPE_IDR == picType);
		m_sData.frameOutI      += (NV_ENC_PIC_TYPE_IDR == picType);
		m_sData.frameOutI      += (NV_ENC_PIC_TYPE_I   == picType);
		m_sData.frameOutP      += (NV_ENC_PIC_TYPE_P   == picType);
		m_sData.frameOutB      += (NV_ENC_PIC_TYPE_B   == picType);
		m_sData.frameOutISize  += (0-(NV_ENC_PIC_TYPE_IDR == picType)) & outputBytes;
		m_sData.frameOutISize  += (0-(NV_ENC_PIC_TYPE_I   == picType)) & outputBytes;
		m_sData.frameOutPSize  += (0-(NV_ENC_PIC_TYPE_P   == picType)) & outputBytes;
		m_sData.frameOutBSize  += (0-(NV_ENC_PIC_TYPE_B   == picType)) & outputBytes;
		m_sData.frameOutIQPSum += (0-(NV_ENC_PIC_TYPE_IDR == picType)) & frameAvgQP;
		m_sData.frameOutIQPSum += (0-(NV_ENC_PIC_TYPE_I   == picType)) & frameAvgQP;
		m_sData.frameOutPQPSum += (0-(NV_ENC_PIC_TYPE_P   == picType)) & frameAvgQP;
		m_sData.frameOutBQPSum += (0-(NV_ENC_PIC_TYPE_B   == picType)) & frameAvgQP;
	}
	virtual void UpdateDisplay(const TCHAR *mes) {
#if UNICODE
		char *mes_char = NULL;
		if (!m_bStdErrWriteToConsole) {
			//コンソールへの出力でなければ、ANSIに変換する
			const int buf_length = (int)(wcslen(mes) + 1) * 2;
			if (NULL != (mes_char = (char *)calloc(buf_length, 1))) {
				WideCharToMultiByte(CP_THREAD_ACP, 0, mes, -1, mes_char, buf_length, NULL, NULL);
				fprintf(stderr, "%s\r", mes_char);
				free(mes_char);
			}
		} else
#endif
			_ftprintf(stderr, _T("%s\r"), mes);

		fflush(stderr); //リダイレクトした場合でもすぐ読み取れるようflush
	}
	virtual void WriteFrameTypeResult(const TCHAR *header, uint32_t count, uint32_t maxCount, uint64_t frameSize, uint64_t maxFrameSize, double avgQP) {
		if (count) {
			TCHAR mes[512] = { 0 };
			int mes_len = 0;
			const int header_len = (int)_tcslen(header);
			memcpy(mes, header, header_len * sizeof(mes[0]));
			mes_len += header_len;

			for (int i = max(0, (int)log10((double)count)); i < (int)log10((double)maxCount) && mes_len < _countof(mes); i++, mes_len++)
				mes[mes_len] = _T(' ');
			mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%u"), count);

			if (avgQP >= 0.0) {
				mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T(",  avgQP  %4.2f"), avgQP);
			}
			
			if (frameSize > 0) {
				const TCHAR *TOTAL_SIZE = _T(",  total size  ");
				memcpy(mes + mes_len, TOTAL_SIZE, _tcslen(TOTAL_SIZE) * sizeof(mes[0]));
				mes_len += (int)_tcslen(TOTAL_SIZE);

				for (int i = max(0, (int)log10((double)frameSize / (double)(1024 * 1024))); i < (int)log10((double)maxFrameSize / (double)(1024 * 1024)) && mes_len < _countof(mes); i++, mes_len++)
					mes[mes_len] = _T(' ');

				mes_len += _stprintf_s(mes + mes_len, _countof(mes) - mes_len, _T("%.2f MB"), (double)frameSize / (double)(1024 * 1024));
			}

			WriteLine(mes);
		}
	}
	virtual void WriteLine(const TCHAR *mes) {
		fprintf(stderr, "%s\n", mes);
	}
	virtual void UpdateDisplay() {
		uint32_t tm = timeGetTime();
		if (tm - m_sData.tmLastUpdate < 800)
			return;
		if (m_sData.frameOut + m_sData.frameDrop) {
			TCHAR mes[256] = { 0 };
			m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / (double)(tm - m_sData.tmStart);
			m_sData.bitrateKbps = (double)m_sData.outFileSize * (m_nOutputFPSRate / (double)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
			if (0 < m_sData.frameTotal) {
				uint32_t remaining_time = (uint32_t)((m_sData.frameTotal - (m_sData.frameOut + m_sData.frameDrop)) * 1000.0 / ((m_sData.frameOut + m_sData.frameDrop) * 1000.0 / (double)(tm - m_sData.tmStart)));
				int hh = remaining_time / (60*60*1000);
				remaining_time -= hh * (60*60*1000);
				int mm = remaining_time / (60*1000);
				remaining_time -= mm * (60*1000);
				int ss = (remaining_time + 500) / 1000;

				int len = _stprintf_s(mes, _countof(mes), _T("[%.1lf%%] %d frames: %.2lf fps, %0.2lf kb/s, remain %d:%02d:%02d  "),
					(m_sData.frameOut + m_sData.frameDrop) * 100 / (double)m_sData.frameTotal,
					(m_sData.frameOut + m_sData.frameDrop),
					m_sData.encodeFps,
					m_sData.bitrateKbps,
					hh, mm, ss );
				if (m_sData.frameDrop)
					_stprintf_s(mes + len - 2, _countof(mes) - len + 2, _T(", afs drop %d/%d  "), m_sData.frameDrop, (m_sData.frameOut + m_sData.frameDrop));
			} else {
				_stprintf_s(mes, _countof(mes), _T("%d frames: %0.2lf fps, %0.2lf kbps  "), 
					(m_sData.frameOut + m_sData.frameDrop),
					m_sData.encodeFps,
					m_sData.bitrateKbps
					);
			}
			UpdateDisplay(mes);
			m_sData.tmLastUpdate = tm;
		}
	}
	virtual void writeResult() {
		uint32_t time_elapsed = timeGetTime() - m_sData.tmStart;

		TCHAR mes[512] = { 0 };
		for (int i = 0; i < 79; i++)
			mes[i] = ' ';
		WriteLine(mes);

		uint32_t tm = timeGetTime();
		m_sData.encodeFps = (m_sData.frameOut + m_sData.frameDrop) * 1000.0 / (double)(tm - m_sData.tmStart);
		m_sData.bitrateKbps = (double)m_sData.outFileSize * (m_nOutputFPSRate / (double)m_nOutputFPSScale) / ((1000 / 8) * (m_sData.frameOut + m_sData.frameDrop));
		m_sData.tmLastUpdate = tm;

		_stprintf_s(mes, _countof(mes), _T("encoded %d frames, %.2f fps, %.2f kbps, %.2f MB"),
			m_sData.frameOut,
			m_sData.encodeFps,
			m_sData.bitrateKbps,
			(double)m_sData.outFileSize / (double)(1024 * 1024)
			);
		WriteLine(mes);

		int hh = time_elapsed / (60*60*1000);
		time_elapsed -= hh * (60*60*1000);
		int mm = time_elapsed / (60*1000);
		time_elapsed -= mm * (60*1000);
		int ss = (time_elapsed + 500) / 1000;
		_stprintf_s(mes, _countof(mes), _T("encode time %d:%02d:%02d / CPU Usage: %.2f%%\n"), hh, mm, ss, GetProcessAvgCPUUsage(GetCurrentProcess(), &m_sStartTime));
		WriteLine(mes);
		
		uint32_t maxCount = MAX3(m_sData.frameOutI, m_sData.frameOutP, m_sData.frameOutB);
		uint64_t maxFrameSize = MAX3(m_sData.frameOutISize, m_sData.frameOutPSize, m_sData.frameOutBSize);

		WriteFrameTypeResult(_T("frame type IDR "), m_sData.frameOutIDR, maxCount,                     0, maxFrameSize, -1.0);
		WriteFrameTypeResult(_T("frame type I   "), m_sData.frameOutI,   maxCount, m_sData.frameOutISize, maxFrameSize, (m_sData.frameOutI) ? m_sData.frameOutIQPSum / (double)m_sData.frameOutI : -1);
		WriteFrameTypeResult(_T("frame type P   "), m_sData.frameOutP,   maxCount, m_sData.frameOutPSize, maxFrameSize, (m_sData.frameOutP) ? m_sData.frameOutPQPSum / (double)m_sData.frameOutP : -1);
		WriteFrameTypeResult(_T("frame type B   "), m_sData.frameOutB,   maxCount, m_sData.frameOutBSize, maxFrameSize, (m_sData.frameOutB) ? m_sData.frameOutBQPSum / (double)m_sData.frameOutB : -1);
	}
public:
	PROCESS_TIME m_sStartTime;
	EncodeStatusData m_sData;
	uint32_t m_nOutputFPSRate = 0;
	uint32_t m_nOutputFPSScale = 0;
	bool m_bStdErrWriteToConsole;
};
