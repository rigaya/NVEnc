#ifndef AFS_CLIENT_H
#define AFS_CLIENT_H

#define AFS_FLAG_SHIFT0      0x01
#define AFS_FLAG_SHIFT1      0x02
#define AFS_FLAG_SHIFT2      0x04
#define AFS_FLAG_SHIFT3      0x08
#define AFS_FLAG_FRAME_DROP  0x10
#define AFS_FLAG_SMOOTHING   0x20
#define AFS_FLAG_FORCE24     0x40
#define AFS_FLAG_ERROR       0x80
#define AFS_MASK_SHIFT0      0xfe
#define AFS_MASK_SHIFT1      0xfd
#define AFS_MASK_SHIFT2      0xfb
#define AFS_MASK_SHIFT3      0xf7
#define AFS_MASK_FRAME_DROP  0xef
#define AFS_MASK_SMOOTHING   0xdf
#define AFS_MASK_FORCE24     0xbf
#define AFS_MASK_ERROR       0x7f

#define AFS_STATUS_DEFAULT   0

#ifndef AFS_CLIENT_NO_SHARE

#define AFS_SHARE_SIZE       0x0018
#define AFS_OFFSET_SHARE_N   0x0000
#define AFS_OFFSET_SHARE_ERR 0x0004
#define AFS_OFFSET_FRAME_N   0x0008
#define AFS_OFFSET_STARTFRM  0x000C
#define AFS_OFFSET_STATUSPTR 0x0010

#define afs_header(x) (*(int*)(MemView+(x)))
#define afs_headerp(x) (*(BYTE**)(MemView+(x)))

static BYTE afs_read(int frame)
{
  static char MemFile[256] = "afs7_"; /* major version in the memory name */
  HANDLE hMemMap;
  BYTE *MemView, *statusp;
  DWORD pid, temp;
  int offset, frame_n, i;
  
  if(MemFile[5] == 0){
    pid = GetCurrentProcessId();
    for(i = 5; pid >= 10;){
      for(temp = 10; pid / 10 >= temp;) temp *= 10;
      MemFile[i++] = '0' + (char)(pid / temp);
      pid %= temp;
    }
    MemFile[i++] = '0' + (char)pid;
    MemFile[i] = 0;
  }
  
  if(frame < 0) return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  
  hMemMap = OpenFileMappingA(FILE_MAP_READ, FALSE, MemFile);
  if(hMemMap == NULL) return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  
  MemView = (BYTE*) MapViewOfFile(hMemMap, FILE_MAP_READ, 0, 0, 0);
  if(MemView == NULL){
    CloseHandle(hMemMap);
    return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  }
  
  if(afs_headerp(AFS_OFFSET_SHARE_ERR)){
    UnmapViewOfFile(MemView);
    CloseHandle(hMemMap);
    return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  }
  statusp = afs_headerp(AFS_OFFSET_STATUSPTR);
  offset  = afs_header(AFS_OFFSET_STARTFRM);
  frame_n = afs_header(AFS_OFFSET_FRAME_N);
  UnmapViewOfFile(MemView);
  CloseHandle(hMemMap);
  
  if(frame >= frame_n) return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  if(statusp == NULL)  return AFS_STATUS_DEFAULT | AFS_FLAG_ERROR;
  return statusp[offset+frame];
}
#endif /* !AFS_CLIENT_NO_SHARE */

#ifndef AFS_CLIENT_NO_TCALC
typedef struct {
  int quarter_jitter;
  int additional_jitter;
  int phase24;
  int position24;
  int prev_jitter;
  BYTE prev_status;
} AFS_CLIENT_STATUS;

static AFS_CLIENT_STATUS afs_client_status;

static int afs_init(BYTE status, int drop24)
{
  afs_client_status.prev_status = status;
  afs_client_status.prev_jitter = 0;
  afs_client_status.additional_jitter = 0;
  afs_client_status.phase24 = 4;
  afs_client_status.position24 = 0;
  if(drop24 ||
     (!(status & AFS_FLAG_SHIFT0) &&
       (status & AFS_FLAG_SHIFT1) &&
       (status & AFS_FLAG_SHIFT2)))
    afs_client_status.phase24 = 0;
  if(status & AFS_FLAG_FORCE24){
    afs_client_status.position24++;
  }else{
    afs_client_status.phase24 -= afs_client_status.position24 + 1;
    afs_client_status.position24 = 0;
  }
  return 0;
}

static int afs_set_status(BYTE status, int drop24)
{
  AFS_CLIENT_STATUS* afs = &afs_client_status;
  int drop, pull_drop, quarter_jitter;
  
  if(status & AFS_FLAG_SHIFT0)
    quarter_jitter = -2;
  else if(afs->prev_status & AFS_FLAG_SHIFT0)
    quarter_jitter = (status & AFS_FLAG_SMOOTHING) ? -1 : -2;
  else
    quarter_jitter = 0;
  quarter_jitter += ((status & AFS_FLAG_SMOOTHING) || afs->additional_jitter != -1) ? afs->additional_jitter : -2;
  
  pull_drop = (status & AFS_FLAG_FRAME_DROP)
              && !((afs->prev_status|status) & AFS_FLAG_SHIFT0)
              && (status & AFS_FLAG_SHIFT1);
  afs->additional_jitter = pull_drop ? -1 : 0;
  
  drop24 = drop24 ||
           (!(status & AFS_FLAG_SHIFT0) &&
             (status & AFS_FLAG_SHIFT1) &&
             (status & AFS_FLAG_SHIFT2));
  if(drop24) afs->phase24 = (afs->position24 + 100) % 5;
  drop24 = 0;
  if(afs->position24 >= afs->phase24 &&
     ((afs->position24 + 100) % 5 == afs->phase24 ||
      (afs->position24 +  99) % 5 == afs->phase24)){
    afs->position24 -= 5;
    drop24 = 1;
  }
  
  if(status & AFS_FLAG_FORCE24){
    pull_drop = drop24;
    quarter_jitter = afs->position24++;
  }else{
    afs->phase24 -= afs->position24 + 1;
    afs->position24 = 0;
  }
  drop = (quarter_jitter - afs->prev_jitter < ((status & AFS_FLAG_FRAME_DROP) ? 0 : -3));
  
  afs->quarter_jitter = quarter_jitter;
  afs->prev_status = status;
  
  return drop || pull_drop;
}

static void afs_drop(void)
{
  afs_client_status.prev_jitter -= 4;
  
  return;
}

static int afs_get_jitter(void)
{
  return afs_client_status.prev_jitter = afs_client_status.quarter_jitter;
}

#ifndef AFS_CLIENT_NO_VBUF
#define AFS_VBUF_N_MAX  16
#define AFS_VBUF_N_MASK (AFS_VBUF_N_MAX - 1)
static struct {
  int frame[AFS_VBUF_N_MAX];
  int mode;
  int size;
  DWORD format;
  void* buf[AFS_VBUF_N_MAX];
  int drop;
} afs_vbuf;

static HANDLE hEventOn;
static HANDLE hEventOff;
static HANDLE hThread;
static DWORD  dwThreadId;
static int    prefetch_frame;
static int    prefetch_stop;

static unsigned __stdcall thread_func(OUTPUT_INFO *oip)
{
  void* pbuf;
  int i;

  while(1){
    WaitForSingleObject(hEventOn, INFINITE);
    if(prefetch_frame < 0) break;
    for(i = prefetch_frame + 2; i < prefetch_frame + AFS_VBUF_N_MAX && i < oip->n && !prefetch_stop; i++){
      if(!VPHELP_get_flag(i)) break;
      if(afs_vbuf.frame[i & AFS_VBUF_N_MASK] == i) continue;
      pbuf = oip->func_get_video_ex(i, afs_vbuf.format);
      memcpy(afs_vbuf.buf[i & AFS_VBUF_N_MASK], pbuf, afs_vbuf.size);
      afs_vbuf.frame[i & AFS_VBUF_N_MASK] = i;
    }
    SetEvent(hEventOff);
  }
  _endthreadex(0);
  return 0;
}

static int afs_vbuf_setup(OUTPUT_INFO *oip, int mode, int size, DWORD format)
{
  unsigned char status;
  void* data;
  int i;
  
  hEventOn  = CreateEvent(NULL, FALSE, FALSE, NULL);
  hEventOff = CreateEvent(NULL, FALSE, FALSE, NULL);
  prefetch_frame = -1;
  hThread   = (HANDLE)_beginthreadex(NULL, 0, (unsigned int (__stdcall*)(void*))thread_func, (LPVOID)oip, 0, (unsigned int*)&dwThreadId);
  
  afs_vbuf.frame[0] = 0;
  for(i=1; i<AFS_VBUF_N_MAX; i++) afs_vbuf.frame[i] = -1;
  afs_vbuf.mode = mode;
  afs_vbuf.size = size;
  afs_vbuf.format = format;
  for(i=0; i<AFS_VBUF_N_MAX; i++) if((afs_vbuf.buf[i] = malloc(size)) == NULL) return 0;
  afs_vbuf.drop = 0;
  
  data = oip->func_get_video_ex(0, format);
  memcpy(afs_vbuf.buf[0], data, size);
  
  if(mode){
    if((status = afs_read(0)) & AFS_FLAG_ERROR) return 0;
    afs_init(status, 0);
  }
  
  return 1;
}

static void* afs_get_video(OUTPUT_INFO *oip, int frame, int* drop, int* next_jitter)
{
  void* data;
  int next_drop, quarter_jitter;
  unsigned char status;
  
  next_drop = 0;
  quarter_jitter = 0;
  
  if(afs_vbuf.frame[frame & AFS_VBUF_N_MASK] != frame){
    data = oip->func_get_video_ex(frame, afs_vbuf.format);
    memcpy(afs_vbuf.buf[frame & AFS_VBUF_N_MASK], data, afs_vbuf.size);
    afs_vbuf.frame[frame & AFS_VBUF_N_MASK] = frame;
  }
  
  if(frame + 1 < oip->n){
    if(afs_vbuf.frame[(frame + 1) & AFS_VBUF_N_MASK] != frame + 1){
      data = oip->func_get_video_ex(frame + 1, afs_vbuf.format);
      memcpy(afs_vbuf.buf[(frame + 1) & AFS_VBUF_N_MASK], data, afs_vbuf.size);
      afs_vbuf.frame[(frame + 1) & AFS_VBUF_N_MASK] = frame + 1;
    }
    
    if(afs_vbuf.mode){
      status = afs_read(frame + 1);
      if(status & AFS_FLAG_ERROR) return NULL;
      next_drop = afs_set_status(status, 0);
      if(next_drop){
        afs_drop();
        quarter_jitter = 0;
      }else
        quarter_jitter = afs_get_jitter();
    }
  }
  
  *drop = afs_vbuf.drop;
  *next_jitter = quarter_jitter;
  afs_vbuf.drop = next_drop;
  
  return afs_vbuf.buf[frame & AFS_VBUF_N_MASK];
}

static void afs_vbuf_release()
{
  int i;
  
  prefetch_frame = -1;
  SetEvent(hEventOn);
  for(i=0; i<AFS_VBUF_N_MAX; i++) if(afs_vbuf.buf[i] != NULL) free(afs_vbuf.buf[i]);
  WaitForSingleObject(hThread, INFINITE);
  CloseHandle(hThread);
  CloseHandle(hEventOff);
  CloseHandle(hEventOn);
}
#endif /* !AFS_CLIENT_NO_VBUF */

#endif /* !AFS_CLIENT_NO_TCALC */

#endif /* AFS_CLIENT_H */
