#ifndef VPHELP_CLIENT_H
#define VPHELP_CLIENT_H

#define VPH_VERSION          0x0100
#define VPH_HEADER_SIZE      0x0010
#define VPH_OFFSET_VERSION   0x0000
#define VPH_OFFSET_N_FRAME   0x0004

static char VPHELP_MemName[256] = "vphelp_";
static HANDLE VPHELP_hMemMap = NULL;
static unsigned char *VPHELP_MemView = NULL;

#define VPHELP_header(x) (*(int*)(VPHELP_MemView+(x)))
#define VPHELP_data(x) (VPHELP_MemView[VPH_HEADER_SIZE+(x)])

BOOL VPHELP_open(void)
{
  DWORD pid, temp;
  int i;
  
  if(VPHELP_MemName[7] == 0){
    pid = GetCurrentProcessId();
    for(i = 7; pid >= 10;){
      for(temp = 10; pid / 10 >= temp;) temp *= 10;
      VPHELP_MemName[i++] = '0' + (char)(pid / temp);
      pid %= temp;
    }
    VPHELP_MemName[i++] = '0' + (char)pid;
    VPHELP_MemName[i] = 0;
  }
  
  VPHELP_hMemMap = OpenFileMapping(FILE_MAP_READ, FALSE, VPHELP_MemName);
  if(VPHELP_hMemMap == NULL) return FALSE;
  VPHELP_MemView = (unsigned char*) MapViewOfFile(VPHELP_hMemMap, FILE_MAP_READ, 0, 0, 0);
  if(VPHELP_MemView == NULL){
    CloseHandle(VPHELP_hMemMap);
    VPHELP_hMemMap = NULL;
    return FALSE;
  }
  
  return TRUE;
}

void VPHELP_close(void)
{
  if(VPHELP_MemView != NULL){
    UnmapViewOfFile(VPHELP_MemView);
    VPHELP_hMemMap = NULL;
  }
  if(VPHELP_hMemMap != NULL){
    CloseHandle(VPHELP_hMemMap);
    VPHELP_MemView = NULL;
  }
}

BOOL VPHELP_get_flag(int frame)
{
  int tmp;
  
  if(VPHELP_MemView == NULL || VPHELP_hMemMap == NULL)
    return FALSE;
  
  if(VPHELP_header(VPH_OFFSET_VERSION) != VPH_VERSION)
    return FALSE;
  
  if(frame < 0 || frame >= VPHELP_header(VPH_OFFSET_N_FRAME))
    return FALSE;
  
  tmp = VPHELP_data(frame >> 3);
  tmp = (tmp >> (frame & 7)) & 1;
  return (tmp == 1);
}

#endif /* VPHELP_CLIENT_H */
