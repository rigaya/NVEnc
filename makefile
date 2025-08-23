include config.mak

vpath %.cpp $(SRCDIR)

OBJS    = $(SRCS:%.cpp=%.cpp.o)
OBJCUS  = $(SRCCUS:%.cu=%.o)
OBJPYWS = $(PYWS:%.pyw=%.o)
OBJBINS = $(BINS:%.bin=%.o)
OBJBINHS = $(BINHS:%.h=%.o)
DEPS = $(SRCS:.cpp=.cpp.d)
DEPCUS = $(SRCCUS:.cu=.d)

all: $(PROGRAM)

$(PROGRAM): $(DEPS) $(DEPCUS) $(OBJCUS) $(OBJS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS)
	$(LD) $(OBJS) $(OBJCUS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS) $(LDFLAGS) -o $(PROGRAM)

%_sse2.cpp.o: %_sse2.cpp
	$(CXX) -c $(CXXFLAGS) -msse2 -o $@ $<

%_ssse3.cpp.o: %_ssse3.cpp
	$(CXX) -c $(CXXFLAGS) -mssse3 -o $@ $<

%_sse41.cpp.o: %_sse41.cpp
	$(CXX) -c $(CXXFLAGS) -msse4.1 -o $@ $<

%_avx.cpp.o: %_avx.cpp
	$(CXX) -c $(CXXFLAGS) -mavx -mpopcnt -o $@ $<

%_avx2.cpp.o: %_avx2.cpp
	$(CXX) -c $(CXXFLAGS) -mavx2 -mfma -mpopcnt -mbmi -mbmi2 -o $@ $<

%_avx512bw.cpp.o: %_avx512bw.cpp
	$(CXX) -c $(CXXFLAGS) -mavx512f -mavx512bw -mpopcnt -mbmi -mbmi2 -o $@ $<

%.cpp.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.cpp.d: %.cpp
	@$(CXX) ./$< $(CXXFLAGS) -g0 -MT $(basename $<).cpp.o -MM > $@

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

%.d: %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

%.d: %.cu
	@$(NVCC) ./$< $(NVCCFLAGS) -MT $(basename $<).o -MM > $@

%.o: %.pyw
	objcopy -I binary -O $(OBJCOPY_ARCH_ELF) -B $(OBJCOPY_ARCH_BIN) $< $@

%.o: %.bin
	objcopy -I binary -O $(OBJCOPY_ARCH_ELF) -B $(OBJCOPY_ARCH_BIN) $< $@

%.o: %.h
	objcopy -I binary -O $(OBJCOPY_ARCH_ELF) -B $(OBJCOPY_ARCH_BIN) $< $@
	
-include $(DEPS)
-include $(DEPCUS)

clean:
	rm -f $(DEPS) $(DEPCUS) $(OBJS) $(OBJCUS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS) $(PROGRAM)

distclean: clean
	rm -f config.mak NVEncCore/rgy_config.h

install: all
	install -d $(PREFIX)/bin
	install -m 755 $(PROGRAM) $(PREFIX)/bin

uninstall:
	rm -f $(PREFIX)/bin/$(PROGRAM)

config.mak:
	./configure
