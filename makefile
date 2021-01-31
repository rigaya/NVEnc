include config.mak

vpath %.cpp $(SRCDIR)

OBJS    = $(SRCS:%.cpp=%.cpp.o)
OBJCUS  = $(SRCCUS:%.cu=%.o)
OBJPYWS = $(PYWS:%.pyw=%.o)
OBJBINS = $(BINS:%.bin=%.o)
OBJBINHS = $(BINHS:%.h=%.o)

all: $(PROGRAM)

$(PROGRAM): .depend $(OBJCUS) $(OBJS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS)
	$(LD) $(OBJS) $(OBJCUS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS) $(LDFLAGS) -o $(PROGRAM)

%.cpp.o: %.cpp .depend
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.o: %.cu .depend
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

%.o: %.pyw
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.o: %.bin
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@

%.o: %.h
	objcopy -I binary -O elf64-x86-64 -B i386 $< $@
	
.depend: config.mak
	@rm -f .depend
	@echo 'generate .depend...'
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -g0 -MT $(SRC:$(SRCDIR)/%.cpp=%.cpp.o) -MM >> .depend;)
	@$(foreach SRC, $(SRCCUS:%=$(SRCDIR)/%), $(NVCC) $(SRC) $(NVCCFLAGS) -MT $(SRC:$(SRCDIR)/%.cu=%.o) -MM >> .depend;)
	
ifneq ($(wildcard .depend),)
include .depend
endif

clean:
	rm -f $(OBJS) $(OBJCUS) $(OBJBINS) $(OBJBINHS) $(OBJPYWS) $(PROGRAM) .depend

distclean: clean
	rm -f config.mak NVEncCore/rgy_config.h

install: all
	install -d $(PREFIX)/bin
	install -m 755 $(PROGRAM) $(PREFIX)/bin

uninstall:
	rm -f $(PREFIX)/bin/$(PROGRAM)

config.mak:
	./configure
