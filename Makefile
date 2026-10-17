# ══════════════════════════════════════════════════════════════════════════
# MPM-STD 빌드 — CPU(build/) 와 GPU(GPU=1 → build_gpu/; .cu 쌍둥이 TU 대체)
#
# 규칙 하나: 실행파일은 전부 apps/<이름>/main.cpp 다. 테스트도 앱이다
# (apps/test_*/main.cpp). 그래서 아래가 전부 성립한다:
#
#   make              전체 빌드 (CPU → build/bin/*)
#   make GPU=1        전체 GPU 빌드 (→ build_gpu/bin/*; GPU 노드에서)
#   make <이름>       개별 앱 빌드   (예: make rbc, make test_grid)
#   make test         test_* 앱 전부 실행 (하나라도 실패하면 exit 1)
#   make clean
#
# src/ 는 와일드카드로 전부 수집한다 — 수동 목록이 파일을 조용히 빌드에서
# 빠뜨리는 사고(MPM-STD_C 의 미빌드 TU 6개)를 원천 차단. 외부 솔버 clone 이
# 들어오면(M6) 그 폴더만 여기서 명시적으로 제외를 추가한다.
# ══════════════════════════════════════════════════════════════════════════
# 컴파일러·옵션·외부 라이브러리 경로는 전부 Makefile.inc 에서 (2026-10-04 결정)
include Makefile.inc
INC := -Isrc -Iapps $(EXT_INC)

# 정본 빌드 디렉터리 (2026-10-06 사용자 결정): CPU = build/, GPU = build_gpu/ —
# obj 는 항상 최신 소스, bin 에는 모든 앱 실행파일. src obj 는 앱들이 공유하고
# 앱별로 다른 것은 obj/apps/<앱>/ 과 bin/<앱> 뿐이다. CPU/GPU 는 같은 .o 경로에
# 다른 플래그의 오브젝트를 섞으면 안 되므로 디렉터리를 분리한다 (BUILD= 로 덮어쓰기
# 가능하지만 돌고 있는 잡의 바이너리를 보존할 때 말고는 쓰지 말 것).
ifdef GPU
BUILD ?= build_gpu
else
BUILD ?= build
endif

# CPU 빌드 = 모든 .cpp. GPU 빌드(GPU=1) = "같은 이름의 .cu 쌍둥이가 있는 .cpp 는
# .cu 로 대체"하고 나머지 .cpp 는 그대로 — 같은 헤더 API 를 백엔드별 TU 가
# 구현한다 (수식은 *_stencil.hpp 공유, 분기 스위치는 MPMSTD_GPU 하나 — D4·D5).
ALL_CPP := $(sort $(shell find src -name '*.cpp'))
ALL_CU  := $(sort $(shell find src -name '*.cu'))
ifdef GPU
CU_TWIN_CPP := $(ALL_CU:.cu=.cpp)
LIB_SRC := $(filter-out $(CU_TWIN_CPP),$(ALL_CPP)) $(ALL_CU)
else
LIB_SRC := $(ALL_CPP)
endif
# .cu 는 .cu.o 로 — .cpp 쌍둥이와 같은 .o 경로를 두면 make 가 패턴 규칙을
# 잘못 집어 구식 오브젝트를 재사용한다 (실측 함정, 2026-10-04)
LIB_OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(filter %.cpp,$(LIB_SRC))) \
           $(patsubst %.cu,$(BUILD)/obj/%.cu.o,$(filter %.cu,$(LIB_SRC)))

APP_SRC  := $(sort $(shell find apps -name 'main.cpp'))
APP_NAME := $(patsubst apps/%/main.cpp,%,$(APP_SRC))
APP_BIN  := $(APP_NAME:%=$(BUILD)/bin/%)
# 앱 오브젝트도 build/obj/ 안으로 — 한 방 컴파일+링크는 nvc++ 가 중간 .o 를
# 실행 폴더(리포 루트)에 남긴다 (실측 2026-10-04). 헤더 의존성은 .d 가 담당.
APP_OBJ_ALL := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(sort $(shell find apps -name '*.cpp')))

# 테스트 = 이름이 test_ 로 시작하는 앱
TEST_BIN := $(filter $(BUILD)/bin/test_%,$(APP_BIN))

all: $(APP_BIN)

# 개별 앱 별칭: make rbc == make build/bin/rbc
.PHONY: $(APP_NAME)
$(APP_NAME): %: $(BUILD)/bin/%

$(BUILD)/obj/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INC) $(DEPFLAGS) -c $< -o $@

$(BUILD)/obj/%.cu.o: %.cu
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INC) $(DEPFLAGS) -c $< -o $@

# 앱 폴더의 모든 .cpp (main.cpp + 그 앱 소유의 settings.cpp 등)를 위의 공용
# 오브젝트 규칙으로 build/obj/apps/<앱>/ 에 컴파일한 뒤 링크만 한다.
# (prerequisite 의 '%' 는 이차 확장 전에 스템으로 치환되므로 patsubst 를 못 쓴다)
.SECONDEXPANSION:
$(BUILD)/bin/%: $$(addprefix $(BUILD)/obj/,$$(subst .cpp,.o,$$(sort $$(wildcard apps/$$*/*.cpp)))) $(LIB_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(EXT_LIBS)

# 검증 테스트 앱 15종은 2026-10-04 사용자 지시로 삭제했다 (channel·rbc 계열
# 앱만 유지). 소스 백업: ../MPM-STD_test_apps_backup_20261004.tar.gz — 풀면
# `make test` 가 다시 전부 집는다 (test_* 는 와일드카드 수집; MPI 조합 실행은
# 그때 test_mpi 타깃을 복원할 것). 하네스 apps/framework.hpp 는 남겨 둔다.
# 시간 측정용 쌍둥이 main (apps/<앱>/main_scale.cpp, tools/scale/instrument_main.py 가 main.cpp 에서 생성):
# make <앱>_scale → build/bin/<앱>_scale. main.cpp 대신 main_scale.cpp 를 -DMPMSTD_SCALE_MAIN 으로 컴파일해 링크한다
# (일반 빌드에서 main_scale.cpp 는 빈 TU). 앱마다 명시 규칙을 만든다 (패턴 규칙 연쇄는 make 가 거부).
SCALE_APPS := $(patsubst apps/%/main_scale.cpp,%,$(wildcard apps/*/main_scale.cpp))
define SCALE_RULE
$(BUILD)/obj/apps/$(1)/main_scale.scale.o: apps/$(1)/main_scale.cpp
	@mkdir -p $$(@D)
	$$(CXX) $$(CXXFLAGS) -DMPMSTD_SCALE_MAIN $$(INC) -c $$< -o $$@
$(BUILD)/bin/$(1)_scale: $(BUILD)/obj/apps/$(1)/main_scale.scale.o $(addprefix $(BUILD)/obj/,$(subst .cpp,.o,$(filter-out apps/$(1)/main.cpp apps/$(1)/main_scale.cpp,$(sort $(wildcard apps/$(1)/*.cpp))))) $(LIB_OBJ)
	@mkdir -p $$(@D)
	$$(CXX) $$(CXXFLAGS) $$^ -o $$@ $$(EXT_LIBS)
.PHONY: $(1)_scale
$(1)_scale: $(BUILD)/bin/$(1)_scale
endef
$(foreach a,$(SCALE_APPS),$(eval $(call SCALE_RULE,$(a))))

test: $(TEST_BIN)
	@fail=0; \
	for t in $(TEST_BIN); do ./$$t || fail=1; done; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TEST FAILED"; exit 1; fi

clean:
	rm -rf $(BUILD)

-include $(LIB_OBJ:.o=.d) $(APP_OBJ_ALL:.o=.d)

# 패턴 규칙 산출물(.o)을 make 가 "중간 파일"로 간주해 지우는 것을 막는다
.SECONDARY: $(LIB_OBJ) $(APP_OBJ_ALL)

.PHONY: all test clean
