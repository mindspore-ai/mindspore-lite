/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "coder/generator/component/allocator_component.h"
#include <memory>
#include <utility>
#include <algorithm>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/utils/coder_utils.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
void CodeAllocatorIncludes(std::ofstream &ofs, bool isRiscv) {
  ofs << "#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n"
      << "#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n\n"
      << "#include <stddef.h>\n";
  if (!isRiscv) {
    ofs << "#include \"stdatomic.h\"\n";
  }
  ofs << "#include \"stdbool.h\"\n";
}

void CodeMemBlockStruct(std::ofstream &ofs, bool isStaticAlloc) {
  ofs << "  typedef struct MemBlock {\n" << "#ifdef __clang__\n";
  if (isStaticAlloc) {
    ofs << "    bool occupied;\n";
  } else {
    ofs << "    atomic_bool occupied;\n";
  }
  ofs << "#else\n"
      << "    bool occupied;\n"
      << "#endif\n"
      << "    size_t size;\n"
      << "    void *addr;\n"
      << "    struct MemBlock *next;\n"
      << "  } MemBlock;\n\n";
}

void CodeAllocatorFuncDecls(std::ofstream &ofs, bool isStaticAlloc, bool isRiscv) {
  if (!isRiscv) {
    ofs << "  void IncRefCount();\n";
  }
  if (isStaticAlloc) {
    ofs << "  void DecRefCount(void);\n" << "  void *GlobalMemory(void);\n";
  } else {
    ofs << "  void DecRefCount();\n" << "  void *GlobalMemory();\n";
    ofs << "  void *Malloc(size_t size);\n";
  }
  ofs << "  bool LockBuffer(void *block);\n" << "  bool UnLockBuffer(void *block);\n";
}

void CodeAllocatorHeaderContent(std::ofstream &ofs, bool isStaticAlloc, bool isRiscv) {
  CodeAllocatorIncludes(ofs, isRiscv);
  CodeMemBlockStruct(ofs, isStaticAlloc);
  CodeAllocatorFuncDecls(ofs, isStaticAlloc, isRiscv);
}

void CodeAllocatorFileHeader(std::ofstream &ofs) {
  ofs << g_hwLicense;
  Configurator *config_ = Configurator::GetInstance();
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  bool isRiscv = (config_->target() == kRiscV);
  CodeAllocatorHeaderContent(ofs, isStaticAlloc, isRiscv);
  ofs << "\n#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_ALLOCATOR_H_\n";
}

void CodeCalcRefCountNonRiscV(std::ofstream &ofs, bool isStaticAlloc) {
  if (isStaticAlloc) {
    ofs << R"RAW(
  void IncRefCount() { ++kReferenceCount; }

  void DecRefCount(void) {
    --kReferenceCount;
    bool expected = false;
    while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
      expected = false;
    }
    if (kReferenceCount == 0) {
      mem_block = NULL;
    }
    kLock = false;
  }

  )RAW";
  } else {
    ofs << R"RAW(
  void IncRefCount() { ++kReferenceCount; }

  void DecRefCount() {
    --kReferenceCount;
    bool expected = false;
    while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
      expected = false;
    }
    if (kReferenceCount == 0) {
      while (mem_block != NULL) {
        MemBlock *next = mem_block->next;
        free(mem_block);
        mem_block = next;
      }
    }
    kLock = false;
  }

  )RAW";
  }
}

void CodeCalcRefCountRiscV(std::ofstream &ofs, bool isStaticAlloc) {
  if (isStaticAlloc) {
    ofs << R"RAW(
void DecRefCount(void) {
  mem_block = NULL;
}

  )RAW";
  } else {
    ofs << R"RAW(
  void DecRefCount() {
    while (mem_block != NULL) {
      MemBlock *next = mem_block->next;
      free(mem_block);
      mem_block = next;
    }
  }

  )RAW";
  }
}

void CodeCalcRefCountContent(std::ofstream &ofs, bool isStaticAlloc, bool isRiscv) {
  if (!isRiscv) {
    CodeCalcRefCountNonRiscV(ofs, isStaticAlloc);
  } else {
    CodeCalcRefCountRiscV(ofs, isStaticAlloc);
  }
}

void CodeCalcRefCount(std::ofstream &ofs) {
  Configurator *config_ = Configurator::GetInstance();
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  bool isRiscv = (config_->target() == kRiscV);
  CodeCalcRefCountContent(ofs, isStaticAlloc, isRiscv);
}

void CodeGlobalMemoryNonRiscV(std::ofstream &ofs, size_t size, bool isStaticAlloc) {
  if (isStaticAlloc) {
    ofs << "void *GlobalMemory(void) {\n" << "    size_t init_size = " << size << ";\n";
    ofs << R"RAW(
    if (mem_block != NULL) {
      return mem_block;
    }
    memset(g_memory_pool, 0, sizeof(g_memory_pool));
    mem_block = &g_mem_block;
    mem_block->occupied = false;
    mem_block->size = init_size;
    mem_block->addr = g_memory_pool + sizeof(MemBlock);
    mem_block->next = NULL;
    return mem_block;
  }
  )RAW";
  } else {
    ofs << "void *GlobalMemory() {\n" << "    size_t init_size = " << size << ";\n";
    ofs << R"RAW(
    bool expected = false;
    while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
      expected = false;
    }
    if (mem_block != NULL) {
      kLock = false;
      return mem_block;
    }
    mem_block = malloc(sizeof(MemBlock) + init_size);
    mem_block->occupied = false;
    mem_block->size = init_size;
    mem_block->addr = (char *)mem_block + sizeof(MemBlock);
    mem_block->next = NULL;
    kLock = false;
    return mem_block;
  }
  )RAW";
  }
}

void CodeGlobalMemoryRiscV(std::ofstream &ofs, size_t size, bool isStaticAlloc) {
  if (isStaticAlloc) {
    ofs << "void *GlobalMemory(void) {\n" << "    size_t init_size = " << size << ";\n";
    ofs << R"RAW(
    if (mem_block != NULL) {
      return mem_block;
    }
    memset(g_memory_pool, 0, sizeof(g_memory_pool));
    mem_block = &g_mem_block;
    mem_block->occupied = false;
    mem_block->size = init_size;
    mem_block->addr = g_memory_pool + sizeof(MemBlock);
    mem_block->next = NULL;
    return mem_block;
  }
  )RAW";
  } else {
    ofs << "void *GlobalMemory() {\n" << "    size_t init_size = " << size << ";\n";
    ofs << R"RAW(
    bool expected = false;
    mem_block = malloc(sizeof(MemBlock) + init_size);
    mem_block->occupied = false;
    mem_block->size = init_size;
    mem_block->addr = (char *)mem_block + sizeof(MemBlock);
    mem_block->next = NULL;
    return mem_block;
  }
  )RAW";
  }
}

void CodeGlobalMemoryContent(std::ofstream &ofs, size_t size, bool isStaticAlloc, bool isRiscv) {
  if (!isRiscv) {
    CodeGlobalMemoryNonRiscV(ofs, size, isStaticAlloc);
  } else {
    CodeGlobalMemoryRiscV(ofs, size, isStaticAlloc);
  }
}

void CodeGlobalMemory(std::ofstream &ofs, size_t size) {
  Configurator *config_ = Configurator::GetInstance();
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  bool isRiscv = (config_->target() == kRiscV);
  CodeGlobalMemoryContent(ofs, size, isStaticAlloc, isRiscv);
}

void CodeMemoryOccupied(std::ofstream &ofs) {
  Configurator *config_ = Configurator::GetInstance();
  bool isStaticAlloc = (config_->allocate_memory_type() == AllocateMemoryType::kStatic);
  if (!isStaticAlloc) {
    if (config_->target() != kRiscV) {
      ofs << R"RAW(
  void *Malloc(size_t size) {
    bool expected = false;
    while (!atomic_compare_exchange_strong(&kLock, &expected, true)) {
      expected = false;
    }
    if (mem_block == NULL) {
      kLock = false;
      return NULL;
    }
    MemBlock *pre = mem_block;
    MemBlock *cur = mem_block;
    MemBlock *find = NULL;
    while (cur != NULL) {
      if (cur->size < size) {
        break;
      }
      if (!cur->occupied) {
        find = cur;
      }
      pre = cur;
      cur = cur->next;
    }
    if (find != NULL) {
      find->occupied = true;
      kLock = false;
      return find;
    }
    MemBlock *block = (MemBlock *)malloc(sizeof(MemBlock) + size);
    block->occupied = true;
    block->size = size;
    block->addr = (char *)block + sizeof(MemBlock);
    block->next = pre->next;
    pre->next = block;
    kLock = false;
    return block;
  }
  )RAW";
    } else {
      ofs << R"RAW(
  void *Malloc(size_t size) {
    bool expected = false;
    MemBlock *pre = mem_block;
    MemBlock *cur = mem_block;
    MemBlock *find = NULL;
    while (cur != NULL) {
      if (cur->size < size) {
        break;
      }
      if (!cur->occupied) {
        find = cur;
      }
      pre = cur;
      cur = cur->next;
    }
    MemBlock *block = (MemBlock *)malloc(sizeof(MemBlock) + size);
    block->occupied = true;
    block->size = size;
    block->addr = (char *)block + sizeof(MemBlock);
    block->next = pre->next;
    pre->next = block;
    return block;
}
  )RAW";
    }
  }
}

void CodeLockOccupied(std::ofstream &ofs, bool isStaticAlloc) {
  Configurator *config_ = Configurator::GetInstance();
  if (config_->target() != kRiscV) {
    ofs << R"RAW(
  bool LockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = false;
    return atomic_compare_exchange_strong(&(m_block->occupied), &expected, true);
  }

  bool UnLockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = true;
    return atomic_compare_exchange_strong(&(m_block->occupied), &expected, false);
  }
  )RAW";
  } else {
    if (isStaticAlloc) {
      ofs << R"RAW(
  bool LockBuffer(void *block) {
    (void)block;
    bool expected = true;
    return expected;
  }

  bool UnLockBuffer(void *block) {
    (void)block;
    bool expected = true;
    return expected;
  }
  )RAW";
    } else {
      ofs << R"RAW(
  bool LockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = true;
    return expected;
  }

  bool UnLockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = true;
    return expected;
  }
  )RAW";
    }
  }
}
}  // namespace mindspore::lite::micro
