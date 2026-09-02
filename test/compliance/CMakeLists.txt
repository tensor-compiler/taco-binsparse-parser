#[[
SPDX-FileCopyrightText: 2026 Binsparse Developers

SPDX-License-Identifier: BSD-3-Clause
]]

add_library(npy_vendor STATIC
  ${binsparse_SOURCE_DIR}/test/compliance/npy_c.c
)
target_include_directories(npy_vendor PUBLIC
  ${binsparse_SOURCE_DIR}/test/compliance
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(compliance_core STATIC reformat.c)
target_include_directories(compliance_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(compliance_core PUBLIC binsparse)

add_executable(npy_to_binsparse npy_to_binsparse.c)
target_link_libraries(npy_to_binsparse PRIVATE compliance_core npy_vendor)

add_executable(binsparse_to_npy binsparse_to_npy.cpp)
target_link_libraries(binsparse_to_npy PRIVATE compliance_core npy_vendor
                                                   bsp-to-taco)
set_target_properties(binsparse_to_npy PROPERTIES LINKER_LANGUAGE CXX)

add_executable(binsparse_to_binsparse binsparse_to_binsparse.cpp)
target_link_libraries(binsparse_to_binsparse PRIVATE compliance_core bsp-to-taco)
