cmake_minimum_required(VERSION 3.16...3.27)
project(MCSHighland VERSION 1.0.0 LANGUAGES CXX)

option(WITH_GEANT4_UIVIS "Build with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

include(${Geant4_USE_FILE})

option(WITH_CADMESH "Build with CADMesh STL support" OFF)
if(WITH_CADMESH)
  find_package(cadmesh QUIET)
  if(cadmesh_FOUND)
    add_definitions(-DUSE_CADMESH)
    message(STATUS "CADMesh found - STL geometry import enabled")
  else()
    message(WARNING "CADMesh not found - falling back to voxelised geometry for gyroid/Voronoi")
    set(WITH_CADMESH OFF)
  endif()
endif()

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(MCSHighland MCSHighland.cc ${sources} ${headers})
target_include_directories(MCSHighland PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(MCSHighland ${Geant4_LIBRARIES})

if(WITH_CADMESH AND cadmesh_FOUND)
  target_link_libraries(MCSHighland cadmesh)
endif()

file(GLOB MACRO_FILES ${PROJECT_SOURCE_DIR}/macros/*.mac)
foreach(_macro ${MACRO_FILES})
  get_filename_component(_macro_name ${_macro} NAME)
  configure_file(${_macro} ${PROJECT_BINARY_DIR}/${_macro_name} COPYONLY)
endforeach()

file(GLOB STL_FILES ${PROJECT_SOURCE_DIR}/geometry/*.stl)
foreach(_stl ${STL_FILES})
  get_filename_component(_stl_name ${_stl} NAME)
  configure_file(${_stl} ${PROJECT_BINARY_DIR}/${_stl_name} COPYONLY)
endforeach()

install(TARGETS MCSHighland DESTINATION bin)
