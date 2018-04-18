# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

#############
# Macro IncludeTargetDirectories
#    target - Name of the target
#
# Adds include directories from the target globally via include_directories()
# Necessary for cuda files
macro(IncludeTargetDirectories targets)
    foreach(target ${targets})
        if(TARGET ${target})
            get_property(includes_set TARGET ${target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES SET)
            if(${includes_set})
                get_property(includes TARGET ${target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
                include_directories(${includes})
            endif()
        endif()
    endforeach()
endmacro()
