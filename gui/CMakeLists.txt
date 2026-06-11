set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(KubaGUI
  main.cpp
  MainWindow.h MainWindow.cpp
  BoardWidget.h BoardWidget.cpp
  GameSettingsDialog.h GameSettingsDialog.cpp
)

target_link_libraries(KubaGUI PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets)
add_dependencies(KubaGUI kuba_engine)

if(WIN32)
  set(KAHU_ENGINE_NAME "kuba_engine.exe")
else()
  set(KAHU_ENGINE_NAME "kuba_engine")
endif()

add_custom_command(TARGET KubaGUI POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:kuba_engine>"
    "$<TARGET_FILE_DIR:KubaGUI>/${KAHU_ENGINE_NAME}"
  COMMENT "Copying kuba_engine next to KubaGUI"
)

if(EXISTS "${CMAKE_SOURCE_DIR}/kuba.nnue")
  add_custom_command(TARGET KubaGUI POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/kuba.nnue"
      "$<TARGET_FILE_DIR:KubaGUI>/kuba.nnue"
    COMMENT "Copying kuba.nnue next to KubaGUI"
  )
endif()

