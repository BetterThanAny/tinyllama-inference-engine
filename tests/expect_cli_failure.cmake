if(NOT DEFINED EXECUTABLE OR NOT DEFINED MODEL_DIR OR NOT DEFINED MAX_TOKENS)
  message(FATAL_ERROR "CLI failure test is missing a required argument")
endif()

set(command "${EXECUTABLE}" "${MODEL_DIR}" "test" "${MAX_TOKENS}")
if(DEFINED EXPECTED_TOKEN)
  list(APPEND command "${EXPECTED_TOKEN}")
endif()

execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(result EQUAL 0)
  message(FATAL_ERROR "CLI unexpectedly accepted invalid input: ${stdout}")
endif()

string(CONCAT output "${stdout}" "${stderr}")
if(NOT output MATCHES "\"code\":\"invalid_argument\"")
  message(FATAL_ERROR "CLI did not return structured invalid_argument: ${output}")
endif()
